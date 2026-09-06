"""The Google Flow sparkle, measured once and then inverted exactly.

Flow composites its sparkle onto stills with a *screen* blend of a fixed,
neutral, soft-edged glyph:

    obs = 1 - (1 - scene)(1 - c)

so removal is the algebraic inverse rather than inpainting - the original
pixels come back instead of being invented. Two other models were measured
against 16 real samples and rejected: alpha-over-white leaves a dark blotch
where the background is dark, and a constant additive offset leaves a dark
star where it is bright. Screen is the only one that holds at both extremes.

The glyph is stored as a coefficient map and *located* by matching, not read
from a hard-coded box. Every sample available was 1376x768, so whether Flow
places the mark at a fixed pixel offset or a fixed fraction of the frame is
genuinely unknown - and a box that assumes wrong would scrub a corner that was
never marked. Matching costs little and answers the question per image.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .detect import detect_in_frames
from .remove import _diffuse_fill, prepare_region

MODELS = Path(__file__).resolve().parent.parent / "models"
GLYPH_FILE = MODELS / "flow_glyph.npz"

C_MIN, C_MAX = -0.25, 0.98   # negative captures the soft dark halo round the star
REFINE_MIN = 0.35            # a match this good may move the box off the prediction
SEARCH_RADIUS = 28           # how far off the prediction a refinement may look
DARK_MAX = 70.0              # polish only where the scene is dark enough to ghost
SCORE_WEIGHT = 0.44          # how far template agreement counts toward confidence
PAD = 22                     # ROI margin round the glyph, in glyph pixels
DILATE = 6


@dataclass
class Glyph:
    """A screen-blend coefficient map for one watermark, plus where it sits."""
    c: np.ndarray                  # (h, w) float32, the screen coefficient
    margin_x: int = 0              # ROI origin measured back from the right edge
    margin_y: int = 0              # ...and up from the bottom edge
    samples: int = 0               # how many images it was measured from

    @property
    def size(self) -> tuple[int, int]:
        return self.c.shape[1], self.c.shape[0]

    def save(self, path: Path | str = GLYPH_FILE) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, c=self.c.astype(np.float32), samples=self.samples,
                            margin_x=self.margin_x, margin_y=self.margin_y)
        return path

    @classmethod
    def load(cls, path: Path | str = GLYPH_FILE) -> "Glyph":
        data = np.load(Path(path))
        return cls(c=data["c"].astype(np.float32),
                   margin_x=int(data["margin_x"]), margin_y=int(data["margin_y"]),
                   samples=int(data["samples"]))


@dataclass
class Match:
    """Where a glyph sits in an image, and how convinced we are."""
    x: int
    y: int
    w: int
    h: int
    score: float                   # template agreement, 0..1
    energy: float                  # how much smoother unscreening makes it
    refined: bool = False          # True when a match moved it off the prediction

    @property
    def confidence(self) -> float:
        """Energy leads; template agreement is only corroboration.

        Ordinary scene texture correlates with the glyph shape well enough to
        score around 0.3 on a picture that carries no mark at all, so the
        template alone cannot be trusted to answer "is it there". The energy
        test can - it asks whether the inverse actually improves the region -
        so the score is discounted to the point where it can support a find
        but never carry one on its own.
        """
        return max(0.0, min(1.0, max(self.energy, self.score * SCORE_WEIGHT)))

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h,
                "score": round(float(self.score), 3),
                "energy": round(float(self.energy), 3),
                "confidence": round(self.confidence, 3), "refined": self.refined}


def read_image(path: Path | str) -> np.ndarray:
    """Load a BGR image.

    Goes through ``np.fromfile`` because ``cv2.imread`` silently returns None
    for non-ASCII paths on Windows, and generated filenames are full of them.
    """
    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not decode image: {path}")
    return image


def write_image(path: Path | str, image: np.ndarray, quality: int = 95) -> Path:
    """Save a BGR image, choosing the encoder from the extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".png"
    params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, image, params)
    if not ok:
        raise ValueError(f"could not encode image as {ext}")
    buf.tofile(str(path))
    return path


def _high_pass(gray: np.ndarray, radius: int = 12) -> np.ndarray:
    blurred = cv2.blur(gray.astype(np.float32), (radius * 2 + 1, radius * 2 + 1))
    return gray.astype(np.float32) - blurred


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #

def calibrate(paths: list[Path | str]) -> Glyph:
    """Measure the screen coefficient map from images that share one watermark.

    Several different scenes carrying the same mark are exactly the frame stack
    the video detector wants, so it finds the glyph here unchanged. The clean
    scene under the mark is then estimated by diffusing the surrounding pixels
    inward, and the per-pixel coefficient read off as the median across images -
    a median because a few samples always have texture the diffusion cannot
    predict, and one bad estimate should not move the answer.
    """
    if len(paths) < 3:
        raise ValueError("calibration needs at least 3 images sharing one watermark")

    images = [read_image(p).astype(np.float32) for p in paths]
    shapes = {im.shape for im in images}
    if len(shapes) != 1:
        raise ValueError("calibration images must all be the same size")

    gray = np.stack([cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2GRAY) for im in images])
    found = detect_in_frames(gray)
    if not found.found:
        raise ValueError("no watermark found across the calibration images")

    height, width = gray.shape[1:]
    prep = prepare_region(found.regions[0], width, height, pad=PAD, dilate=DILATE)
    unknown = prep.mask > 0

    obs = np.stack([prep.crop(im) for im in images]) / 255.0
    clean = np.stack([
        np.dstack([_diffuse_fill(prep.crop(im)[:, :, ch], unknown, iters=400)
                   for ch in range(3)])
        for im in images
    ]) / 255.0

    # screen:  obs = 1 - (1 - clean)(1 - c)   ->   c = 1 - (1 - obs)/(1 - clean)
    coeff = 1.0 - (1.0 - obs) / np.maximum(1.0 - clean, 1e-3)
    coeff = np.clip(coeff, C_MIN, C_MAX)
    c_map = np.median(coeff, axis=0).mean(axis=2).astype(np.float32)  # the glyph is neutral
    c_map[~unknown] = 0.0
    return Glyph(c=c_map, margin_x=width - prep.x, margin_y=height - prep.y,
                 samples=len(images))


# --------------------------------------------------------------------------- #
# locating and removing
# --------------------------------------------------------------------------- #

def energy_score(image: np.ndarray, glyph: Glyph, x: int, y: int) -> float:
    """How much smoother unscreening at (x, y) makes the region.

    A real mark's edges cancel and the local high-frequency energy drops. Get
    the position wrong and the inverse *stamps a star-shaped distortion in*,
    pushing the score negative. It is the one measure that tests the model
    rather than the picture.
    """
    h, w = glyph.c.shape
    roi = image[y:y + h, x:x + w]
    if roi.shape[:2] != (h, w):
        return -1.0
    sel = glyph.c > 0.05
    if not sel.any():
        return -1.0

    before = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    norm = roi.astype(np.float32) / 255.0
    recovered = np.clip(1.0 - (1.0 - norm) / np.maximum(1.0 - glyph.c[:, :, None], 1e-3), 0, 1)
    after = cv2.cvtColor((recovered * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)

    e_before = float(np.abs(_high_pass(before, 6))[sel].mean())
    e_after = float(np.abs(_high_pass(after, 6))[sel].mean())
    return 1.0 - e_after / max(e_before, 1e-6)


def locate(image: np.ndarray, glyph: Glyph, refine: bool = True) -> Match:
    """Work out where the glyph sits in this image.

    Flow's placement is deterministic, so the stored margin *predicts* the spot
    and a template match only ever refines it - and only when the match is
    convincing. That ordering matters: searching first looked reasonable and
    scored 11/16 on the samples, because on a bright scene the mark's amplitude
    is ``c * (1 - background)``, which shrinks toward nothing and loses to
    ordinary scene texture elsewhere in the frame. Predicting first is 16/16,
    and a weak match is then evidence about the *mark*, not about the position.
    """
    h, w = glyph.c.shape
    height, width = image.shape[:2]
    px = max(0, min(width - w, width - glyph.margin_x))
    py = max(0, min(height - h, height - glyph.margin_y))

    x, y, score, refined = px, py, 0.0, False
    if refine and width >= w and height >= h:
        rad = SEARCH_RADIUS
        x0, y0 = max(0, px - rad), max(0, py - rad)
        x1, y1 = min(width, px + w + rad), min(height, py + h + rad)
        field = _high_pass(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))[y0:y1, x0:x1]
        if field.shape[0] >= h and field.shape[1] >= w:
            response = cv2.matchTemplate(field, glyph.c, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(response)
            score = float(score)
            if score >= REFINE_MIN:
                x, y, refined = int(loc[0]) + x0, int(loc[1]) + y0, True

    return Match(x=x, y=y, w=w, h=h, score=score,
                 energy=energy_score(image, glyph, x, y), refined=refined)


def unscreen(image: np.ndarray, glyph: Glyph, match: Match) -> np.ndarray:
    """Invert the screen blend where the glyph sits, leaving the rest untouched."""
    out = image.astype(np.float32) / 255.0
    c = glyph.c
    if (match.w, match.h) != (c.shape[1], c.shape[0]):
        c = cv2.resize(c, (match.w, match.h), interpolation=cv2.INTER_LINEAR)

    roi = out[match.y:match.y + match.h, match.x:match.x + match.w]
    c = c[:roi.shape[0], :roi.shape[1], None]
    recovered = 1.0 - (1.0 - roi) / np.maximum(1.0 - c, 1e-3)
    roi[...] = np.clip(recovered, 0.0, 1.0)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def residual_mask(before: np.ndarray, after: np.ndarray, match: Match,
                  glyph: Glyph, threshold: float = 6.0) -> np.ndarray:
    """Pixels where the inverse did not fully settle, for an optional polish pass.

    The inverse is exact in arithmetic but not in practice: JPEG has already
    quantised the marked pixels, and on a very dark background the recovered
    value is a small difference between two small numbers. What survives is a
    faint ghost, and this is where it lives.
    """
    c = glyph.c
    if (match.w, match.h) != (c.shape[1], c.shape[0]):
        c = cv2.resize(c, (match.w, match.h), interpolation=cv2.INTER_LINEAR)
    region = after[match.y:match.y + match.h, match.x:match.x + match.w]
    c = c[:region.shape[0], :region.shape[1]]

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    ghost = np.abs(_high_pass(gray, radius=6))
    # Only where the scene is dark. Screen adds c*(1 - scene), so on a bright
    # background the mark barely registers and the inverse is already clean -
    # inpainting there replaces good pixels with a blur blob, which is worse
    # than the ghost it was sent to fix.
    dark = cv2.blur(gray.astype(np.float32), (25, 25)) < DARK_MAX
    mask = ((ghost > threshold) & (c > 0.02) & dark).astype(np.uint8) * 255
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
