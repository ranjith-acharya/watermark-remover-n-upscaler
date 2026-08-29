"""Automatic detection of static overlay watermarks.

A watermark is a small, spatially-fixed overlay that persists while the scene
behind it changes. The detector exploits exactly that: high-pass each sampled
frame to strip the low-frequency scene, then look for pixels whose response
stays strong in nearly every frame. Moving scene edges score low because they
only light up a given pixel some of the time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .ffmpegio import VideoInfo, probe, read_gray_frames

# Measured from Google Flow / Veo output (the sparkle glyph), as fractions of
# frame size. Used as a fallback when detection is not confident.
FLOW_PRESET = (0.8000, 0.8875, 0.8667, 0.9250)  # x0, y0, x1, y1

MIN_ISOLATION = 0.90    # how quiet the ring around a candidate must be
DOMINANCE = 0.90        # secondary regions must score this share of the best
MAX_ANALYSIS_DIM = 1280
MAX_SAMPLES = 64


@dataclass
class Region:
    """A detected watermark, in coordinates of the *original* video."""
    x: int
    y: int
    w: int
    h: int
    confidence: float
    source: str = "detected"
    mask: np.ndarray | None = field(default=None, repr=False)

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    def padded(self, pad: int, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
        x0 = max(0, self.x - pad)
        y0 = max(0, self.y - pad)
        x1 = min(frame_w, self.x + self.w + pad)
        y1 = min(frame_h, self.y + self.h + pad)
        return x0, y0, x1 - x0, y1 - y0

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h,
                "confidence": round(float(self.confidence), 3), "source": self.source}


@dataclass
class Detection:
    regions: list[Region]
    persistence: np.ndarray | None = field(default=None, repr=False)

    @property
    def found(self) -> bool:
        return any(r.source == "detected" for r in self.regions)


def _high_pass(frame: np.ndarray, radius: int) -> np.ndarray:
    k = 2 * radius + 1
    f = frame.astype(np.float32)
    return f - cv2.blur(f, (k, k))


def analyse(frames: np.ndarray, radius: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Return (persistence, strength) maps for a (N, H, W) stack.

    persistence[y, x] is the fraction of frames where the pixel carries a strong
    high-pass response; strength is the temporal median of that response (signed,
    so a logo darker than its background is still found).
    """
    if frames.ndim != 3 or len(frames) < 4:
        raise ValueError("need a stack of at least 4 frames")

    hp = np.stack([_high_pass(f, radius) for f in frames])

    # Robust noise floor of the high-pass response across the whole clip.
    sigma = 1.4826 * float(np.median(np.abs(hp))) or 1.0
    tau = max(3.0 * sigma, 2.0)

    persistence = (np.abs(hp) > tau).mean(axis=0)
    strength = np.median(hp, axis=0)
    return persistence, strength


def _fill_holes(binary: np.ndarray) -> np.ndarray:
    """Close the interior of a glyph outline.

    The high-pass response peaks on a glyph's edges and nearly vanishes across
    its flat interior, so a raw threshold returns a ring. Inpainting a ring
    leaves the solid centre behind, which looks worse than the original — so the
    enclosed area has to be filled back in before the mask is usable.
    """
    img = (binary.astype(np.uint8) * 255)
    padded = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    ff_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    filled = cv2.bitwise_or(padded, cv2.bitwise_not(flood))
    return filled[1:-1, 1:-1] > 0


GROW = 3       # px added around the detected core, to catch the anti-aliased fringe


def _solidify(blob: np.ndarray, grow: int = GROW) -> np.ndarray:
    """Turn a detected outline into the solid shape it encloses.

    The threshold only keeps the glyph's strong core, so the soft fringe and any
    tapering points fall outside it. Growing the result covers them; leaving them
    behind would show as a faint ghost of the mark after removal.
    """
    padded = np.pad(blob, grow, mode="constant", constant_values=False)
    closed = cv2.morphologyEx(padded.astype(np.uint8), cv2.MORPH_CLOSE,
                              np.ones((5, 5), np.uint8))
    solid = _fill_holes(closed.astype(bool))
    # A shape whose outline never closed stays hollow after filling; the convex
    # hull is the safe answer there, at the cost of covering a little extra.
    if solid.mean() < 0.35:
        pts = cv2.findNonZero(closed)
        if pts is not None and len(pts) >= 3:
            hull = np.zeros_like(closed)
            cv2.fillConvexPoly(hull, cv2.convexHull(pts), 1)
            solid = hull > 0
    return cv2.dilate(solid.astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                      iterations=grow).astype(bool)


def _isolation(seed: np.ndarray, x: int, y: int, bw: int, bh: int) -> float:
    """Fraction of the ring around a candidate that is *not* also persistent.

    1.0 means the response stops dead at the box - an overlay sitting on top of
    the picture. Low values mean it carries on into the surroundings, which is
    what scene geometry does.
    """
    H, W = seed.shape
    mx, my = max(4, bw // 2), max(4, bh // 2)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(W, x + bw + mx), min(H, y + bh + my)

    window = seed[y0:y1, x0:x1]
    ring = np.ones(window.shape, dtype=bool)
    ring[y - y0:y - y0 + bh, x - x0:x - x0 + bw] = False
    if ring.sum() < 8:
        return 0.0
    return 1.0 - float((window[ring] > 0).mean())


def _flow_prior(x0: int, y0: int, x1: int, y1: int, W: int, H: int) -> float:
    """How closely a candidate matches Flow's sparkle geometry. 1.0 = exact."""
    px0, py0, px1, py1 = FLOW_PRESET
    ref = np.array([px0 * W, py0 * H, px1 * W, py1 * H])
    got = np.array([x0, y0, x1, y1])
    tol = 0.05 * max(W, H)
    err = float(np.abs(got - ref).max())
    return max(0.0, 1.0 - err / tol)


def _candidates(persistence: np.ndarray, strength: np.ndarray,
                min_persistence: float) -> list[dict]:
    H, W = persistence.shape
    strong = np.abs(strength)
    # Threshold against the noise floor, never against a percentile of the frame:
    # one very bright static object would otherwise drag a percentile up until the
    # real watermark fell below it and broke into fragments.
    sigma = 1.4826 * float(np.median(strong)) or 1.0
    seed = ((persistence >= min_persistence) & (strong >= max(2.0, 4.0 * sigma))
            ).astype(np.uint8)
    if seed.sum() == 0:
        return []

    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(seed, connectivity=8)

    frame_area = W * H
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                              stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT],
                              stats[i, cv2.CC_STAT_AREA])
        if area < 16 or area > 0.02 * frame_area:
            continue
        if bw > 0.25 * W or bh > 0.25 * H:
            continue
        if area / float(bw * bh) < 0.20:          # too straggly to be a glyph
            continue
        edge_dist = min(x, y, W - (x + bw), H - (y + bh))
        if edge_dist > 0.30 * min(W, H):          # watermarks hug an edge
            continue

        blob = labels[y:y + bh, x:x + bw] == i
        persist_mean = float(persistence[y:y + bh, x:x + bw][blob].mean())
        strength_mean = float(strong[y:y + bh, x:x + bw][blob].mean())

        # A watermark is an isolated island: the persistent response stops at its
        # edges. A static scene feature - a girder, a doorway, a painted line -
        # is just as persistent, but goes on past the box the threshold cut it
        # down to. Sampling the ring around the box tells the two apart.
        isolation = _isolation(seed, x, y, bw, bh)
        if isolation < MIN_ISOLATION:
            continue

        score = persist_mean * min(1.0, strength_mean / 8.0) * isolation
        if min(x, W - (x + bw)) < 0.25 * W and min(y, H - (y + bh)) < 0.25 * H:
            score *= 1.2                          # corner placement is typical
        score *= 1.0 + 0.6 * _flow_prior(x, y, x + bw, y + bh, W, H)

        # _solidify pads the blob by GROW on every side, so the box grows with it.
        # Trim whatever falls outside the frame.
        grown = _solidify(blob)
        gx, gy = max(0, x - GROW), max(0, y - GROW)
        grown = grown[(y - GROW < 0) * (GROW - y):, (x - GROW < 0) * (GROW - x):]
        gh, gw = min(grown.shape[0], H - gy), min(grown.shape[1], W - gx)
        out.append({"x": int(gx), "y": int(gy), "w": int(gw), "h": int(gh),
                    "score": min(1.0, score), "mask": grown[:gh, :gw]})

    out.sort(key=lambda c: c["score"], reverse=True)
    return out


def preset_region(width: int, height: int) -> Region:
    x0, y0, x1, y1 = FLOW_PRESET
    x, y = int(round(x0 * width)), int(round(y0 * height))
    w, h = int(round(x1 * width)) - x, int(round(y1 * height)) - y
    mask = np.ones((h, w), dtype=bool)
    return Region(x, y, w, h, confidence=0.0, source="preset", mask=mask)


def detect_in_frames(frames: np.ndarray, scale: float = 1.0, radius: int = 12,
                     min_persistence: float = 0.80, min_confidence: float = 0.30,
                     max_regions: int = 3) -> Detection:
    persistence, strength = analyse(frames, radius=radius)
    regions: list[Region] = []
    candidates = _candidates(persistence, strength, min_persistence)
    # One generator renders all of its marks the same way, so a genuine second
    # watermark scores close to the first. A scene feature that merely survived
    # the filters does not - it trails the real mark by a wide margin.
    if candidates:
        floor = max(min_confidence, DOMINANCE * candidates[0]["score"])
        candidates = [c for c in candidates if c["score"] >= floor]

    for cand in candidates[:max_regions]:
        inv = 1.0 / scale
        mask = cand["mask"]
        if scale != 1.0:
            mask = cv2.resize(mask.astype(np.uint8),
                              (max(1, int(round(cand["w"] * inv))),
                               max(1, int(round(cand["h"] * inv)))),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        regions.append(Region(
            x=int(round(cand["x"] * inv)), y=int(round(cand["y"] * inv)),
            w=mask.shape[1], h=mask.shape[0],
            confidence=cand["score"], source="detected", mask=mask))
    return Detection(regions=regions, persistence=persistence)


def detect(path: str, info: VideoInfo | None = None,
           max_samples: int = MAX_SAMPLES, **kwargs) -> Detection:
    """Detect watermarks in a video file, falling back to the Flow preset."""
    info = info or probe(path)

    scale = min(1.0, MAX_ANALYSIS_DIM / max(info.width, info.height))
    size = None
    if scale < 1.0:
        size = (int(round(info.width * scale)) // 2 * 2,
                int(round(info.height * scale)) // 2 * 2)
        scale = size[0] / info.width

    stride = max(1, info.n_frames // max_samples) if info.n_frames else 1
    frames = read_gray_frames(path, stride=stride, limit=max_samples, scale=size)

    radius = max(6, int(round(12 * (frames.shape[2] / 720.0))))
    det = detect_in_frames(frames, scale=scale, radius=radius, **kwargs)
    if not det.regions:
        det.regions = [preset_region(info.width, info.height)]
    return det
