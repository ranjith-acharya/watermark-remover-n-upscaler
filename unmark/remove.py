"""Watermark removal engines.

Engines that fill the masked pixels:

  auto      LaMa when PyTorch is present, Telea otherwise - the default
  fast      border interpolation across the masked box (the delogo approach)
  balanced  Telea inpainting of the masked pixels, per frame
  ai        LaMa inpainting on the cropped patch, per frame (needs PyTorch)

`auto` exists because the gap between Telea and LaMa is large and entirely
invisible from the settings screen. Telea propagates colour inward from the
boundary, which holds up over flat or blurred backgrounds and collapses into a
smeared rectangle over structure - a roofline, a collar, a building edge. LaMa
reconstructs those. Making people discover that by trying both, on a control
they have no basis to judge, was a design mistake.

On top of whichever engine runs, an *alpha matte* may be recovered and used to
put back the real pixels instead of invented ones. For a masked pixel,
observed = (1-a)*bg + a*C with a and C constant over the whole clip. A constant
overlay scales the scene's contrast by (1-a), so wherever the video moves, the
temporal variance under the watermark is (1-a)^2 of what it would otherwise be.
Diffusing the surrounding pixels inward estimates the un-marked mean and
variance, which gives alpha from the variance ratio and the colour from the
mean. Inverting the blend then brings the original detail back.

That solve needs a background that actually moves. A dark, flat, low-contrast
scene gives it nothing to work with, and a bad matte looks visibly worse than a
plain inpaint - so it must clear three independent gates before it is applied at
all: how much of the glyph it explains, whether the two halves of the sampled
frames agree on alpha, and whether the recovered patch joins its surroundings
without a seam. Failing any of them discards it and the fill engine stands
alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .detect import Region

ENGINES = ("auto", "fast", "balanced", "ai")

ALPHA_MAX = 0.95            # above this, (1-a) is too small to divide by safely
MIN_SPREAD = 25.0           # background variance needed before alpha is solvable
MAX_CHANNEL_SPREAD = 0.30   # per-channel alphas must agree; they measure one thing
MIN_COVERAGE = 0.25         # matte must explain a real share of the glyph to be used
MIN_SEAM = 0.65             # ...leave no visible seam (see _seam_score)
MAX_ALPHA_DRIFT = 0.06      # ...and give the same alpha from either half
MIN_STABILITY = 0.40        #    of the sampled frames


@dataclass
class Matte:
    alpha: np.ndarray                       # (h, w) float32 in [0, 1]
    color: np.ndarray                       # (h, w, 3) float32 BGR
    reliable: np.ndarray                    # (h, w) bool
    seam: float = 0.0
    stability: float = 0.0
    glyph_px: int = 0                       # masked pixel count, the coverage base

    @property
    def coverage(self) -> float:
        """Share of the *glyph* solved, not of the padded ROI around it."""
        return float(self.reliable.sum()) / max(self.glyph_px, 1)

    @property
    def usable(self) -> bool:
        return (self.coverage >= MIN_COVERAGE and self.seam >= MIN_SEAM
                and self.stability >= MIN_STABILITY)

    def reject_reason(self) -> str:
        if self.coverage < MIN_COVERAGE:
            return (f"matte covered only {self.coverage:.0%} of the glyph "
                    f"(needs {MIN_COVERAGE:.0%}) - background too flat to solve")
        if self.stability < MIN_STABILITY:
            return (f"alpha was not stable across the clip "
                    f"(score {self.stability:.2f}, needs {MIN_STABILITY:.2f}) - "
                    f"noise, not signal, was driving the solve")
        if self.seam < MIN_SEAM:
            return (f"recovered patch left a visible seam "
                    f"(score {self.seam:.2f}, needs {MIN_SEAM:.2f})")
        return ""

    def summary(self) -> dict:
        return {"seam": round(self.seam, 3), "stability": round(self.stability, 3),
                "coverage": round(self.coverage, 3),
                "peak_alpha": round(float(self.alpha.max()), 3),
                "usable": self.usable}


@dataclass
class PreparedRegion:
    """A region resolved into concrete ROI coordinates and blending masks."""
    region: Region
    x: int
    y: int
    w: int
    h: int
    mask: np.ndarray = field(repr=False)      # (h, w) uint8 0/255, dilated glyph
    feather: np.ndarray = field(repr=False)   # (h, w, 1) float32 0..1
    matte: Matte | None = field(default=None, repr=False)
    matte_note: str = ""

    def crop(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.y:self.y + self.h, self.x:self.x + self.w]


def _dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(mask, kernel, iterations=iterations)


def prepare_region(region: Region, frame_w: int, frame_h: int,
                   pad: int | None = None, dilate: int = 3) -> PreparedRegion:
    """Place a region mask into a padded ROI with a feathered blend edge."""
    if pad is None:
        pad = max(6, int(round(0.35 * max(region.w, region.h))))
    x, y, w, h = region.padded(pad, frame_w, frame_h)

    local = np.zeros((h, w), dtype=np.uint8)
    src = region.mask
    if src is None:
        src = np.ones((region.h, region.w), dtype=bool)
    ox, oy = region.x - x, region.y - y
    sub = local[oy:oy + src.shape[0], ox:ox + src.shape[1]]
    sub[...] = np.where(src[:sub.shape[0], :sub.shape[1]], 255, 0)

    mask = _dilate(local, dilate)
    # Feather: fully replace the glyph core, fade out over a few pixels so the
    # patch never shows a hard seam against untouched video.
    blur = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), sigmaX=1.5)
    feather = np.clip(blur, 0.0, 1.0)[:, :, None].astype(np.float32)
    return PreparedRegion(region=region, x=x, y=y, w=w, h=h, mask=mask, feather=feather)


# --------------------------------------------------------------------------- #
# fills
# --------------------------------------------------------------------------- #

def fill_fast(roi: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Inverse-distance interpolation from the ROI border, like the delogo filter."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return roi.astype(np.float32)
    h, w = mask.shape
    out = roi.astype(np.float32).copy()
    left, right = out[:, 0, :], out[:, w - 1, :]
    top, bottom = out[0, :, :], out[h - 1, :, :]

    wl = 1.0 / (xs + 1).astype(np.float32)
    wr = 1.0 / (w - xs).astype(np.float32)
    wt = 1.0 / (ys + 1).astype(np.float32)
    wb = 1.0 / (h - ys).astype(np.float32)
    total = wl + wr + wt + wb

    vals = (left[ys] * wl[:, None] + right[ys] * wr[:, None] +
            top[xs] * wt[:, None] + bottom[xs] * wb[:, None]) / total[:, None]
    out[ys, xs] = vals
    return out


def fill_inpaint(roi: np.ndarray, mask: np.ndarray, radius: int = 4) -> np.ndarray:
    return cv2.inpaint(np.ascontiguousarray(roi.astype(np.uint8)), mask,
                       radius, cv2.INPAINT_TELEA).astype(np.float32)


# --------------------------------------------------------------------------- #
# alpha matte solve
# --------------------------------------------------------------------------- #

def _diffuse_fill(field: np.ndarray, unknown: np.ndarray, iters: int = 96) -> np.ndarray:
    """Extend a float field across the masked area by repeated smoothing.

    cv2.inpaint only takes 8-bit images; the mean and variance fields below are
    floats with a wide range, so they get their own filler.
    """
    out = field.copy()
    hole = unknown[:, :, None] if field.ndim == 3 else unknown
    out[hole.repeat(field.shape[2], axis=2) if field.ndim == 3 else hole] = 0.0
    for _ in range(iters):
        blurred = cv2.blur(out, (5, 5))
        np.copyto(out, blurred, where=hole)
    return out


def _solve(obs: np.ndarray, glyph: np.ndarray):
    """One variance-ratio solve over a stack of ROI frames."""
    mean_obs = obs.mean(axis=0)
    var_obs = obs.var(axis=0)
    mean_bg = _diffuse_fill(mean_obs, glyph)
    var_bg = _diffuse_fill(var_obs, glyph)

    ratio = np.divide(var_obs, var_bg, out=np.ones_like(var_obs), where=var_bg > 1e-3)
    alpha_ch = 1.0 - np.clip(np.sqrt(np.clip(ratio, 0.0, 1.0)), 0.0, 1.0)

    # Alpha is geometric, so the three channels must agree; trust the channels
    # with the most background movement, and treat disagreement as a failure.
    wgt = np.clip(var_bg, 0.0, None) + 1e-6
    alpha = np.clip((alpha_ch * wgt).sum(axis=2) / wgt.sum(axis=2), 0.0, 1.0)
    spread = alpha_ch.max(axis=2) - alpha_ch.min(axis=2)

    safe = np.maximum(alpha[:, :, None], 1e-3)
    color = np.clip((mean_obs - (1.0 - safe) * mean_bg) / safe, 0.0, 255.0)
    return alpha, color, spread, var_bg


def estimate_matte(rois: np.ndarray, mask: np.ndarray) -> Matte:
    """Solve for alpha and watermark colour from a stack of sampled ROIs.

    A constant overlay scales the scene's contrast by (1-a): where the video
    varies over time, the variance under the watermark is (1-a)^2 times what it
    would otherwise be. Estimating the un-marked mean and variance by diffusing
    the surrounding pixels inward gives alpha from the variance ratio and the
    colour from the mean, with no per-frame background proxy to go wrong.

    The same arithmetic runs on two disjoint halves of the sampled frames. A
    solve driven by real signal gives the same alpha from either half; one
    driven by sensor and compression noise does not. That disagreement is what
    separates a matte worth applying from one that would speckle the patch, and
    a raw variance threshold cannot tell them apart on dark footage.

    `rois` is (N, h, w, 3) uint8; `mask` is the 0/255 glyph mask.
    """
    obs = rois.astype(np.float32)
    glyph = mask > 0

    alpha, color, spread, var_bg = _solve(obs, glyph)
    alpha_a, *_ = _solve(obs[0::2], glyph)
    alpha_b, *_ = _solve(obs[1::2], glyph)
    drift = np.abs(alpha_a - alpha_b)

    reliable = (glyph
                & (alpha > 0.02) & (alpha < ALPHA_MAX)
                & (spread < MAX_CHANNEL_SPREAD)
                & (drift < MAX_ALPHA_DRIFT)
                & (var_bg.mean(axis=2) > MIN_SPREAD))

    stability = 0.0
    if reliable.any():
        stability = float(np.clip(
            1.0 - np.median(drift[reliable]) / MAX_ALPHA_DRIFT, 0.0, 1.0))

    matte = Matte(alpha=alpha.astype(np.float32), color=color.astype(np.float32),
                  reliable=reliable, seam=0.0, stability=stability,
                  glyph_px=int(glyph.sum()))
    matte.seam = _seam_score(obs, mask, matte)
    return matte


def _seam_score(obs: np.ndarray, mask: np.ndarray, matte: Matte) -> float:
    """Independent check: does the recovered patch join its surroundings cleanly?

    A wrong alpha or colour leaves a step at the mask boundary. This measures
    image gradient on the boundary ring and compares it against gradient in
    untouched pixels a little further out - a check the solve was never fitted
    to. Deliberately local: unblending divides by (1-a), which amplifies
    compression noise everywhere inside the glyph, so judging the interior's
    overall roughness would condemn a perfectly good matte.

    1.0 means the boundary is indistinguishable from ordinary image content;
    0.0 means it is twice as sharp, i.e. a visible seam.
    """
    if matte.reliable.sum() < 12:
        return 0.0
    m = (mask > 0).astype(np.uint8)
    k3 = np.ones((3, 3), np.uint8)
    boundary = (cv2.dilate(m, k3) > 0) & (cv2.erode(m, k3) == 0)
    reference = (cv2.dilate(m, np.ones((11, 11), np.uint8)) > 0) & (cv2.dilate(m, k3) == 0)
    if boundary.sum() < 8 or reference.sum() < 8:
        return 0.0

    ratios = []
    for frame in obs[:: max(1, len(obs) // 12)]:
        patched = np.where(matte.reliable[:, :, None], unblend(frame, matte),
                           frame.astype(np.float32))
        gray = patched.mean(axis=2)
        grad = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
                        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        ref = float(grad[reference].mean())
        if ref > 1e-3:
            ratios.append(float(grad[boundary].mean()) / ref)
    if not ratios:
        return 0.0
    return float(np.clip(2.0 - float(np.median(ratios)), 0.0, 1.0))


def unblend(roi: np.ndarray, matte: Matte) -> np.ndarray:
    """Invert the alpha blend, recovering the pixels the watermark sat on."""
    obs = roi.astype(np.float32)
    a = matte.alpha[:, :, None]
    return np.clip((obs - a * matte.color) / np.maximum(1.0 - a, 1e-3), 0.0, 255.0)


# --------------------------------------------------------------------------- #
# remover
# --------------------------------------------------------------------------- #

class Remover:
    """Applies prepared regions to frames with the chosen engine."""

    def __init__(self, regions: list[PreparedRegion], engine: str = "auto",
                 lama=None):
        if engine not in ENGINES:
            raise ValueError(f"unknown engine {engine!r}; expected one of {ENGINES}")
        if engine == "auto":
            # Resolved by the pipeline, which knows whether the model loaded.
            engine = "ai" if lama is not None else "balanced"
        if engine == "ai" and lama is None:
            engine = "balanced"          # caller could not load the model
        self.regions = regions
        self.engine = engine
        self.lama = lama

    def _fill(self, roi: np.ndarray, prep: PreparedRegion) -> np.ndarray:
        if self.engine == "fast":
            return fill_fast(roi, prep.mask)
        if self.engine == "ai":
            return self.lama.inpaint(roi, prep.mask)
        return fill_inpaint(roi, prep.mask)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        out = frame
        for prep in self.regions:
            roi = prep.crop(out)
            filled = self._fill(roi, prep)
            if prep.matte is not None:
                # Verified matte: prefer recovered real pixels over invented ones.
                filled = np.where(prep.matte.reliable[:, :, None],
                                  unblend(roi, prep.matte), filled)
            blended = roi.astype(np.float32) * (1.0 - prep.feather) + filled * prep.feather
            if out is frame:
                out = frame.copy()
            out[prep.y:prep.y + prep.h, prep.x:prep.x + prep.w] = \
                np.clip(blended, 0, 255).astype(np.uint8)
        return out
