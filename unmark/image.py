"""Watermark removal and upscaling for still images.

The video path has to guess what was under a watermark, because an opaque
overlay destroys the pixels it covers. Flow's still-image sparkle does not:
it is a screen blend, and screen is invertible, so this path *recovers* the
original picture instead of inventing a plausible one. See `glyph` for how the
coefficient map was measured and why the two competing models were rejected.

Everything downstream - the upscaler, the inpainting engines - is the same code
the video pipeline uses. Only the reading, the removal and the writing differ.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import upscale as up
from .glyph import Glyph, Match, locate, read_image, residual_mask, unscreen, write_image
from .pipeline import Options, ProgressFn, _noop

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

MIN_CONFIDENCE = 0.20   # below this we say so rather than scrub a clean corner
POLISH_GHOST = 6.0      # residual high-pass level worth a second pass


def is_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_SUFFIXES


@dataclass
class ImageResult:
    output: str
    info: dict
    match: dict | None = None
    plan: dict = field(default_factory=dict)
    removed: bool = False
    polished: bool = False
    reason: str = ""
    upscaler: str = ""


def default_output(input_path: str | Path, options: Options,
                   out_dir: str | Path = "output", fmt: str = "png") -> Path:
    stem = Path(input_path).stem
    bits = []
    if options.remove:
        bits.append("clean")
    if options.target != "off":
        bits.append(options.target)
    suffix = "_".join(bits) or "out"
    ext = fmt if fmt.startswith(".") else f".{fmt}"
    return Path(out_dir) / f"{stem}_{suffix}{ext}"


def remove_watermark(image: np.ndarray, glyph: Glyph, polish: bool = True,
                     engine: str = "auto", lama=None) -> tuple[np.ndarray, Match, bool]:
    """Invert the screen blend, then optionally clean up what it could not.

    The inverse is exact in arithmetic, but the marked pixels reached us through
    JPEG, and on a very dark background the recovered value is a small
    difference between two small numbers. A faint ghost survives there, and only
    there - so the polish pass repaints the pixels that still stand out rather
    than the whole glyph, which would throw away the recovery we just made.
    """
    match = locate(image, glyph)
    cleaned = unscreen(image, glyph, match)

    polished = False
    if polish:
        ghost = residual_mask(image, cleaned, match, glyph, threshold=POLISH_GHOST)
        if int((ghost > 0).sum()) > 8:
            from .detect import Region
            from .remove import Remover, prepare_region
            region = Region(x=match.x, y=match.y, w=match.w, h=match.h,
                            confidence=match.confidence, source="flow-glyph",
                            mask=ghost > 0)
            prep = prepare_region(region, image.shape[1], image.shape[0], pad=4, dilate=1)
            cleaned = Remover([prep], engine, lama=lama).apply(cleaned)
            polished = True

    return cleaned, match, polished


def run(input_path: str | Path, output_path: str | Path,
        options: Options | None = None, on_progress: ProgressFn | None = None,
        glyph: Glyph | None = None, polish: bool = True, jpeg_quality: int = 95,
        lama=None) -> ImageResult:
    """Clean and/or upscale one still image."""
    options = options or Options()
    progress = on_progress or _noop

    progress("read", 0.05, "Reading image")
    image = read_image(input_path)
    height, width = image.shape[:2]
    info = {"width": width, "height": height}

    match: Match | None = None
    removed = polished = False
    reason = ""

    if options.remove:
        progress("detect", 0.15, "Looking for the Flow sparkle")
        glyph = glyph or Glyph.load()
        probe = locate(image, glyph)
        if probe.confidence < MIN_CONFIDENCE:
            reason = (f"no Flow watermark found (confidence {probe.confidence:.2f}); "
                      "the image was left as it is")
            match = probe
        else:
            progress("remove", 0.35, "Reversing the screen blend")
            image, match, polished = remove_watermark(image, glyph, polish=polish,
                                                      engine=options.engine, lama=lama)
            removed = True
    else:
        reason = "removal was not requested"

    plan = up.plan_upscale(width, height, options.target, options.upscale_mode)
    if plan.changes_size:
        progress("upscale", 0.55, f"Upscaling to {plan.out_w}x{plan.out_h} ({plan.mode})")
        if plan.mode == "ai":
            image = up.RealESRGAN(options.model).upscale(image)
        image = up.resize_to(image, plan.out_w, plan.out_h)

    progress("write", 0.9, "Writing image")
    out = write_image(output_path, image, quality=jpeg_quality)
    progress("done", 1.0, "Finished")

    return ImageResult(output=str(out), info=info,
                       match=match.to_dict() if match else None,
                       plan=asdict(plan),
                       removed=removed, polished=polished, reason=reason,
                       upscaler=plan.mode if plan.changes_size else "")
