"""Synthetic fixtures.

Real footage cannot be asserted against: there is no ground truth for what sits
behind the watermark. So the tests build clips where the clean frames, the glyph
shape, its alpha and its colour are all known, then check that the pipeline
recovers them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unmark import ffmpegio  # noqa: E402

WIDTH, HEIGHT, FRAMES, FPS = 320, 480, 60, 24.0
GLYPH_BOX = (232, 392, 40, 40)      # x, y, w, h - bottom right, like Flow
GLYPH_ALPHA = 0.55
GLYPH_COLOR = (210.0, 210.0, 205.0)  # BGR


def diamond_mask(w: int, h: int) -> np.ndarray:
    """A four-pointed star, close enough in shape to the Flow sparkle."""
    yy, xx = np.mgrid[0:h, 0:w]
    nx = (xx - (w - 1) / 2) / ((w - 1) / 2)
    ny = (yy - (h - 1) / 2) / ((h - 1) / 2)
    return (np.abs(nx) ** 0.6 + np.abs(ny) ** 0.6) <= 1.0


def clean_frame(i: int, width: int = WIDTH, height: int = HEIGHT,
                busy: bool = True) -> np.ndarray:
    """A frame whose content moves, so a static overlay is separable from it."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    phase = i * 0.21
    if busy:
        base = (110
                + 70 * np.sin(xx / 23.0 + phase * 3.0)
                + 55 * np.cos(yy / 17.0 - phase * 2.0)
                + 40 * np.sin((xx + yy) / 31.0 + phase))
    else:
        base = np.full((height, width), 30.0) + 2.0 * np.sin(phase)
    frame = np.stack([base, base * 0.92 + 12, base * 0.85 + 24], axis=2)
    return np.clip(frame, 0, 255).astype(np.uint8)


def blend_glyph(frame: np.ndarray, box=GLYPH_BOX, alpha: float = GLYPH_ALPHA,
                color=GLYPH_COLOR) -> np.ndarray:
    x, y, w, h = box
    out = frame.copy()
    shape = diamond_mask(w, h)[:, :, None].astype(np.float32)
    a = alpha * shape
    roi = out[y:y + h, x:x + w].astype(np.float32)
    out[y:y + h, x:x + w] = np.clip(
        (1 - a) * roi + a * np.array(color, dtype=np.float32), 0, 255).astype(np.uint8)
    return out


def _write_clip(path: Path, frames) -> Path:
    first = next(iter(frames))
    proc = ffmpegio.open_encoder(path, first.shape[1], first.shape[0], FPS,
                                 encoder="libx264", quality=14)
    proc.stdin.write(np.ascontiguousarray(first).tobytes())
    for frame in frames:
        proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    proc.stdin.close()
    assert proc.wait() == 0, proc.stderr.read().decode("utf-8", "replace")
    return path


@pytest.fixture(scope="session")
def marked_clip(tmp_path_factory) -> Path:
    """A moving scene with a known watermark blended on top."""
    path = tmp_path_factory.mktemp("clips") / "marked.mp4"
    return _write_clip(path, (blend_glyph(clean_frame(i)) for i in range(FRAMES)))


@pytest.fixture(scope="session")
def flat_marked_clip(tmp_path_factory) -> Path:
    """Same watermark over a near-static, near-flat scene - the matte cannot solve."""
    path = tmp_path_factory.mktemp("clips") / "flat.mp4"
    return _write_clip(path, (blend_glyph(clean_frame(i, busy=False))
                              for i in range(FRAMES)))


@pytest.fixture(scope="session")
def clean_clip(tmp_path_factory) -> Path:
    """The same scene with no watermark at all."""
    path = tmp_path_factory.mktemp("clips") / "clean.mp4"
    return _write_clip(path, (clean_frame(i) for i in range(FRAMES)))
