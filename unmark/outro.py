"""Detection of branded end cards.

Many tools append a short outro to their exports - a logo on a plain or frozen
background, spliced on after the real footage. It leaves three signatures at
once, and requiring all three is what keeps a legitimately quiet final shot from
being mistaken for one:

  a hard cut     the join is the sharpest frame-to-frame change in the clip
  a frozen tail  what follows barely moves compared with the rest
  a short tail   it lasts seconds, not minutes

Any one of those alone is common in ordinary footage. Together they are not.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ffmpegio import VideoInfo, probe, read_gray_frames

ANALYSIS_WIDTH = 160
MAX_OUTRO_SECONDS = 10.0
MIN_OUTRO_SECONDS = 0.3
SEARCH_FRACTION = 0.30      # only look for the join in the last part of the clip
CUT_RATIO = 3.0             # the join must stand out this far above typical motion
STILL_RATIO = 0.25          # the tail must be this much calmer than the clip
STILL_ABSOLUTE = 4.0        # ...and quiet in absolute terms too


@dataclass
class Outro:
    start_frame: int
    start_time: float
    frames: int
    seconds: float
    confidence: float
    reason: str = ""

    def to_dict(self) -> dict:
        return {"start_frame": self.start_frame,
                "start_time": round(self.start_time, 2),
                "frames": self.frames,
                "seconds": round(self.seconds, 2),
                "confidence": round(self.confidence, 3),
                "reason": self.reason}


def detect_outro(path: str, info: VideoInfo | None = None) -> Outro | None:
    """Find a trailing end card, or None when the clip just ends."""
    info = info or probe(path)
    if info.n_frames < 30 or info.fps <= 0:
        return None

    height = max(2, int(round(ANALYSIS_WIDTH * info.height / info.width)) // 2 * 2)
    frames = read_gray_frames(path, scale=(ANALYSIS_WIDTH, height)).astype(np.float32)
    if len(frames) < 30:
        return None

    # diffs[i] is the change between frame i and i+1, so a cut *into* frame i+1
    # shows up at index i.
    diffs = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))
    typical = float(np.median(diffs))
    if typical <= 0:
        typical = 1e-3

    n = len(frames)
    earliest = max(1, n - int(max(SEARCH_FRACTION * n, MAX_OUTRO_SECONDS * info.fps)))
    latest = n - max(2, int(MIN_OUTRO_SECONDS * info.fps))
    if latest <= earliest:
        return None

    window = diffs[earliest:latest]
    cut_index = earliest + int(np.argmax(window))
    cut_strength = float(diffs[cut_index])
    start = cut_index + 1

    tail = frames[start:]
    if len(tail) < 2:
        return None
    tail_motion = float(np.abs(np.diff(tail, axis=0)).mean())
    tail_seconds = len(tail) / info.fps

    if cut_strength < max(CUT_RATIO * typical, 10.0):
        return None
    if tail_motion > min(STILL_RATIO * typical, STILL_ABSOLUTE):
        return None
    if not (MIN_OUTRO_SECONDS <= tail_seconds <= MAX_OUTRO_SECONDS):
        return None

    # Map back to source frames: the analysis pass may have sampled fewer.
    scale = info.n_frames / float(n)
    start_frame = int(round(start * scale))
    remaining = max(0, info.n_frames - start_frame)

    cut_score = min(1.0, cut_strength / (CUT_RATIO * typical))
    still_score = min(1.0, (STILL_RATIO * typical) / max(tail_motion, 1e-3))
    confidence = float(min(1.0, 0.5 * cut_score + 0.5 * min(1.0, still_score)))

    return Outro(
        start_frame=start_frame,
        start_time=start_frame / info.fps,
        frames=remaining,
        seconds=remaining / info.fps,
        confidence=confidence,
        reason=(f"hard cut {cut_strength:.0f} vs typical {typical:.1f}, "
                f"then {tail_motion:.2f} motion over {tail_seconds:.1f}s"),
    )
