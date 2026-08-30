"""Detection of branded end cards.

Many tools append a short outro to their exports - a logo on a plain or frozen
background, spliced on after the real footage. It leaves three signatures at
once, and requiring all three is what keeps a legitimately quiet final shot from
being mistaken for one:

  a hard cut     the join is the sharpest frame-to-frame change in the clip
  a frozen tail  what follows barely moves, in absolute terms
  a plain tail   it carries much less detail than the footage before it
  a short tail   it lasts seconds, not minutes

Any one of those alone is common in ordinary footage. Together they are not.

Stillness is deliberately judged absolutely rather than against the clip's own
motion. A relative test looks reasonable and fails badly on calm footage: a
slideshow-paced clip has a median inter-frame motion near 0.2, which would demand
the end card be around ten times stiller than genuinely frozen to qualify. The
question is whether the tail is frozen, not whether it is frozen relative to how
lively the rest happened to be.
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
STILL_ABSOLUTE = 2.0        # the tail must be near-frozen, judged in absolute terms
DETAIL_RATIO = 0.65         # ...and visibly plainer than the footage before it


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

    # A branded card is plainer than real footage: flat ground, a logo, a line
    # of text. Comparing spatial detail against the body of the clip catches
    # that, and it is what separates a card from a held final shot, which
    # freezes but goes on looking like the film it belongs to.
    body_detail = float(frames[:start].std())
    tail_detail = float(tail.std())

    if cut_strength < max(CUT_RATIO * typical, 10.0):
        return None
    if tail_motion > STILL_ABSOLUTE:
        return None
    if body_detail > 1e-6 and tail_detail > DETAIL_RATIO * body_detail:
        return None
    if not (MIN_OUTRO_SECONDS <= tail_seconds <= MAX_OUTRO_SECONDS):
        return None

    # Map back to source frames: the analysis pass may have sampled fewer.
    scale = info.n_frames / float(n)
    start_frame = int(round(start * scale))
    remaining = max(0, info.n_frames - start_frame)

    cut_score = min(1.0, cut_strength / max(CUT_RATIO * typical, 10.0))
    still_score = min(1.0, STILL_ABSOLUTE / max(tail_motion, 1e-3))
    plain_score = min(1.0, (DETAIL_RATIO * body_detail) / max(tail_detail, 1e-3))
    confidence = float(min(1.0, (cut_score + still_score + plain_score) / 3.0))

    return Outro(
        start_frame=start_frame,
        start_time=start_frame / info.fps,
        frames=remaining,
        seconds=remaining / info.fps,
        confidence=confidence,
        reason=(f"hard cut {cut_strength:.0f} vs typical {typical:.1f}, then "
                f"{tail_motion:.2f} motion and {tail_detail:.0f} detail "
                f"(vs {body_detail:.0f}) over {tail_seconds:.1f}s"),
    )
