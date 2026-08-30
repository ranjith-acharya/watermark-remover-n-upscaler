"""Detection of branded end cards.

Many tools append a short outro to their exports - a logo on a plain or frozen
background, spliced on after the real footage.

Finding it is a matter of picking the right anchor. Three things are reliably
true of a card and rarely true together of real footage:

  it settles     the clip ends on a near-frozen stretch
  it is joined   a hard cut separates it from the footage before it
  it holds       that frozen stretch is a large share of the card itself

The anchor is the frozen ending, because that is the one part guaranteed to be
present and easy to measure. The join is then the strongest cut between that
and the outro-length limit, and the tests run against it.

Two things this deliberately does *not* do, both learned the hard way:

It does not judge the card against the clip's own averages. A relative test
reads well and fails on exactly the footage people bring: on dark or calm video
the clip-wide median sinks until the card cannot clear a bar derived from it.
Five samples ending in the *identical* card were split three-to-two by that
alone. Comparisons are against the seconds immediately before the join, which is
the shot the card actually replaced.

It does not walk backwards while frames "look like" the card. Cards routinely
fade or animate their logo in over a second or more, so their detail and
brightness ramp rather than hold, and any such walk stops partway in - which
then puts the supposed join inside the card, where there is no cut to find.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ffmpegio import VideoInfo, probe, read_gray_frames

ANALYSIS_WIDTH = 160
MAX_OUTRO_SECONDS = 10.0
MIN_OUTRO_SECONDS = 0.3

FROZEN_MOTION = 1.0          # per-frame change counting as "not moving"
MIN_FROZEN_SECONDS = 0.25    # the clip must actually settle before it ends
CUT_RATIO = 3.0              # the join must stand out this far above typical motion
MIN_CUT = 10.0               # ...and be a real cut in absolute terms
TAIL_MOTION_MAX = 3.0        # a card may animate, but it is not action footage
MIN_FROZEN_FRACTION = 0.35   # most of a card is a held frame, not a moving one
DETAIL_RATIO = 1.10          # a card is never busier than the shot it replaced
BODY_WINDOW_SECONDS = 2.0    # how much footage before the join to compare against


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
    n = len(frames)
    if n < 30:
        return None

    # diffs[i] is the change between frame i and i+1, so a cut *into* frame i+1
    # shows up at index i.
    diffs = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))
    detail = frames.std(axis=(1, 2))
    typical = float(np.median(diffs)) or 1e-3

    # 1. Anchor on the frozen ending.
    frozen = n - 1
    while frozen > 0 and diffs[frozen - 1] <= FROZEN_MOTION:
        frozen -= 1
    if (n - frozen) < max(2, int(MIN_FROZEN_SECONDS * info.fps)):
        return None

    # 2. Consider every cut between the outro limit and that ending, latest
    #    first, and take the first one whose tail behaves like a card.
    #
    #    Not the strongest cut: an ordinary scene change in the content is
    #    routinely sharper than the join. On one sample the content cut scored
    #    39.3 against the join's 38.8 and won by that margin, producing a 5.4s
    #    "card" of real footage. The card is the *last* segment, not the most
    #    dramatic one, so the search runs backwards from the end.
    threshold = max(CUT_RATIO * typical, MIN_CUT)
    lowest = max(1, n - int(MAX_OUTRO_SECONDS * info.fps))
    highest = max(lowest + 1, frozen)

    peaks = [i for i in range(lowest, min(highest, len(diffs)))
             if diffs[i] >= threshold
             and diffs[i] >= diffs[i - 1]
             and (i + 1 >= len(diffs) or diffs[i] >= diffs[i + 1])]

    for cut_index in reversed(peaks):
        cut_strength = float(diffs[cut_index])
        start = cut_index + 1

        tail = frames[start:]
        if len(tail) < 2 or start < 2:
            continue
        tail_seconds = len(tail) / info.fps
        if not (MIN_OUTRO_SECONDS <= tail_seconds <= MAX_OUTRO_SECONDS):
            continue
        tail_motion = float(np.abs(np.diff(tail, axis=0)).mean())
        if tail_motion > TAIL_MOTION_MAX:
            continue
        tail_detail = float(tail.std())

        # A card is mostly a held frame. Real footage that happens to follow a
        # cut and settle at the very end is not: the frozen part is a sliver of
        # it. This is what stops the search latching onto a scene change several
        # seconds early and calling four seconds of footage an end card.
        frozen_fraction = (n - frozen) / float(len(tail))
        if frozen_fraction < MIN_FROZEN_FRACTION:
            continue

        # Detail is only a sanity check. It cannot carry the decision: on darker
        # clips the card and the footage it replaces measure almost the same
        # (26.4 against 23.0 on one sample), and a stricter test there rejects
        # the right answer.
        body_from = max(0, start - int(BODY_WINDOW_SECONDS * info.fps))
        body = frames[body_from:start]
        body_detail = float(body.std()) if len(body) else 0.0
        if body_detail > 1e-6 and tail_detail > DETAIL_RATIO * body_detail:
            continue
        break
    else:
        return None

    # Map back to source frames: the analysis pass may have sampled fewer.
    scale = info.n_frames / float(n)
    start_frame = int(round(start * scale))
    remaining = max(0, info.n_frames - start_frame)
    if remaining < 1:
        return None

    cut_score = min(1.0, cut_strength / max(CUT_RATIO * typical, MIN_CUT))
    still_score = min(1.0, TAIL_MOTION_MAX / max(tail_motion, 1e-3))
    hold_score = min(1.0, frozen_fraction / MIN_FROZEN_FRACTION)
    confidence = float(min(1.0, (cut_score + still_score + hold_score) / 3.0))

    return Outro(
        start_frame=start_frame,
        start_time=start_frame / info.fps,
        frames=remaining,
        seconds=remaining / info.fps,
        confidence=confidence,
        reason=(f"cut {cut_strength:.0f} into a {tail_seconds:.1f}s tail that is "
                f"{frozen_fraction:.0%} held frames ({tail_detail:.0f} detail "
                f"against {body_detail:.0f} before it)"),
    )
