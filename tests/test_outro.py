"""End-card detection.

The failure that matters here is a false positive: trimming seconds off footage
that simply ended quietly. So the fixtures include the awkward cases - a clip
that ends on a held shot, and one that ends on a hard cut without freezing.
"""
from __future__ import annotations

import numpy as np
import pytest

from unmark import ffmpegio
from unmark.outro import detect_outro
from conftest import FPS, _write_clip, clean_frame


def _card(width: int = 320, height: int = 480) -> np.ndarray:
    """A plain branded end card: flat background, a mark, almost no detail."""
    frame = np.full((height, width, 3), 18, dtype=np.uint8)
    frame[height // 2 - 22:height // 2 + 22, width // 2 - 70:width // 2 + 70] = 210
    return frame


@pytest.fixture(scope="session")
def clip_with_card(tmp_path_factory):
    path = tmp_path_factory.mktemp("outro") / "with_card.mp4"
    card = _card()
    return _write_clip(path, ([clean_frame(i) for i in range(90)] +
                              [card.copy() for _ in range(48)]))


@pytest.fixture(scope="session")
def clip_without_card(tmp_path_factory):
    path = tmp_path_factory.mktemp("outro") / "no_card.mp4"
    return _write_clip(path, [clean_frame(i) for i in range(120)])


@pytest.fixture(scope="session")
def clip_ending_on_a_held_shot(tmp_path_factory):
    """Ends quiet, but eases in with no cut - a director's hold, not an end card."""
    path = tmp_path_factory.mktemp("outro") / "held.mp4"
    frames = [clean_frame(i) for i in range(90)]
    last = frames[-1]
    frames += [last.copy() for _ in range(40)]
    return _write_clip(path, frames)


def test_finds_the_end_card(clip_with_card):
    card = detect_outro(str(clip_with_card))
    assert card is not None, "a frozen branded card after a hard cut must be found"

    info = ffmpegio.probe(str(clip_with_card))
    assert abs(card.start_frame - 90) <= 3, card.to_dict()
    assert 1.5 <= card.seconds <= 2.5
    assert card.confidence > 0.5
    assert card.start_frame < info.n_frames


def test_ordinary_clip_has_no_end_card(clip_without_card):
    assert detect_outro(str(clip_without_card)) is None


def test_a_held_final_shot_is_not_an_end_card(clip_ending_on_a_held_shot):
    """Freezing alone is not enough - without a cut into it, nothing is trimmed."""
    assert detect_outro(str(clip_ending_on_a_held_shot)) is None


@pytest.fixture(scope="session")
def calm_clip_with_card(tmp_path_factory):
    """Slideshow-paced footage: barely any motion, then a card.

    Regression. Stillness used to be judged against the clip's own median
    motion, so a calm clip set an impossible bar for its own end card - the
    quieter the footage, the more frozen the card had to be to qualify.
    """
    path = tmp_path_factory.mktemp("outro") / "calm.mp4"
    body = [clean_frame(i // 6) for i in range(120)]
    return _write_clip(path, body + [_card().copy() for _ in range(48)])


def test_calm_footage_does_not_hide_its_end_card(calm_clip_with_card):
    card = detect_outro(str(calm_clip_with_card))
    assert card is not None, "a calm clip must not mask its own end card"
    assert abs(card.start_frame - 120) <= 4, card.to_dict()
    assert 1.5 <= card.seconds <= 2.5


def test_reason_is_reported(clip_with_card):
    card = detect_outro(str(clip_with_card))
    assert card.reason and "cut" in card.reason
    assert set(card.to_dict()) >= {"start_frame", "seconds", "confidence", "reason"}


@pytest.fixture(scope="session")
def clip_with_a_harder_content_cut(tmp_path_factory):
    """A scene change in the content that is sharper than the card's join.

    Regression. Taking the strongest cut in the window picked the content
    change, which made a "card" out of several seconds of real footage.
    """
    path = tmp_path_factory.mktemp("outro") / "harder_cut.mp4"
    dark = (clean_frame(0) * 0.25).astype(np.uint8)
    body = ([dark.copy() for _ in range(40)] +          # dark opening
            [clean_frame(i) for i in range(40, 100)])   # hard, bright scene change
    return _write_clip(path, body + [_card().copy() for _ in range(48)])


def test_content_cut_does_not_win_over_the_join(clip_with_a_harder_content_cut):
    card = detect_outro(str(clip_with_a_harder_content_cut))
    assert card is not None
    assert abs(card.start_frame - 100) <= 4, card.to_dict()
    assert card.seconds <= 2.5, "must not swallow footage before the card"


@pytest.fixture(scope="session")
def clip_with_a_fading_card(tmp_path_factory):
    """A card that fades its logo up over half a second before holding.

    Regression. Growing the card backwards while frames "look like" it stopped
    inside the fade, putting the supposed join where there was no cut.
    """
    path = tmp_path_factory.mktemp("outro") / "fading.mp4"
    card = _card()
    fade = [(card * (0.15 + 0.85 * k / 15.0)).astype(np.uint8) for k in range(15)]
    return _write_clip(path, [clean_frame(i) for i in range(90)]
                       + fade + [card.copy() for _ in range(33)])


def test_a_card_that_fades_in_is_found_whole(clip_with_a_fading_card):
    card = detect_outro(str(clip_with_a_fading_card))
    assert card is not None
    assert abs(card.start_frame - 90) <= 4, card.to_dict()


def test_short_clips_are_left_alone(tmp_path):
    path = _write_clip(tmp_path / "tiny.mp4", [clean_frame(i) for i in range(12)])
    assert detect_outro(str(path)) is None
