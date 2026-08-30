"""End to end: a marked clip goes in, a clean one comes out."""
from __future__ import annotations

import numpy as np
import pytest

from unmark import ffmpegio
from unmark.pipeline import Options, default_output, run
from conftest import GLYPH_BOX, clean_frame


def _frame(path, index: int) -> np.ndarray:
    return ffmpegio.read_frame_at(str(path), index)


def _glyph_error(frame: np.ndarray, index: int) -> float:
    """Mean absolute error against the known clean frame, over the glyph box."""
    x, y, w, h = GLYPH_BOX
    truth = clean_frame(index)[y:y + h, x:x + w].astype(np.float64)
    got = frame[y:y + h, x:x + w].astype(np.float64)
    return float(np.abs(truth - got).mean())


def test_removal_gets_closer_to_the_clean_truth(marked_clip, tmp_path):
    out = tmp_path / "clean.mp4"
    result = run(marked_clip, out, Options(engine="balanced", target="off"))

    assert out.exists() and out.stat().st_size > 0
    assert result.frames > 0
    assert result.regions and result.regions[0]["source"] == "detected"

    index = 20
    before = _glyph_error(_frame(marked_clip, index), index)
    after = _glyph_error(_frame(out, index), index)
    assert after < before * 0.5, f"error {before:.1f} -> {after:.1f}"


def test_output_keeps_source_dimensions_when_not_upscaling(marked_clip, tmp_path):
    out = tmp_path / "same.mp4"
    run(marked_clip, out, Options(engine="fast", target="off"))
    src, dst = ffmpegio.probe(str(marked_clip)), ffmpegio.probe(str(out))
    assert (dst.width, dst.height) == (src.width, src.height)
    assert dst.n_frames == src.n_frames


def test_upscale_reaches_the_requested_size(marked_clip, tmp_path):
    out = tmp_path / "big.mp4"
    result = run(marked_clip, out, Options(engine="fast", target="1080p"))
    info = ffmpegio.probe(str(out))
    assert (info.width, info.height) == (result.plan["out_w"], result.plan["out_h"])
    assert info.height == 1080 * 480 // 320 or info.width == 1080


def test_matte_note_explains_itself(marked_clip, tmp_path):
    result = run(marked_clip, tmp_path / "n.mp4", Options(engine="balanced"))
    assert result.matte and result.matte["note"], "the UI needs a reason to show"


def test_flat_background_still_removes_the_mark(flat_marked_clip, tmp_path):
    """The matte cannot solve here, so this exercises the inpaint fallback path."""
    out = tmp_path / "flat_clean.mp4"
    result = run(flat_marked_clip, out, Options(engine="balanced"))
    assert "inpaint only" in result.matte["note"]

    index = 20
    before = _glyph_error(_frame(flat_marked_clip, index), index)
    after = _glyph_error(_frame(out, index), index)
    assert after < before


def test_cancellation_stops_early(marked_clip, tmp_path):
    from unmark.pipeline import Cancelled

    calls = {"n": 0}

    def cancel_after_a_few() -> bool:
        calls["n"] += 1
        return calls["n"] > 5

    with pytest.raises(Cancelled):
        run(marked_clip, tmp_path / "cancelled.mp4", Options(engine="fast"),
            should_cancel=cancel_after_a_few)


def test_doing_nothing_is_rejected():
    with pytest.raises(ValueError):
        Options(remove=False, target="off", trim_outro=False).validate()


def test_trimming_alone_is_a_valid_job():
    Options(remove=False, target="off", trim_outro=True).validate()


def test_end_card_is_trimmed_from_the_output(marked_clip, tmp_path, tmp_path_factory):
    """A clip with a branded card should come out shorter, card gone."""
    from conftest import _write_clip
    from unmark.outro import detect_outro

    card = np.full((480, 320, 3), 18, dtype=np.uint8)
    card[218:262, 90:230] = 210
    src = _write_clip(tmp_path / "carded.mp4",
                      [clean_frame(i) for i in range(90)] +
                      [card.copy() for _ in range(48)])
    assert detect_outro(str(src)) is not None, "fixture must actually have a card"

    out = tmp_path / "trimmed.mp4"
    result = run(src, out, Options(remove=False, target="off", trim_outro=True))

    assert result.outro is not None
    assert result.frames < ffmpegio.probe(str(src)).n_frames
    assert abs(result.frames - 90) <= 3, result.frames
    # The final frame must be real footage, not the flat card.
    assert _frame(out, result.frames - 1).std() > 20


def test_end_card_is_kept_when_trimming_is_off(marked_clip, tmp_path):
    from conftest import _write_clip

    card = np.full((480, 320, 3), 18, dtype=np.uint8)
    card[218:262, 90:230] = 210
    src = _write_clip(tmp_path / "carded2.mp4",
                      [clean_frame(i) for i in range(90)] +
                      [card.copy() for _ in range(48)])

    result = run(src, tmp_path / "kept.mp4",
                 Options(remove=False, target="1080p", trim_outro=False))
    assert result.outro is None
    assert result.frames == ffmpegio.probe(str(src)).n_frames


def test_no_watermark_and_nothing_else_to_do_is_an_error(clean_clip, tmp_path):
    """Guessing a position would clean untouched footage and miss the real mark."""
    with pytest.raises(ValueError, match="no watermark"):
        run(clean_clip, tmp_path / "x.mp4",
            Options(remove=True, target="off", trim_outro=False))


def test_no_watermark_still_upscales(clean_clip, tmp_path):
    result = run(clean_clip, tmp_path / "up.mp4",
                 Options(remove=True, target="1080p", trim_outro=False))
    assert result.regions == []
    assert result.engine == "none"
    assert result.plan["out_w"] == 1080


def test_default_output_name_describes_the_job(tmp_path):
    name = default_output("01.mp4", Options(remove=True, target="4k"), tmp_path).name
    assert name == "01_clean_4k.mp4"
    name = default_output("01.mp4", Options(remove=True, target="off"), tmp_path).name
    assert name == "01_clean.mp4"


def test_progress_reports_reach_completion(marked_clip, tmp_path):
    seen = []
    run(marked_clip, tmp_path / "p.mp4", Options(engine="fast"),
        on_progress=lambda stage, frac, msg: seen.append((stage, frac)))
    stages = {s for s, _ in seen}
    assert {"detect", "encode", "done"} <= stages
    assert seen[-1][1] == 1.0
