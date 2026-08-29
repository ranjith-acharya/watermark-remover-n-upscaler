"""Removal engines and the alpha matte solve."""
from __future__ import annotations

import numpy as np
import pytest

from flowclean.detect import Region
from flowclean.remove import (ENGINES, Remover, estimate_matte, fill_fast,
                              fill_inpaint, prepare_region, unblend)
from conftest import (GLYPH_ALPHA, GLYPH_BOX, GLYPH_COLOR, blend_glyph, clean_frame,
                      diamond_mask)


def _region() -> Region:
    x, y, w, h = GLYPH_BOX
    return Region(x, y, w, h, confidence=1.0, mask=diamond_mask(w, h))


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse < 1e-9 else 10 * np.log10(255.0 ** 2 / mse)


def test_prepare_region_pads_and_feathers():
    prep = prepare_region(_region(), 320, 480)
    assert prep.w > GLYPH_BOX[2] and prep.h > GLYPH_BOX[3]
    assert prep.mask.shape == (prep.h, prep.w)
    assert prep.feather.max() <= 1.0 and prep.feather.min() >= 0.0
    # The glyph centre must be fully replaced, the ROI corner fully preserved.
    assert prep.feather[prep.h // 2, prep.w // 2, 0] > 0.95
    assert prep.feather[0, 0, 0] < 0.05


def test_prepare_region_clamps_at_the_frame_edge():
    region = Region(300, 460, 40, 40, confidence=1.0, mask=np.ones((40, 40), bool))
    prep = prepare_region(region, 320, 480)
    assert prep.x >= 0 and prep.y >= 0
    assert prep.x + prep.w <= 320 and prep.y + prep.h <= 480


@pytest.mark.parametrize("engine", ["fast", "balanced"])
def test_engines_remove_the_visible_mark(engine):
    clean = clean_frame(0)
    marked = blend_glyph(clean)
    prep = prepare_region(_region(), 320, 480)

    out = Remover([prep], engine).apply(marked)
    x, y, w, h = GLYPH_BOX
    before = _psnr(marked[y:y + h, x:x + w], clean[y:y + h, x:x + w])
    after = _psnr(out[y:y + h, x:x + w], clean[y:y + h, x:x + w])
    assert after > before + 3, f"{engine}: {before:.1f} dB -> {after:.1f} dB"


def test_remover_leaves_the_rest_of_the_frame_untouched():
    marked = blend_glyph(clean_frame(3))
    prep = prepare_region(_region(), 320, 480)
    out = Remover([prep], "balanced").apply(marked)

    untouched = np.ones(marked.shape[:2], dtype=bool)
    untouched[prep.y:prep.y + prep.h, prep.x:prep.x + prep.w] = False
    assert np.array_equal(out[untouched], marked[untouched])


def test_remover_does_not_mutate_its_input():
    marked = blend_glyph(clean_frame(5))
    original = marked.copy()
    Remover([prepare_region(_region(), 320, 480)], "balanced").apply(marked)
    assert np.array_equal(marked, original)


def test_unknown_engine_is_rejected():
    with pytest.raises(ValueError):
        Remover([], "magic")
    assert set(ENGINES) == {"fast", "balanced", "ai"}


def test_ai_engine_degrades_when_the_model_is_missing():
    remover = Remover([prepare_region(_region(), 320, 480)], "ai", lama=None)
    assert remover.engine == "balanced"


def test_fill_helpers_only_touch_masked_pixels():
    roi = clean_frame(0)[:64, :64]
    mask = np.zeros((64, 64), np.uint8)
    mask[20:40, 20:40] = 255
    for filled in (fill_fast(roi, mask), fill_inpaint(roi, mask)):
        assert np.allclose(filled[mask == 0], roi[mask == 0], atol=1e-4)


# --------------------------------------------------------------------------- #
# matte
# --------------------------------------------------------------------------- #

def _roi_stack(busy: bool = True, n: int = 48):
    prep = prepare_region(_region(), 320, 480)
    rois = np.stack([
        blend_glyph(clean_frame(i, busy=busy))[prep.y:prep.y + prep.h,
                                               prep.x:prep.x + prep.w]
        for i in range(n)])
    return prep, rois


def test_matte_recovers_alpha_and_colour_on_a_varied_background():
    prep, rois = _roi_stack(busy=True)
    matte = estimate_matte(rois, prep.mask)

    assert matte.usable, matte.reject_reason()
    core = matte.reliable & (matte.alpha > 0.3)
    assert core.sum() > 40, "the solve should cover the body of the glyph"
    assert abs(float(matte.alpha[core].mean()) - GLYPH_ALPHA) < 0.12
    assert np.allclose(matte.color[core].mean(axis=0), GLYPH_COLOR, atol=45)


def test_matte_unblend_beats_inpainting_where_it_applies():
    prep, rois = _roi_stack(busy=True)
    matte = estimate_matte(rois, prep.mask)
    prep.matte = matte

    clean = clean_frame(0)[prep.y:prep.y + prep.h, prep.x:prep.x + prep.w]
    marked = rois[0]
    where = matte.reliable

    inpainted = fill_inpaint(marked, prep.mask)
    recovered = unblend(marked, matte)
    assert _psnr(recovered[where], clean[where]) > _psnr(inpainted[where], clean[where])


def test_matte_is_rejected_when_the_solve_is_corrupted():
    """The seam check must be independent enough to catch a wrong alpha/colour."""
    import copy

    from flowclean.remove import _seam_score

    prep, rois = _roi_stack(busy=True)
    good = estimate_matte(rois, prep.mask)
    bad = copy.deepcopy(good)
    bad.alpha = np.clip(bad.alpha * 1.8, 0.0, 0.94)
    bad.color = np.full_like(bad.color, 255.0)
    bad.seam = _seam_score(rois.astype(np.float32), prep.mask, bad)

    assert good.usable and not bad.usable
    assert bad.seam < good.seam


def test_matte_is_rejected_on_a_flat_background():
    """The solve has nothing to regress against; it must decline, not guess."""
    prep, rois = _roi_stack(busy=False)
    matte = estimate_matte(rois, prep.mask)
    assert not matte.usable
    assert matte.reject_reason()


def test_matte_stability_separates_signal_from_noise():
    """Pure noise under the mask must not read as a solvable watermark."""
    prep, rois = _roi_stack(busy=True)
    rng = np.random.default_rng(0)
    noisy = np.clip(rng.normal(30, 3, size=rois.shape), 0, 255).astype(np.uint8)

    assert estimate_matte(rois, prep.mask).stability > 0.7
    assert estimate_matte(noisy, prep.mask).stability < 0.4


def test_matte_summary_is_json_friendly():
    prep, rois = _roi_stack()
    summary = estimate_matte(rois, prep.mask).summary()
    assert set(summary) == {"seam", "stability", "coverage", "peak_alpha", "usable"}
    assert all(isinstance(v, (int, float, bool)) for v in summary.values())
