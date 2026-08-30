"""Resolution planning. Targets are keyed on the short side, so vertical video
scales the way people expect rather than being squashed to a landscape height."""
from __future__ import annotations

import pytest

from unmark.upscale import DEFAULT_MODEL, MODELS, TARGETS, plan_upscale, resize_to
from conftest import clean_frame


def test_vertical_720p_to_4k():
    plan = plan_upscale(720, 1280, "4k")
    assert (plan.out_w, plan.out_h) == (2160, 3840)


def test_landscape_720p_to_4k():
    plan = plan_upscale(1280, 720, "4k")
    assert (plan.out_w, plan.out_h) == (3840, 2160)


@pytest.mark.parametrize("target,expected", [
    ("1080p", (1080, 1920)), ("1440p", (1440, 2560)), ("720p", (720, 1280)),
])
def test_targets_hit_the_short_side(target, expected):
    plan = plan_upscale(720, 1280, target)
    assert (plan.out_w, plan.out_h) == expected


def test_off_keeps_the_source_size():
    plan = plan_upscale(720, 1280, "off")
    assert (plan.out_w, plan.out_h) == (720, 1280)
    assert not plan.changes_size
    assert plan.net_scale == 1


def test_odd_sources_still_give_even_output():
    plan = plan_upscale(721, 1281, "1080p")
    assert plan.out_w % 2 == 0 and plan.out_h % 2 == 0


def test_ai_mode_sets_the_network_scale():
    plan = plan_upscale(720, 1280, "4k", mode="ai")
    assert plan.mode == "ai"
    assert plan.net_scale == MODELS[DEFAULT_MODEL]["scale"]


def test_ai_mode_downgrades_when_not_actually_upscaling():
    """Running a super-resolution network to then shrink the result is waste."""
    plan = plan_upscale(1080, 1920, "720p", mode="ai")
    assert plan.mode == "lanczos"
    assert plan.net_scale == 1


def test_auto_upscaler_prefers_a_gpu(monkeypatch):
    """A dedicated GPU is the default; the CPU path is the fallback."""
    import unmark.upscale as up

    monkeypatch.setattr(up, "torch_status",
                        lambda: {"available": True, "cuda": True, "device": "gpu",
                                 "reason": ""})
    assert up.resolve_mode("auto") == "ai"
    assert up.plan_upscale(720, 1280, "4k", "auto").mode == "ai"


def test_auto_upscaler_falls_back_without_a_gpu(monkeypatch):
    import unmark.upscale as up

    monkeypatch.setattr(up, "torch_status",
                        lambda: {"available": False, "cuda": False, "device": None,
                                 "reason": "no torch"})
    assert up.resolve_mode("auto") == "lanczos"
    assert up.plan_upscale(720, 1280, "4k", "auto").mode == "lanczos"


def test_explicit_modes_are_left_alone(monkeypatch):
    import unmark.upscale as up

    monkeypatch.setattr(up, "torch_status",
                        lambda: {"available": True, "cuda": True, "device": "gpu",
                                 "reason": ""})
    assert up.resolve_mode("lanczos") == "lanczos"


def test_unknown_target_is_rejected():
    with pytest.raises(ValueError):
        plan_upscale(720, 1280, "8k")
    assert set(TARGETS) == {"off", "720p", "1080p", "1440p", "4k"}


def test_resize_is_a_noop_at_the_same_size():
    frame = clean_frame(0)
    assert resize_to(frame, frame.shape[1], frame.shape[0]) is frame


def test_resize_changes_dimensions():
    out = resize_to(clean_frame(0), 160, 240)
    assert out.shape == (240, 160, 3)
