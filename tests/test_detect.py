"""Detection: does it find the right box, and is the mask solid?"""
from __future__ import annotations

import numpy as np

from flowclean.detect import FLOW_PRESET, detect, detect_in_frames, preset_region
from conftest import GLYPH_BOX, blend_glyph, clean_frame, diamond_mask


def test_finds_the_watermark_box(marked_clip):
    det = detect(str(marked_clip))
    assert det.found, "watermark should be detected, not fall back to the preset"

    region = det.regions[0]
    x, y, w, h = GLYPH_BOX
    assert abs(region.x - x) <= 3 and abs(region.y - y) <= 3
    assert abs(region.w - w) <= 5 and abs(region.h - h) <= 5
    assert region.confidence > 0.3


def test_mask_is_solid_not_a_ring(marked_clip):
    """The high-pass response peaks on edges; the mask must still cover the middle."""
    region = detect(str(marked_clip)).regions[0]
    expected_fill = diamond_mask(region.w, region.h).mean()
    assert region.mask.mean() > 0.75 * expected_fill

    # The centre is the pixel a ring-shaped mask would miss.
    cy, cx = region.h // 2, region.w // 2
    assert region.mask[cy, cx]


def test_no_watermark_falls_back_to_preset(clean_clip):
    det = detect(str(clean_clip))
    assert not det.found
    assert det.regions[0].source == "preset"


def test_preset_matches_measured_flow_geometry():
    region = preset_region(720, 1280)
    assert (region.x, region.y, region.w, region.h) == (576, 1136, 48, 48)
    assert region.source == "preset"


def test_static_scene_edges_are_not_reported():
    """A hard edge that never moves is persistent too - size and placement rules
    are what keep it from being called a watermark."""
    frames = []
    for _ in range(24):
        frame = np.full((480, 320), 40, dtype=np.uint8)
        frame[:, 150:] = 200          # a full-height edge through the middle
        frames.append(frame)
    det = detect_in_frames(np.stack(frames))
    assert det.regions == []


def test_detection_survives_a_moving_scene_with_hard_edges():
    frames = [blend_glyph(clean_frame(i)) for i in range(40)]
    gray = np.stack([f[:, :, 0] for f in frames])
    det = detect_in_frames(gray)
    assert det.regions, "the glyph should still be found among moving content"
    assert abs(det.regions[0].x - GLYPH_BOX[0]) <= 4


def test_static_scene_feature_does_not_become_a_second_region():
    """Regression: a short clip of a slow pan reported a girder as a watermark.

    A bright static bar near the frame edge passes every size and placement
    rule. What rules it out is that the response continues past its box, and
    that it scores well below the real mark.
    """
    frames = []
    for i in range(40):
        frame = blend_glyph(clean_frame(i))[:, :, 0].copy()
        frame[120:190, 290:300] = 245        # a hard, unmoving vertical feature
        frames.append(frame)

    regions = detect_in_frames(np.stack(frames)).regions
    assert len(regions) == 1, [r.to_dict() for r in regions]
    assert abs(regions[0].x - GLYPH_BOX[0]) <= 5


def test_flow_preset_is_within_the_frame():
    x0, y0, x1, y1 = FLOW_PRESET
    assert 0 < x0 < x1 < 1 and 0 < y0 < y1 < 1
