"""Detection: does it find the right box, and is the mask solid?"""
from __future__ import annotations

import numpy as np

from unmark.detect import FLOW_PRESET, detect, detect_in_frames, preset_region
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


def test_no_watermark_reports_nothing_rather_than_guessing(clean_clip):
    """Guessing a position is worse than useless on a clip from another tool:
    it cleans an untouched corner and leaves the real watermark in place."""
    det = detect(str(clean_clip))
    assert not det.found
    assert det.regions == []


def test_flow_preset_is_available_on_request(clean_clip):
    det = detect(str(clean_clip), fallback=True)
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

    Structural geometry is every bit as persistent as an overlay, so
    persistence alone cannot separate them. What does is that the response
    carries on past whatever box the threshold cut it down to - the girder
    belongs to a larger structure, and the ring around it is just as hot.

    Note the fixture has to be structure, not an isolated bright rectangle. A
    perfectly isolated, perfectly static rectangle *is* a watermark by every
    property the detector can measure, and no rule should claim otherwise.
    """
    frames = []
    for i in range(40):
        frame = blend_glyph(clean_frame(i))[:, :, 0].copy()
        # A run of static structure: bright rails with gaps, so thresholding
        # fragments it the way it fragmented the real girder.
        for row in range(100, 240, 12):
            frame[row:row + 5, 250:315] = 245
        frames.append(frame)

    regions = detect_in_frames(np.stack(frames)).regions
    assert len(regions) == 1, [r.to_dict() for r in regions]
    assert abs(regions[0].x - GLYPH_BOX[0]) <= 5


def test_faint_static_texture_in_a_dark_area_is_not_a_watermark():
    """Regression: a near-black corner produced two spurious regions.

    Static texture in shadow is as persistent and as isolated as a real
    overlay - the only thing separating them is how far it stands out from the
    picture, and the score used to saturate that measurement at a response of
    8, scoring a 13-level artifact the same as an 87-level watermark.

    The shadow has to be a soft vignette rather than a pasted rectangle: a hard
    unmoving edge is a static overlay by every definition here, and flagging it
    would be correct behaviour on a wrong fixture.
    """
    height, width = 480, 320
    xx = np.mgrid[0:height, 0:width][1].astype(np.float32)
    vignette = np.clip(xx / 70.0, 0.0, 1.0)      # crushed at the left, no hard edge

    frames = []
    for i in range(40):
        frame = blend_glyph(clean_frame(i))[:, :, 0].astype(np.float32) * vignette
        frame[44:58, 8:22] += 13.0               # faint unmoving speck in the shadow
        frames.append(np.clip(frame, 0, 255).astype(np.uint8))

    regions = detect_in_frames(np.stack(frames)).regions
    assert len(regions) == 1, [r.to_dict() for r in regions]
    assert abs(regions[0].x - GLYPH_BOX[0]) <= 5


def test_watermark_away_from_any_edge_is_still_found():
    """Plenty of tools stamp a mark across the middle of the frame.

    Placement is a ranking hint, never a veto: an edge-distance cutoff drops
    these silently, and the preset fallback then cleans a corner that was
    never marked.
    """
    frames = []
    for i in range(40):
        frame = clean_frame(i)
        frame = blend_glyph(frame, box=(140, 220, 40, 40))     # dead centre
        frames.append(frame[:, :, 0])

    regions = detect_in_frames(np.stack(frames)).regions
    assert regions, "a centred watermark must not be filtered out by placement"

    # Bounds can come back partial: against a high-contrast moving background
    # the threshold only keeps the glyph's strongest pixels. What this test
    # pins down is that placement no longer vetoes the find at all.
    got = regions[0]
    cx, cy = got.x + got.w // 2, got.y + got.h // 2
    assert 140 <= cx <= 180 and 220 <= cy <= 260, got.to_dict()


def test_flow_preset_is_within_the_frame():
    x0, y0, x1, y1 = FLOW_PRESET
    assert 0 < x0 < x1 < 1 and 0 < y0 < y1 < 1


def test_a_bright_static_prop_does_not_outrank_a_faint_watermark():
    """A clip that barely moves, where scene furniture is brighter than the mark.

    Collage-style clips settle after a second and then hold, which leaves plenty
    of high-contrast scene texture as persistent as the watermark. The mark
    still wins on presence - it is in every frame, the prop arrives a few frames
    in - but being semi-transparent it is far lower in contrast.

    With the strength term saturating at 30 the prop took first place and the
    dominance rule then discarded the real mark, so the watermark survived into
    the output untouched. Strength is a gate, not a ranking: past the point
    where a candidate clearly stands out, more contrast is not more
    watermark-like.
    """
    import cv2

    from conftest import FRAMES, HEIGHT, WIDTH

    rng = np.random.default_rng(0)
    prop = cv2.GaussianBlur(
        (rng.random((26, 22)) > 0.5).astype(np.float32) * 255.0, (0, 0), 0.6)

    frames = []
    for i in range(FRAMES):
        frame = clean_frame(i // 5, WIDTH, HEIGHT).copy()   # barely moving
        if i >= 8:                                          # prop lands, then holds
            frame[60:86, 40:62] = prop[:, :, None]
        # Fainter than the usual fixture, so the prop really does out-contrast it.
        frames.append(cv2.cvtColor(blend_glyph(frame, alpha=0.30), cv2.COLOR_BGR2GRAY))

    detection = detect_in_frames(np.stack(frames))
    assert detection.found

    x, y = GLYPH_BOX[0], GLYPH_BOX[1]
    best = detection.regions[0]
    assert abs(best.x - x) <= 4 and abs(best.y - y) <= 4, (
        f"ranked {best.box} first, but the watermark is at {GLYPH_BOX}")

    for region in detection.regions[1:]:
        assert not (30 <= region.x <= 70 and 50 <= region.y <= 90),             "the prop must not survive as a second region and get painted over"
