"""Still-image watermark removal.

The video tests can only ever check that a fill *looks* plausible, because real
footage has no un-marked original to compare against. Screen blending is
invertible, so these tests can do the stronger thing: composite a known glyph
over a known scene and demand the original back.
"""
from __future__ import annotations

import numpy as np
import pytest

from unmark import image as img
from unmark.glyph import (Glyph, calibrate, locate, read_image, unscreen,
                          write_image)
from unmark.pipeline import Options

from conftest import clean_frame

WIDTH, HEIGHT = 320, 240
GLYPH_W, GLYPH_H = 40, 40
MARGIN_X, MARGIN_Y = 70, 60          # of the ROI origin, from right and bottom


def make_glyph(w: int = GLYPH_W, h: int = GLYPH_H, peak: float = 0.35) -> Glyph:
    """A soft four-pointed star, like Flow's, as a screen coefficient map."""
    ys, xs = np.mgrid[0:h, 0:w]
    nx = (xs - (w - 1) / 2) / ((w - 1) / 2)
    ny = (ys - (h - 1) / 2) / ((h - 1) / 2)
    star = np.abs(nx) ** 0.6 + np.abs(ny) ** 0.6
    c = np.clip((1.0 - star) * 2.0, 0.0, 1.0).astype(np.float32) * peak
    return Glyph(c=c, margin_x=MARGIN_X, margin_y=MARGIN_Y, samples=0)


def screen(scene: np.ndarray, glyph: Glyph, x: int, y: int) -> np.ndarray:
    """Composite the glyph the way Flow does: obs = 1 - (1 - scene)(1 - c)."""
    out = scene.astype(np.float32) / 255.0
    h, w = glyph.c.shape
    roi = out[y:y + h, x:x + w]
    roi[...] = 1.0 - (1.0 - roi) * (1.0 - glyph.c[:, :, None])
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def place(glyph: Glyph, width: int = WIDTH, height: int = HEIGHT) -> tuple[int, int]:
    return width - glyph.margin_x, height - glyph.margin_y


@pytest.fixture
def glyph() -> Glyph:
    return make_glyph()


@pytest.fixture
def marked(glyph):
    scene = clean_frame(7, WIDTH, HEIGHT)
    x, y = place(glyph)
    return scene, screen(scene, glyph, x, y), (x, y)


def test_the_original_pixels_come_back(marked, glyph):
    """The whole point: recovery, not a plausible invention."""
    scene, marked_image, (x, y) = marked
    match = locate(marked_image, glyph)
    recovered = unscreen(marked_image, glyph, match)

    h, w = glyph.c.shape
    error = np.abs(recovered[y:y + h, x:x + w].astype(float)
                   - scene[y:y + h, x:x + w].astype(float))
    assert error.mean() < 2.0, f"mean error {error.mean():.2f} levels under the mark"


def test_the_mark_is_actually_visible_first(marked):
    """Guard the test itself: a glyph too faint to see would pass trivially."""
    scene, marked_image, (x, y) = marked
    difference = np.abs(marked_image.astype(float) - scene.astype(float))
    assert difference.max() > 20.0


def test_everything_outside_the_mark_is_untouched(marked, glyph):
    scene, marked_image, (x, y) = marked
    recovered = unscreen(marked_image, glyph, locate(marked_image, glyph))
    h, w = glyph.c.shape
    outside = np.ones(recovered.shape[:2], bool)
    outside[y:y + h, x:x + w] = False
    assert np.array_equal(recovered[outside], marked_image[outside])


def test_position_is_predicted_from_the_stored_margin(marked, glyph):
    _, marked_image, (x, y) = marked
    match = locate(marked_image, glyph)
    assert abs(match.x - x) <= 2 and abs(match.y - y) <= 2


def test_a_clean_image_scores_low(glyph):
    """Nothing there means nothing to remove - the corner must not be scrubbed."""
    clean = clean_frame(3, WIDTH, HEIGHT)
    assert locate(clean, glyph).confidence < img.MIN_CONFIDENCE


def test_calibration_recovers_the_coefficient(tmp_path):
    """Measured from scratch, the map should match the glyph that was composited."""
    truth = make_glyph(peak=0.45)
    paths = []
    for i in range(8):
        scene = clean_frame(i * 5, WIDTH, HEIGHT)
        x, y = place(truth)
        path = tmp_path / f"sample_{i}.png"
        write_image(path, screen(scene, truth, x, y))
        paths.append(path)

    measured = calibrate(paths)
    assert measured.samples == 8
    assert abs(measured.c.max() - truth.c.max()) < 0.12


def test_non_ascii_filenames_load(tmp_path, marked):
    """cv2.imread returns None for these on Windows; read_image must not."""
    _, marked_image, _ = marked
    path = tmp_path / "Investigative_collage_with_house—.png"
    write_image(path, marked_image)
    assert read_image(path).shape == marked_image.shape


def test_run_leaves_an_unmarked_image_alone(tmp_path, glyph):
    source = tmp_path / "plain.png"
    write_image(source, clean_frame(3, WIDTH, HEIGHT))
    out = tmp_path / "out.png"

    result = img.run(source, out, Options(target="off"), glyph=glyph)
    assert not result.removed
    assert "no Flow watermark found" in result.reason
    assert np.array_equal(read_image(out), read_image(source))


def test_run_removes_and_upscales(tmp_path, marked, glyph):
    scene, marked_image, _ = marked
    source = tmp_path / "marked.png"
    write_image(source, marked_image)
    out = tmp_path / "big.png"

    result = img.run(source, out, Options(target="1080p", upscale_mode="lanczos"),
                     glyph=glyph, polish=False)
    assert result.removed
    written = read_image(out)
    assert min(written.shape[:2]) == 1080      # targets key on the short side


def test_upscale_target_uses_the_short_side_on_a_tall_image(tmp_path, glyph):
    """A portrait photo should reach the target on its short side, not its long one."""
    tall = clean_frame(2, 240, 320)
    source = tmp_path / "tall.png"
    write_image(source, tall)
    out = tmp_path / "tall_big.png"

    img.run(source, out, Options(remove=False, target="1080p", upscale_mode="lanczos"),
            glyph=glyph)
    written = read_image(out)
    assert written.shape[1] == 1080 and written.shape[0] > 1080


def test_is_image_recognises_what_it_should():
    assert img.is_image("a.PNG") and img.is_image("b.jpeg")
    assert not img.is_image("c.mp4")


def test_a_faint_mark_on_a_bright_background_is_still_found(glyph):
    """The regression: screen adds c*(1 - scene), so on a bright picture the
    mark almost vanishes. Energy and template tests both miss it there; the
    model-fit correlation does not care about amplitude."""
    bright = np.full((HEIGHT, WIDTH, 3), 225, np.uint8)
    bright[::3, ::3] = 235                       # a little texture to fit against
    x, y = place(glyph)
    marked = screen(bright, glyph, x, y)

    faintness = np.abs(marked.astype(float) - bright.astype(float)).max()
    assert faintness < 30, "fixture is meant to be a faint mark"
    assert locate(marked, glyph).confidence >= img.MIN_CONFIDENCE


def test_confidence_comes_from_the_fit_not_the_template(glyph):
    """Scene texture alone correlates with the glyph shape well enough to score
    on the template test, so confidence must not be able to ride on that."""
    match = locate(clean_frame(3, WIDTH, HEIGHT), glyph)
    assert match.confidence == max(0.0, min(1.0, match.fit))
