"""Training augmentation. TensorFlow isn't installed in CI, so these run on the training machine only."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from kanji_ml.augment import thicker, thinner, vary_stroke_width  # noqa: E402


def vertical_line() -> np.ndarray:
    """A batch of one 48x48 image: a 4-pixel-wide stroke at full ink, columns 20-23."""
    image = np.zeros((1, 48, 48, 1), dtype=np.float32)
    image[0, 10:38, 20:24, 0] = 255
    return image


def test_thicker_strokes_gain_a_half_strength_pixel_on_each_side():
    out = thicker(vertical_line()).numpy()[0, 24, :, 0]  # a row across the middle of the stroke

    np.testing.assert_array_equal(out[18:26], [0, 127.5, 255, 255, 255, 255, 127.5, 0])


def test_thinner_strokes_lose_half_their_outermost_pixels():
    out = thinner(vertical_line()).numpy()[0, 24, :, 0]

    np.testing.assert_array_equal(out[18:26], [0, 0, 127.5, 255, 255, 127.5, 0, 0])


def test_each_image_comes_out_unchanged_thicker_or_thinner_and_all_three_happen():
    images = np.repeat(vertical_line(), 60, axis=0)

    out = vary_stroke_width(tf.constant(images)).numpy()

    assert out.shape == images.shape and out.dtype == np.float32
    options = {"same": images[0], "thicker": thicker(images[:1]).numpy()[0], "thinner": thinner(images[:1]).numpy()[0]}
    seen = set()
    for image in out:
        matches = [name for name, option in options.items() if np.array_equal(image, option)]
        assert len(matches) == 1
        seen.add(matches[0])
    assert seen == set(options)  # with 60 images, each of the three is all but certain to come up
