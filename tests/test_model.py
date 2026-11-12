"""The model's shape and size. TensorFlow isn't installed in CI, so these run on the training machine only."""

import numpy as np
import pytest

keras = pytest.importorskip("keras")

from kanji_ml.model import build_model  # noqa: E402


def test_gives_one_probability_per_kanji_for_each_48x48_image_of_raw_pixels():
    model = build_model(classes=10)
    images = np.zeros((3, 48, 48, 1), dtype=np.float32)
    images[:, 10:38, 22:26, 0] = 255  # the app passes the preprocessed image as it is, 0-255

    out = np.asarray(model(images, training=False))

    assert out.shape == (3, 10)
    np.testing.assert_allclose(out.sum(axis=1), 1, rtol=1e-5)
    assert (out >= 0).all()


def test_has_the_1_to_2_million_parameters_adr_0005_plans_for_etl9g():
    etl9g_kanji = 2965

    assert 1_000_000 <= build_model(classes=etl9g_kanji).count_params() <= 2_000_000
