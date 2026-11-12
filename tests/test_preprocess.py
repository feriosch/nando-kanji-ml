import numpy as np

from kanji_ml.preprocess import to_model_input


def etl_frame() -> np.ndarray:
    """A blank image the size of an ETL9G sample."""
    return np.zeros((127, 128), dtype=np.uint8)


def test_returns_a_48x48_uint8_image_with_the_strongest_ink_at_255():
    image = etl_frame()
    image[50:70, 50:70] = 11  # ETL ink rarely reaches 15

    out = to_model_input(image)

    assert out.shape == (48, 48)
    assert out.dtype == np.uint8
    assert out.max() == 255


def test_crops_a_mark_in_a_corner_and_scales_it_to_fill_the_centre_40x40():
    image = etl_frame()
    image[0:20, 0:20] = 11

    out = to_model_input(image)

    expected = np.zeros((48, 48), dtype=np.uint8)
    expected[4:44, 4:44] = 255
    np.testing.assert_array_equal(out, expected)


def test_keeps_the_aspect_ratio_of_a_tall_thin_mark():
    image = etl_frame()
    image[30:70, 80:90] = 11  # 40 tall, 10 wide, like the stroke of 丨

    out = to_model_input(image)

    expected = np.zeros((48, 48), dtype=np.uint8)
    expected[4:44, 19:29] = 255
    np.testing.assert_array_equal(out, expected)


def test_ignores_faint_specks_far_from_the_ink():
    clean = etl_frame()
    clean[0:20, 0:20] = 11
    noisy = clean.copy()
    noisy[120, 120] = 2  # scanner noise, well under a fifth of the ink's strength
    noisy[60, 100] = 1
    noisy[5, 30] = 2

    np.testing.assert_array_equal(to_model_input(noisy), to_model_input(clean))


def test_a_blank_image_gives_all_zeros():
    out = to_model_input(etl_frame())

    np.testing.assert_array_equal(out, np.zeros((48, 48), dtype=np.uint8))


def test_shrinking_averages_the_area_each_output_pixel_covers():
    image = etl_frame()
    image[10:41, 10:70] = 15  # 31 rows of strong ink...
    image[41:70, 10:70] = 6   # ...over 29 rows of weaker ink: a 60x60 mark, shrunk to 40x40

    out = to_model_input(image)

    expected = np.zeros((48, 48), dtype=np.uint8)
    expected[4:24, 4:44] = 255   # output rows 0-19 cover only strong source rows
    expected[24, 4:44] = 204     # output row 20 covers source rows 30-31.5: (15 + 6 * 0.5) / 1.5 = 12 -> 12/15 of 255
    expected[25:44, 4:44] = 102  # the rest cover only weak rows: 6/15 of 255
    np.testing.assert_array_equal(out, expected)


def test_gives_the_same_result_whatever_the_input_resolution():
    small = np.zeros((64, 64), dtype=np.uint8)
    small[10:50, 10:16] = 9   # an L: a vertical stroke...
    small[44:50, 10:40] = 13  # ...and a stronger horizontal one
    large = np.kron(small, np.ones((4, 4), dtype=np.uint8))  # the same drawing at 4x the resolution

    np.testing.assert_array_equal(to_model_input(large), to_model_input(small))
