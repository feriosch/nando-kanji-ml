import numpy as np
import pytest

from kanji_ml import etl9g
from kanji_ml.preprocess import to_model_input
from tests import fake_etl


def test_decodes_each_record_jis_code_to_its_character(tmp_path):
    path = tmp_path / "ETL9G_01"
    fake_etl.write(path, [
        fake_etl.record(0x3021, fake_etl.blank()),  # 亜
        fake_etl.record(0x2422, fake_etl.blank()),  # あ
        fake_etl.record(0x3F65, fake_etl.blank()),  # 水
    ])

    characters, _ = etl9g.read_file(path)

    assert characters == ["亜", "あ", "水"]


def test_unpacks_each_image_with_the_left_pixel_from_the_high_nibble(tmp_path):
    first = fake_etl.blank()
    first[0, 0] = 15   # top-left corner: high nibble of the first image byte
    first[0, 1] = 3    # its right neighbour: low nibble of the same byte
    first[126, 127] = 9  # bottom-right corner
    second = fake_etl.blank()
    second[60, 64] = 7
    path = tmp_path / "ETL9G_01"
    fake_etl.write(path, [fake_etl.record(0x3021, first), fake_etl.record(0x3022, second)])

    _, images = etl9g.read_file(path)

    assert images.shape == (2, 127, 128)
    assert images.dtype == "uint8"
    assert images[0, 0, 0] == 15 and images[0, 0, 1] == 3 and images[0, 126, 127] == 9
    assert images[0].sum() == 15 + 3 + 9
    assert images[1, 60, 64] == 7 and images[1].sum() == 7


def test_rejects_a_file_that_is_not_a_whole_number_of_records(tmp_path):
    path = tmp_path / "ETL9G_01"
    path.write_bytes(fake_etl.record(0x3021, fake_etl.blank()) + b"\x00" * 100)

    with pytest.raises(ValueError, match="ETL9G_01"):
        etl9g.read_file(path)


def test_names_the_file_and_record_when_a_jis_code_is_not_a_character(tmp_path):
    path = tmp_path / "ETL9G_07"
    fake_etl.write(path, [fake_etl.record(0x3021, fake_etl.blank()), fake_etl.record(0x0000, fake_etl.blank())])

    with pytest.raises(ValueError, match=r"ETL9G_07.*record 1"):
        etl9g.read_file(path)


def cross() -> np.ndarray:
    """A kanji-like mark in the middle of the frame: a horizontal and a vertical stroke, like 十."""
    image = fake_etl.blank()
    image[60:65, 35:95] = 11
    image[30:100, 62:67] = 11
    return image


def test_a_stroke_from_a_neighbouring_box_at_the_frame_edge_does_not_change_the_model_input():
    image = cross()
    image[20:110, 0:3] = 11  # cut off by the frame, and heavy: a third of all the ink

    np.testing.assert_array_equal(to_model_input(etl9g.without_form_marks(image)), to_model_input(cross()))


def test_a_small_dotted_guide_line_near_the_frame_edge_does_not_change_the_model_input():
    image = cross()
    image[10:115:6, 6:8] = 11  # 2-pixel dots every 6 rows, 6 pixels in from the edge

    np.testing.assert_array_equal(to_model_input(etl9g.without_form_marks(image)), to_model_input(cross()))


def test_faint_specks_stay_background_when_the_strongest_ink_was_a_form_mark():
    image = cross() // 11 * 9  # the kanji's ink is 9...
    image[0:3, 20:100] = 15    # ...and a form line along the top is 15, so these specks are under a fifth of it
    image[110, 120] = 2
    image[5, 30] = 2

    np.testing.assert_array_equal(to_model_input(etl9g.without_form_marks(image)), to_model_input(cross()))


def test_keeps_separate_parts_of_a_kanji_away_from_the_frame_edge():
    image = fake_etl.blank()
    image[40:44, 40:90] = 11  # two strokes 40 pixels apart, like 二...
    image[84:88, 30:100] = 11
    image[30:33, 95:98] = 11  # ...and a small separate dot

    np.testing.assert_array_equal(etl9g.without_form_marks(image), image)


def test_keeps_the_heaviest_ink_when_all_of_it_is_near_the_frame_edge():
    kanji = fake_etl.blank()
    kanji[0:127, 60:64] = 11  # one stroke from the top of the frame to the bottom
    image = kanji.copy()
    image[40:44, 110:128] = 11  # a lighter, separate one cut off by the right edge

    np.testing.assert_array_equal(etl9g.without_form_marks(image), kanji)


def test_keeps_a_kanji_written_up_to_the_frame_edge_when_it_also_has_a_separate_part_inside():
    image = fake_etl.blank()
    image[1:120, 60:64] = 11   # a stroke reaching the top of the frame...
    image[60:64, 20:110] = 11  # ...crossing another one...
    image[30:33, 30:33] = 11   # ...and a small separate dot away from the edge

    np.testing.assert_array_equal(etl9g.without_form_marks(image), image)


def test_keeps_a_lighter_part_of_a_kanji_that_runs_from_the_middle_to_the_frame_edge():
    image = fake_etl.blank()
    image[40:90, 20:50] = 11  # a heavy left part away from the edge, like 言 in 訂...
    image[50:54, 60:100] = 11  # ...and a separate 丁 whose stroke runs down to the bottom of the frame
    image[50:127, 78:82] = 11

    np.testing.assert_array_equal(etl9g.without_form_marks(image), image)


def test_refuses_an_image_that_is_not_an_etl9g_frame():
    with pytest.raises(ValueError, match="127x128"):
        etl9g.without_form_marks(np.zeros((28, 28), dtype=np.uint8))


def test_sets_pixels_under_a_fifth_of_the_strongest_to_zero_and_leaves_clean_images_unchanged():
    image = cross()
    image[50:55, 50:55] = 2  # faint, under a fifth of 11

    out = etl9g.without_form_marks(image)

    np.testing.assert_array_equal(out, cross())
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(etl9g.without_form_marks(fake_etl.blank()), fake_etl.blank())
