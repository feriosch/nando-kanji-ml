import pytest

from kanji_ml import etl9g
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
