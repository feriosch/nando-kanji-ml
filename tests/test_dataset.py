import pytest
import json

import numpy as np

from kanji_ml import dataset
from kanji_ml.preprocess import to_model_input
from tests import fake_etl

A = 0x3021   # 亜
AA = 0x3022  # 唖
HIRAGANA_A = 0x2422  # あ


def mark(top: int, left: int, height: int, width: int) -> np.ndarray:
    image = fake_etl.blank()
    image[top:top + height, left:left + width] = 11
    return image


def fake_etl9g(folder, files: dict[str, list[tuple[int, np.ndarray]]]):
    """A folder laid out like the real ETL9G one, including its non-data ETL9INFO file."""
    folder.mkdir()
    (folder / "ETL9INFO").write_bytes(b"not records")
    for name, records in files.items():
        fake_etl.write(folder / name, [fake_etl.record(jis, image) for jis, image in records])
    return folder


def test_labels_list_each_kanji_once_in_jis_order_without_hiragana(tmp_path):
    etl = fake_etl9g(tmp_path / "ETL9G", {
        "ETL9G_01": [(AA, mark(10, 10, 30, 30)), (HIRAGANA_A, mark(10, 10, 30, 30)), (A, mark(10, 10, 30, 30))],
        "ETL9G_46": [(A, mark(10, 10, 30, 30)), (AA, mark(10, 10, 30, 30))],
    })

    dataset.build(etl, tmp_path / "out")

    labels = json.loads((tmp_path / "out" / "labels.json").read_text(encoding="utf-8"))
    assert labels == ["亜", "唖"]


def test_splits_by_writer_file_and_keeps_each_image_with_its_label(tmp_path):
    a_in_45, aa_in_45 = mark(10, 10, 40, 8), mark(10, 10, 20, 20)
    a_in_46, aa_in_46 = mark(10, 10, 8, 40), mark(30, 30, 12, 30)
    etl = fake_etl9g(tmp_path / "ETL9G", {
        "ETL9G_45": [(A, a_in_45), (HIRAGANA_A, mark(10, 10, 30, 30)), (AA, aa_in_45)],
        "ETL9G_46": [(AA, aa_in_46), (A, a_in_46)],
    })

    dataset.build(etl, tmp_path / "out")

    labels = json.loads((tmp_path / "out" / "labels.json").read_text(encoding="utf-8"))
    train = np.load(tmp_path / "out" / "train.npz")
    test = np.load(tmp_path / "out" / "test.npz")
    assert [labels[i] for i in train["y"]] == ["亜", "唖"]
    assert [labels[i] for i in test["y"]] == ["唖", "亜"]
    assert train["x"].dtype == np.uint8 and train["x"].shape == (2, 48, 48)
    assert train["y"].dtype == np.int32  # what TensorFlow's sparse losses accept
    np.testing.assert_array_equal(train["x"][0], to_model_input(a_in_45))
    np.testing.assert_array_equal(train["x"][1], to_model_input(aa_in_45))
    np.testing.assert_array_equal(test["x"][0], to_model_input(aa_in_46))
    np.testing.assert_array_equal(test["x"][1], to_model_input(a_in_46))


def test_removes_marks_from_the_etl_form_before_preprocessing(tmp_path):
    kanji = mark(40, 40, 30, 30)
    with_form_line = kanji.copy()
    with_form_line[:, 0:2] = 11  # a box border on the form, cut off by the frame
    etl = fake_etl9g(tmp_path / "ETL9G", {"ETL9G_01": [(A, with_form_line)], "ETL9G_46": [(A, kanji)]})

    dataset.build(etl, tmp_path / "out")

    train = np.load(tmp_path / "out" / "train.npz")
    np.testing.assert_array_equal(train["x"][0], to_model_input(kanji))


def test_returns_how_many_kanji_and_samples_it_kept(tmp_path):
    etl = fake_etl9g(tmp_path / "ETL9G", {
        "ETL9G_01": [(A, mark(10, 10, 30, 30)), (HIRAGANA_A, mark(10, 10, 30, 30)), (AA, mark(10, 10, 30, 30))],
        "ETL9G_02": [(A, mark(10, 10, 30, 30))],
        "ETL9G_50": [(AA, mark(10, 10, 30, 30))],
    })

    summary = dataset.build(etl, tmp_path / "out")

    assert summary == {"kanji": 2, "train": 3, "test": 1}


def test_refuses_a_folder_with_no_etl9g_files(tmp_path):
    folder = tmp_path / "ETL9G"
    folder.mkdir()
    (folder / "ETL9INFO").write_bytes(b"not records")

    with pytest.raises(FileNotFoundError, match="ETL9G_01"):
        dataset.build(folder, tmp_path / "out")

    assert not (tmp_path / "out").exists()


def file_ordered(files: int, kanji: int) -> tuple[np.ndarray, np.ndarray]:
    """Samples laid out like train.npz: file after file, each holding 4 writers x every kanji."""
    y = np.tile(np.arange(kanji, dtype=np.int32), files * 4)
    x = np.arange(len(y), dtype=np.uint8).reshape(-1, 1, 1)  # each image marks its position
    return x, y


def test_holds_out_the_last_files_so_validation_is_on_unseen_writers():
    x, y = file_ordered(files=3, kanji=5)  # 20 samples per file

    (x_train, y_train), (x_val, y_val) = dataset.hold_out_last_files(x, y, kanji=5, files=1)

    np.testing.assert_array_equal(x_train.ravel(), np.arange(40))
    np.testing.assert_array_equal(x_val.ravel(), np.arange(40, 60))
    np.testing.assert_array_equal(y_val, np.tile(np.arange(5), 4))


def test_refuses_samples_that_are_not_whole_files():
    x, y = file_ordered(files=3, kanji=5)

    with pytest.raises(ValueError, match="whole files"):
        dataset.hold_out_last_files(x[:-1], y[:-1], kanji=5, files=1)


def test_refuses_to_hold_out_every_file():
    x, y = file_ordered(files=2, kanji=5)

    with pytest.raises(ValueError, match="2 files"):
        dataset.hold_out_last_files(x, y, kanji=5, files=2)


def test_refuses_samples_that_are_not_in_file_order():
    x, y = file_ordered(files=3, kanji=5)
    shuffled = np.random.default_rng(0).permutation(len(y))

    with pytest.raises(ValueError, match="file order"):
        dataset.hold_out_last_files(x[shuffled], y[shuffled], kanji=5, files=1)
