"""Builds the training files from the ETL9G folder.

Each ETL9G file holds 4 writers' samples, so splitting by file keeps every
writer on one side: files 01-45 train the model, and files 46-50 test it on
handwriting it has never seen.
"""

import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from kanji_ml import etl9g
from kanji_ml.preprocess import SIZE, to_model_input

FIRST_TEST_FILE = 46
WRITERS_PER_FILE = 4  # each wrote every character once
MAX_WORKERS = 8  # each holds one ETL9G file, about 300 MB, in memory


def hold_out_last_files(x: np.ndarray, y: np.ndarray, kanji: int, files: int):
    """Splits train.npz's samples into (x, y) for training and for the last `files` files.

    The samples are in file order, as build() writes them, so the held-out part
    is whole files: writers the rest has never seen.
    """
    per_file = WRITERS_PER_FILE * kanji
    in_data, leftover = divmod(len(x), per_file)
    if leftover:
        raise ValueError(f"{len(x)} samples aren't whole files of {WRITERS_PER_FILE} writers x {kanji} kanji")
    if files >= in_data:
        raise ValueError(f"can't hold out {files} files when there are only {in_data} files")
    split = len(x) - files * per_file
    if (np.bincount(y[split:], minlength=kanji) != files * WRITERS_PER_FILE).any():
        raise ValueError("the held-out samples don't hold each kanji once per writer, so they aren't in file order")
    return (x[:split], y[:split]), (x[split:], y[split:])


def _is_hiragana(character: str) -> bool:
    return "ぁ" <= character <= "ゟ"


def _jis_order(character: str) -> bytes:
    return character.encode("euc_jp")  # EUC-JP keeps the JIS X 0208 order


def _process_file(path: Path) -> tuple[list[str], np.ndarray]:
    """The kanji in one ETL9G file and their model inputs, without the hiragana."""
    file_characters, file_images = etl9g.read_file(path)
    kept = [i for i, c in enumerate(file_characters) if not _is_hiragana(c)]
    processed = np.empty((len(kept), SIZE, SIZE), dtype=np.uint8)
    for row, i in enumerate(kept):
        processed[row] = to_model_input(etl9g.without_form_marks(file_images[i]))
    return [file_characters[i] for i in kept], processed


def build(etl_dir: Path, out_dir: Path) -> dict[str, int]:
    etl_dir, out_dir = Path(etl_dir), Path(out_dir)
    paths = sorted(etl_dir.glob("ETL9G_[0-9][0-9]"))
    if not paths:
        raise FileNotFoundError(f"no ETL9G data files (ETL9G_01 to ETL9G_50) in {etl_dir}")

    characters = {"train": [], "test": []}
    images = {"train": [], "test": []}  # one uint8 array per file
    workers = min(len(paths), os.cpu_count() or 1, MAX_WORKERS)
    # forkserver, not fork: a process that has started threads (TensorFlow, say) can deadlock a forked child
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("forkserver")) as pool:
        for path, (file_characters, processed) in zip(paths, pool.map(_process_file, paths)):
            split = "test" if int(path.name[-2:]) >= FIRST_TEST_FILE else "train"
            characters[split].extend(file_characters)
            images[split].append(processed)

    labels = sorted({c for split in characters.values() for c in split}, key=_jis_order)
    index = {character: i for i, character in enumerate(labels)}

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "labels.json").write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")
    for split in ("train", "test"):
        np.savez(
            out_dir / f"{split}.npz",
            x=np.concatenate(images[split]) if images[split] else np.empty((0, SIZE, SIZE), np.uint8),
            # int32: TensorFlow's sparse losses don't accept unsigned labels
            y=np.array([index[c] for c in characters[split]], dtype=np.int32),
        )
    return {"kanji": len(labels), "train": len(characters["train"]), "test": len(characters["test"])}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build train.npz, test.npz and labels.json from ETL9G.")
    parser.add_argument("--etl", default="data/ETL9G", help="folder holding ETL9G_01 to ETL9G_50")
    parser.add_argument("--out", default="data/prepared", help="where to write the training files")
    args = parser.parse_args()

    summary = build(Path(args.etl), Path(args.out))
    print(f"kanji: {summary['kanji']}  train: {summary['train']}  test: {summary['test']}")
    print("expected with the full ETL9G: kanji: 2965  train: 533700  test: 59300")
