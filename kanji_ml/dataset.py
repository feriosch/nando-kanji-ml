"""Builds the training files from the ETL9G folder.

Each ETL9G file holds 4 writers' samples, so splitting by file keeps every
writer on one side: files 01-45 train the model, and files 46-50 test it on
handwriting it has never seen.
"""

import json
from pathlib import Path

import numpy as np

from kanji_ml import etl9g
from kanji_ml.preprocess import to_model_input

FIRST_TEST_FILE = 46


def _is_hiragana(character: str) -> bool:
    return "ぁ" <= character <= "ゟ"


def _jis_order(character: str) -> bytes:
    return character.encode("euc_jp")  # EUC-JP keeps the JIS X 0208 order


def build(etl_dir: Path, out_dir: Path) -> dict[str, int]:
    etl_dir, out_dir = Path(etl_dir), Path(out_dir)
    samples = {"train": ([], []), "test": ([], [])}  # split -> (characters, images)
    for path in sorted(etl_dir.glob("ETL9G_[0-9][0-9]")):
        split = "test" if int(path.name[-2:]) >= FIRST_TEST_FILE else "train"
        characters, images = etl9g.read_file(path)
        for character, image in zip(characters, images):
            if not _is_hiragana(character):
                samples[split][0].append(character)
                samples[split][1].append(to_model_input(image))

    labels = sorted({c for characters, _ in samples.values() for c in characters}, key=_jis_order)
    index = {character: i for i, character in enumerate(labels)}

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "labels.json").write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")
    for split, (characters, images) in samples.items():
        np.savez(
            out_dir / f"{split}.npz",
            x=np.array(images, dtype=np.uint8).reshape(-1, 48, 48),
            y=np.array([index[c] for c in characters], dtype=np.uint16),
        )
    return {
        "kanji": len(labels),
        "train": len(samples["train"][0]),
        "test": len(samples["test"][0]),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build train.npz, test.npz and labels.json from ETL9G.")
    parser.add_argument("--etl", default="data/ETL9G", help="folder holding ETL9G_01 to ETL9G_50")
    parser.add_argument("--out", default="data/prepared", help="where to write the training files")
    args = parser.parse_args()

    summary = build(Path(args.etl), Path(args.out))
    print(f"kanji: {summary['kanji']}  train: {summary['train']}  test: {summary['test']}")
    print("expected with the full ETL9G: kanji: 2965  train: 533700  test: 59300")
