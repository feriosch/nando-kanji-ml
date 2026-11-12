"""Reads ETL9G files: fixed-size binary records of handwritten characters."""

from pathlib import Path

import numpy as np

RECORD_SIZE = 8199
WIDTH, HEIGHT = 128, 127


def _jis_to_character(jis: int) -> str:
    escaped = bytes([0x1B, 0x24, 0x42, jis >> 8, jis & 0xFF, 0x1B, 0x28, 0x42])
    return escaped.decode("iso2022_jp")


def read_file(path: Path) -> tuple[list[str], np.ndarray]:
    data = Path(path).read_bytes()
    count, leftover = divmod(len(data), RECORD_SIZE)
    if leftover:
        raise ValueError(f"{path} is {len(data)} bytes, not a whole number of {RECORD_SIZE}-byte ETL9G records")
    records = np.frombuffer(data, dtype=np.uint8).reshape(count, RECORD_SIZE)
    jis = records[:, 2].astype(np.uint16) << 8 | records[:, 3]
    characters = [_jis_to_character(int(code)) for code in jis]
    packed = records[:, 64:64 + WIDTH * HEIGHT // 2]
    images = np.stack([packed >> 4, packed & 0x0F], axis=-1).reshape(count, HEIGHT, WIDTH)
    return characters, images
