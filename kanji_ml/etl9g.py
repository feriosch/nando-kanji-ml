"""Reads ETL9G files: fixed-size binary records of handwritten characters."""

from pathlib import Path

import numpy as np

RECORD_SIZE = 8199
WIDTH, HEIGHT = 128, 127


def _jis_to_character(jis: int) -> str | None:
    """The JIS X 0208 character for a code, or None if the code isn't one.

    Both bytes must be 0x21-0x7E: Python's codec turns anything else into
    control characters rather than failing.
    """
    high, low = jis >> 8, jis & 0xFF
    if not (0x21 <= high <= 0x7E and 0x21 <= low <= 0x7E):
        return None
    try:
        character = bytes([0x1B, 0x24, 0x42, high, low, 0x1B, 0x28, 0x42]).decode("iso2022_jp")
    except UnicodeDecodeError:
        return None
    return character if len(character) == 1 else None


def read_file(path: Path) -> tuple[list[str], np.ndarray]:
    data = Path(path).read_bytes()
    count, leftover = divmod(len(data), RECORD_SIZE)
    if leftover:
        raise ValueError(f"{path} is {len(data)} bytes, not a whole number of {RECORD_SIZE}-byte ETL9G records")
    records = np.frombuffer(data, dtype=np.uint8).reshape(count, RECORD_SIZE)
    jis = records[:, 2].astype(np.uint16) << 8 | records[:, 3]
    characters = []
    for i, code in enumerate(jis):
        character = _jis_to_character(int(code))
        if character is None:
            raise ValueError(f"{path} record {i} has JIS code {int(code):#06x}, which is not a JIS X 0208 character")
        characters.append(character)
    packed = records[:, 64:64 + WIDTH * HEIGHT // 2]
    images = np.stack([packed >> 4, packed & 0x0F], axis=-1).reshape(count, HEIGHT, WIDTH)
    return characters, images
