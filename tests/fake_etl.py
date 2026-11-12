"""Writes fake ETL9G files for tests, straight from the published record layout.

An ETL9G record is 8,199 bytes. The JIS X 0208 code is a big-endian uint16 at
byte 2, and the 128x127 image is stored from byte 64, two 4-bit pixels per byte,
the left pixel in the high nibble.
"""

import numpy as np

RECORD_SIZE = 8199
WIDTH, HEIGHT = 128, 127


def record(jis: int, image: np.ndarray) -> bytes:
    assert image.shape == (HEIGHT, WIDTH) and image.max() <= 15
    out = bytearray(RECORD_SIZE)
    out[2:4] = jis.to_bytes(2, "big")
    pixels = image.astype(np.uint8).reshape(-1)
    out[64:64 + HEIGHT * WIDTH // 2] = (pixels[0::2] << 4 | pixels[1::2]).tobytes()
    return bytes(out)


def blank() -> np.ndarray:
    return np.zeros((HEIGHT, WIDTH), dtype=np.uint8)


def write(path, records: list[bytes]) -> None:
    path.write_bytes(b"".join(records))
