"""Reads ETL9G files: fixed-size binary records of handwritten characters.

Each sample was scanned from a box on a paper form, and the scan's frame also
caught pieces of the form around it: the borders and guide lines of the
neighbouring boxes, and the ends of strokes written in them. These form marks
widened the ink crop in about 3 of 4 samples, so the kanji was drawn small and
off-centre. without_form_marks() removes them.

It runs on ETL images only. A Drawing has no form around it, so the app's copy
of preprocess.py doesn't need it.
"""

from pathlib import Path

import numpy as np

from kanji_ml.preprocess import INK_THRESHOLD

RECORD_SIZE = 8199
WIDTH, HEIGHT = 128, 127

JOIN = 3  # ink separated by up to 2 * JOIN background pixels, across a row or column, is one cluster
NEAR_EDGE = 15  # a cluster with ink this close to the frame edge may be a form mark...
AT_EDGE = 3  # ...and is one if it has ink this close, cut off by the frame...
SMALL = 0.1  # ...or holds less than this share of the ink...
DEPTH = 25  # ...unless it reaches further into the frame than this, as only the kanji does


def _grow(mask: np.ndarray, distance: int) -> np.ndarray:
    """The pixels within distance of mask, across rows, columns and diagonals."""
    out = mask.copy()
    for step in range(1, distance + 1):
        out[step:, :] |= mask[:-step, :]
        out[:-step, :] |= mask[step:, :]
    columns = out.copy()
    for step in range(1, distance + 1):
        out[:, step:] |= columns[:, :-step]
        out[:, :-step] |= columns[:, step:]
    return out


def _runs(region: np.ndarray) -> np.ndarray:
    """Numbers each horizontal run of region pixels, counting along the rows."""
    starts = region & ~np.pad(region, ((0, 0), (1, 0)))[:, :-1]
    return np.cumsum(starts).reshape(region.shape)


def _connected(seeds: np.ndarray, region: np.ndarray) -> np.ndarray:
    """The pixels of region joined to seeds through region, by rows and columns."""
    row_runs, column_runs = _runs(region), _runs(region.T)
    reached = seeds & region
    while True:
        hit = np.zeros(row_runs.max() + 1, dtype=bool)
        hit[row_runs[reached]] = True
        along_rows = region & hit[row_runs]
        hit = np.zeros(column_runs.max() + 1, dtype=bool)
        hit[column_runs[along_rows.T]] = True
        along_columns = region & hit[column_runs].T
        if (along_columns == reached).all():
            return reached
        reached = along_columns


def _border(shape: tuple[int, int], width: int) -> np.ndarray:
    border = np.ones(shape, dtype=bool)
    border[width:-width, width:-width] = False
    return border


def without_form_marks(image: np.ndarray) -> np.ndarray:
    """The image with form marks and background set to 0.

    Background is every pixel at or below a fifth of the strongest one, as in
    preprocess.py. It's cleared here too, so that preprocess.py doesn't measure
    its fifth against a removed mark, which would bring faint specks back.

    The ink is split into clusters. A cluster with ink near the frame edge is a
    form mark if the frame cuts it off or it holds under a tenth of the ink,
    and it stays in the band along the edge where the form shows. Clusters
    away from the edge are never removed. If one cluster near the
    edge outweighs all the ink away from it, it's the kanji written up to the
    edge, and is kept.
    """
    if image.shape != (HEIGHT, WIDTH):
        raise ValueError(f"form marks are found in a {HEIGHT}x{WIDTH} ETL9G frame, not {image.shape[0]}x{image.shape[1]}")
    ink = image > image.max() * INK_THRESHOLD
    out = np.where(ink, image, 0).astype(image.dtype)
    if not ink.any():
        return out

    clustered = _grow(ink, JOIN)
    near_edge = _connected(ink & _border(ink.shape, NEAR_EDGE), clustered)
    clusters = []
    left = near_edge.copy()
    while left.any():
        seed = np.zeros_like(left)
        seed.flat[np.flatnonzero(left)[0]] = True
        cluster = _connected(seed, left)
        left &= ~cluster
        clusters.append(cluster & ink)
    if not clusters:
        return out

    weight = out.astype(np.float64)
    total = weight.sum()
    masses = [weight[cluster].sum() for cluster in clusters]
    heaviest = int(np.argmax(masses))
    keep = heaviest if masses[heaviest] > weight[ink & ~near_edge].sum() else None
    at_edge, band = _border(ink.shape, AT_EDGE), _border(ink.shape, DEPTH)
    for k, (cluster, mass) in enumerate(zip(clusters, masses)):
        in_band = not (cluster & ~band).any()
        if k != keep and in_band and ((cluster & at_edge).any() or mass < SMALL * total):
            out[cluster] = 0
    return out


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
