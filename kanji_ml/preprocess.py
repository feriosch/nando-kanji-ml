"""Turns a grayscale image of one kanji into the model's 48x48 input.

The app runs the same steps in TypeScript on the Owner's Drawing, so this is
plain NumPy arithmetic that can be copied exactly. It rounds halves up
(floor(x + 0.5)), like JavaScript's Math.round(), rather than to even like
Python's round(), and computes in float64, like JavaScript numbers.
"""

import numpy as np

SIZE = 48
INK_BOX = 40  # the ink's longer side is scaled to this, leaving a 4-pixel margin
INK_THRESHOLD = 0.2  # pixels at or below this fraction of the strongest pixel are background


def _round(x):
    """Rounds halves up, like JavaScript's Math.round()."""
    return np.floor(np.asarray(x) + 0.5)


def _area_weights(source: int, target: int) -> np.ndarray:
    """A (target, source) matrix that averages source pixels into target pixels by overlap."""
    edges = np.arange(target + 1) * source / target
    start, end = edges[:-1, None], edges[1:, None]
    pixel = np.arange(source)[None, :]
    overlap = np.clip(np.minimum(end, pixel + 1) - np.maximum(start, pixel), 0, None)
    return overlap / (source / target)


def _resize(image: np.ndarray, height: int, width: int) -> np.ndarray:
    return _area_weights(image.shape[0], height) @ image @ _area_weights(image.shape[1], width).T


def to_model_input(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float64)
    image[image <= image.max() * INK_THRESHOLD] = 0
    if not image.any():
        return np.zeros((SIZE, SIZE), dtype=np.uint8)
    rows = np.flatnonzero(image.any(axis=1))
    cols = np.flatnonzero(image.any(axis=0))
    ink = image[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

    scale = INK_BOX / max(ink.shape)
    height = max(1, int(_round(ink.shape[0] * scale)))
    width = max(1, int(_round(ink.shape[1] * scale)))
    top, left = (SIZE - height) // 2, (SIZE - width) // 2

    out = np.zeros((SIZE, SIZE))
    out[top:top + height, left:left + width] = _resize(ink, height, width)
    return _round(out * 255 / out.max()).astype(np.uint8)
