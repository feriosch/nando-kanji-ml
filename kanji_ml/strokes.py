"""Turns a Drawing's strokes into the model's 48x48 input.

The app runs the same steps in TypeScript, so this is plain float64
arithmetic in a fixed order. The fixtures in fixtures/drawings.json check that
both versions agree. The spec is in nando-kanji-ml#1: the Drawing is fitted
into a box 4x the model's 40 pixels, drawn with a soft pen whose ink fades
from the centre line like ETL's scanned strokes, then given to to_model_input.
"""

import numpy as np

from kanji_ml.preprocess import _round, to_model_input

BOX = 160  # the Drawing is drawn to fit this square, pen included: 4x the model's 40-pixel box
PEN = 6  # ink fades from 1 on a stroke's centre line to 0 this many pixels away
MARGIN = 2  # blank pixels around the box


def _render(drawing: list[list[list[float]]]) -> np.ndarray:
    points = np.array([point for stroke in drawing for point in stroke], dtype=np.float64)
    if not points.size:
        raise ValueError("the Drawing has no points, so there is nothing to recognise")
    low = points.min(axis=0)
    extent = (points.max(axis=0) - low).max()
    scale = (BOX - 2 * PEN) / extent if extent > 0 else 0.0
    size = BOX + 2 * MARGIN
    y, x = np.mgrid[0:size, 0:size] + 0.5
    ink = np.zeros((size, size))
    for stroke in drawing:
        if not stroke:
            continue
        p =(np.array(stroke, dtype=np.float64) - low) * scale + (MARGIN + PEN)
        segments = list(zip(p[:-1], p[1:])) or [(p[0], p[0])]
        for (ax, ay), (bx, by) in segments:
            dx, dy = bx - ax, by - ay
            length2 = dx * dx + dy * dy
            t = np.clip(((x - ax) * dx + (y - ay) * dy) / length2, 0, 1) if length2 > 0 else 0.0
            ex, ey = x - (ax + t * dx), y - (ay + t * dy)
            ink = np.maximum(ink, 1 - np.sqrt(ex * ex + ey * ey) / PEN)
    return _round(ink * 255).astype(np.uint8)


def drawing_to_model_input(drawing: list[list[list[float]]]) -> np.ndarray:
    """A Drawing, as strokes of [x, y] points with y pointing down, as the model's 48x48 input."""
    return to_model_input(_render(drawing))
