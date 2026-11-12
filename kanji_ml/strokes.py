"""Turns a Drawing's strokes into the model's 48x48 input.

The app runs the same steps in TypeScript, so this is plain float64
arithmetic in a fixed order. The fixtures in fixtures/drawings.json check that
both versions agree. The spec is in nando-kanji-ml#1: the Drawing is fitted
into a box 4x the model's 40 pixels, drawn with a soft pen whose ink fades
from the centre line like ETL's scanned strokes, then given to to_model_input.
"""

import numpy as np

from kanji_ml.preprocess import to_model_input

Drawing = list[list[list[float]]]  # strokes in the order drawn, each a list of [x, y] points, y pointing down

BOX = 160  # the Drawing is drawn to fit this square, pen included: 4x the model's 40-pixel box
PEN = 6  # ink fades from 1 on a stroke's centre line to 0 this many pixels away
MARGIN = 2  # blank pixels around the box


def _render(drawing: Drawing) -> np.ndarray:
    """The Drawing drawn with the soft pen on a (BOX + 2 * MARGIN)-pixel square, as uint8 0-255.

    A pixel's ink is measured at its centre, (column + 0.5, row + 0.5). Ink starts
    at 0 everywhere and each stroke can only raise it, so pixels further than PEN
    from every stroke stay 0. Strokes with no points are skipped.
    """
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
        fitted = (np.array(stroke, dtype=np.float64) - low) * scale + (MARGIN + PEN)
        segments = list(zip(fitted[:-1], fitted[1:])) or [(fitted[0], fitted[0])]  # a tap is a segment of length 0
        for (ax, ay), (bx, by) in segments:
            dx, dy = bx - ax, by - ay
            length2 = dx * dx + dy * dy
            t = np.clip(((x - ax) * dx + (y - ay) * dy) / length2, 0, 1) if length2 > 0 else 0.0
            off_x, off_y = x - (ax + t * dx), y - (ay + t * dy)  # from the nearest point on the segment
            ink = np.maximum(ink, 1 - np.sqrt(off_x * off_x + off_y * off_y) / PEN)
    return np.floor(ink * 255 + 0.5).astype(np.uint8)  # rounds halves up, like JavaScript's Math.round()


def drawing_to_model_input(drawing: Drawing) -> np.ndarray:
    """A Drawing as the model's 48x48 input: rendered to 0-255 by _render, then to_model_input."""
    return to_model_input(_render(drawing))
