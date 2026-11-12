"""Writes fixtures/drawings.json: Drawings and the 48x48 model input each should give.

nando-web tests its TypeScript copy of strokes.py and preprocess.py against
these, and each release copies them there with the model. Run it again only
when the preprocessing changes on purpose:

    python -m kanji_ml.fixtures
"""

import json
from pathlib import Path

import numpy as np

from kanji_ml.strokes import Drawing, drawing_to_model_input

OUT = Path(__file__).resolve().parent.parent / "fixtures" / "drawings.json"

# Strokes as a finger draws them on the app's ~370-point canvas, start to end.
KANJI = {
    "十": [[(60, 185), (310, 185)], [(185, 50), (185, 330)]],
    "口": [[(90, 90), (90, 290)], [(90, 90), (280, 90), (280, 290)], [(90, 285), (280, 285)]],
    "三": [[(90, 80), (280, 80)], [(110, 180), (260, 180)], [(60, 290), (310, 290)]],
    "日": [[(110, 60), (110, 320)], [(110, 60), (260, 60), (260, 320)], [(110, 190), (260, 190)], [(110, 315), (260, 315)]],
    "木": [[(50, 140), (320, 140)], [(185, 40), (185, 340)], [(180, 145), (60, 300)], [(190, 145), (320, 300)]],
    "田": [[(80, 70), (80, 300)], [(80, 70), (290, 70), (290, 300)], [(185, 70), (185, 300)], [(80, 185), (290, 185)], [(80, 295), (290, 295)]],
    "国": [[(60, 50), (60, 330)], [(60, 50), (310, 50), (310, 330)], [(110, 110), (260, 110)], [(120, 185), (250, 185)],
          [(185, 110), (185, 260)], [(100, 260), (270, 260)], [(220, 210), (240, 230)], [(60, 325), (310, 325)]],
    "書": [[(80, 60), (280, 60), (280, 160)], [(60, 110), (320, 110)], [(80, 160), (290, 160)], [(40, 200), (330, 200)],
          [(185, 30), (185, 230)], [(110, 230), (110, 340)], [(110, 230), (260, 230), (260, 340)], [(110, 285), (260, 285)],
          [(110, 335), (260, 335)]],
    "一": [[(40, 190), (330, 180)]],
    "丨": [[(185, 30), (182, 340)]],
}


def _resample(stroke, step, wobble=0.0, rng=None):
    """Points every `step` along a stroke's straight sections, like pointer events, optionally wobbling."""
    out = []
    for a, b in zip(stroke[:-1], stroke[1:]):
        a, b = np.array(a, dtype=float), np.array(b, dtype=float)
        n = max(1, int(np.linalg.norm(b - a) / step))
        out += [a + (b - a) * k / n for k in range(n)]
    out.append(np.array(stroke[-1], dtype=float))
    if wobble:
        out = [p + rng.normal(0, wobble, 2) for p in out]
    return [[round(float(x), 2), round(float(y), 2)] for x, y in out]


def drawings() -> dict[str, Drawing]:
    rng = np.random.default_rng(1)
    cases = {name: [_resample(s, 4, wobble=1.2, rng=rng) for s in strokes] for name, strokes in KANJI.items()}
    far = [[[x + 5000, y + 3000] for x, y in s] for s in cases["十"]]
    cases["十 far from the origin"] = far
    cases["十 far from the origin, at 10x scale"] = [[[x * 10, y * 10] for x, y in s] for s in far]
    cross = KANJI["十"]
    cases["十 straight"] = [_resample(s, 4) for s in cross]  # without wobble, so the two below match it
    cases["十 straight, with dense points"] = [_resample(s, 0.5) for s in cross]
    cases["十 straight, with 2 points per stroke"] = [[list(map(float, p)) for p in s] for s in cross]
    cases["木 with strokes that have no points"] = [[], *cases["木"], []]  # skipped, as if they weren't there
    cases["tap"] = [[[200.0, 150.0]]]
    cases["crossing strokes"] = [_resample(s, 4, 1.2, rng) for s in [[(60, 60), (310, 310)], [(310, 60), (60, 310)]]]
    cases["a stroke that doubles back"] = [_resample([(60, 120), (300, 120), (80, 135), (290, 260)], 4, 1.2, rng)]
    return cases


def write() -> None:
    lines = []
    for name, strokes in drawings().items():
        expected = drawing_to_model_input(strokes)
        rows = ",\n".join("    " + json.dumps(row.tolist()) for row in expected)
        lines.append(
            "{\n"
            f'  "name": {json.dumps(name, ensure_ascii=False)},\n'
            f'  "strokes": {json.dumps(strokes)},\n'
            f'  "expected": [\n{rows}\n  ]\n'
            "}"
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("[\n" + ",\n".join(lines) + "\n]\n", encoding="utf-8")


if __name__ == "__main__":
    write()
    print(f"wrote {OUT}")
