"""The fixtures nando-web tests its TypeScript preprocessing against (spec in nando-kanji-ml#1)."""

import json
from functools import cache

import numpy as np
import pytest

from kanji_ml.fixtures import OUT
from kanji_ml.strokes import drawing_to_model_input


@cache
def cases() -> dict[str, dict]:
    return {case["name"]: case for case in json.loads(OUT.read_text(encoding="utf-8"))}


def expected(name: str) -> np.ndarray:
    return np.array(cases()[name]["expected"], dtype=np.int16)


def test_the_fixtures_cover_the_cases_in_the_spec():
    assert set(cases()) >= {
        "十", "口", "三", "木", "日", "田", "国", "書",
        "tap", "一", "丨",
        "十 far from the origin", "十 far from the origin, at 10x scale",
        "十 straight, with dense points", "十 straight, with 2 points per stroke",
        "crossing strokes", "a stroke that doubles back",
    }


@pytest.mark.parametrize("name", sorted(cases()))
def test_each_fixture_is_what_the_python_preprocessing_gives(name):
    out = drawing_to_model_input(cases()[name]["strokes"])

    # Exact here: the fixtures were written by this code. nando-web's TypeScript copy may
    # differ by 1, because to_model_input's matrix product adds up in a different order.
    np.testing.assert_array_equal(out, expected(name))


@pytest.mark.parametrize("variant, original", [
    ("十 far from the origin", "十"),
    ("十 far from the origin, at 10x scale", "十"),
    ("十 straight, with dense points", "十 straight"),
    ("十 straight, with 2 points per stroke", "十 straight"),
    ("木 with strokes that have no points", "木"),
])
def test_the_same_drawing_moved_scaled_or_sampled_differently_gives_the_same_fixture(variant, original):
    assert np.abs(expected(variant) - expected(original)).max() <= 1
