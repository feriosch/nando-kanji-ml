"""The fixtures nando-web tests its TypeScript preprocessing against (spec in nando-kanji-ml#1)."""

import json
from pathlib import Path

import numpy as np
import pytest

from kanji_ml.strokes import drawing_to_model_input

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "drawings.json"


def cases() -> dict[str, dict]:
    return {case["name"]: case for case in json.loads(FIXTURES.read_text(encoding="utf-8"))}


def test_the_fixtures_cover_the_cases_in_the_spec():
    assert set(cases()) >= {
        "十", "口", "三", "木", "日", "田", "国", "書",
        "tap", "一", "丨",
        "十 far from the origin", "十 at 10x scale",
        "十 with dense points", "十 with 2 points per stroke",
        "crossing strokes", "a stroke that doubles back",
    }


@pytest.mark.parametrize("name", sorted(cases()))
def test_each_fixture_matches_the_python_preprocessing_to_within_1(name):
    case = cases()[name]
    expected = np.array(case["expected"], dtype=np.int16)

    out = drawing_to_model_input(case["strokes"]).astype(np.int16)

    assert expected.shape == (48, 48)
    assert np.abs(out - expected).max() <= 1


@pytest.mark.parametrize("variant", ["十 far from the origin", "十 at 10x scale", "十 with dense points", "十 with 2 points per stroke"])
def test_the_same_drawing_moved_scaled_or_sampled_differently_gives_the_same_fixture(variant):
    original = np.array(cases()["十"]["expected"], dtype=np.int16)

    assert np.abs(np.array(cases()[variant]["expected"], dtype=np.int16) - original).max() <= 1
