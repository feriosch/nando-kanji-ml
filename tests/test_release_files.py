"""The committed release/ folder, checked in CI: what a vN tag would ship to nando-web."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RELEASE = ROOT / "release"


def released() -> dict:
    return json.loads((RELEASE / "release.json").read_text(encoding="utf-8"))


def test_ships_the_litert_js_version_the_bench_measured():
    pinned = re.search(r"@litertjs/core@([\w.-]+)/", (ROOT / "bench" / "variants.js").read_text(encoding="utf-8"))

    assert pinned, "bench/variants.js doesn't pin @litertjs/core"
    assert released()["runtime"] == f"@litertjs/core@{pinned.group(1)}"


def test_has_one_label_per_model_output_and_records_the_test_accuracy():
    labels = json.loads((RELEASE / "labels.json").read_text(encoding="utf-8"))

    assert len(labels) == released()["classes"] == len(set(labels))
    assert 0 < released()["top1"] <= released()["top8"] <= 1
    assert (RELEASE / "kanji.tflite").stat().st_size > 0
