"""Builds release/: the model nando-web runs, chosen in step 6 of nando-kanji-ml#1.

    python -m kanji_ml.release

Writes, and commits to git:
- kanji.tflite: the float16 LiteRT model, for LiteRT.js on WASM
- labels.json: output i is the kanji labels[i]
- release.json: the runtime and settings it was chosen with, and its test accuracy

The model must match Keras's top 8 on every fixture Drawing, and the labels must
match its outputs, or nothing is written. Tagging a commit vN then makes the
release workflow copy these, the fixtures and a version file to nando-web.
"""

import json
import shutil
from pathlib import Path

import tensorflow as tf

from kanji_ml.export import check_matches_keras, fixture_inputs, inference_model, litert_scores, to_litert, top

ROOT = Path(__file__).resolve().parent.parent
RUNTIME = "@litertjs/core@2.6.0"  # the README's "Runtime choice"; tests check bench/variants.js pins the same
ACCELERATOR = "wasm"


def build(trained: Path, labels: Path, out: Path, metrics: dict) -> None:
    model = inference_model(trained)
    names = json.loads(Path(labels).read_text(encoding="utf-8"))
    outputs = model.output_shape[-1]
    if len(names) != outputs:
        raise ValueError(f"{len(names)} labels for a model with {outputs} outputs")
    if not {"top1", "top8"} <= metrics.keys():
        raise ValueError("a release records its test accuracy: metrics needs top1 and top8 (from python -m kanji_ml.train)")

    out = Path(out)
    building = out.with_name(out.name + ".building")  # swapped in for out only once the model is checked
    shutil.rmtree(building, ignore_errors=True)
    building.mkdir(parents=True)
    work = building / ".saved_model"
    input_shape = (1, *model.input_shape[1:])  # one Drawing at a time
    model.export(work, format="tf_saved_model", verbose=False, input_signature=[tf.TensorSpec(input_shape, tf.float32)])
    to_litert(work, building / "kanji.tflite", float16=True)
    shutil.rmtree(work)

    _, x = fixture_inputs()
    check_matches_keras("kanji.tflite", litert_scores(building / "kanji.tflite", x), top(model.predict(x, verbose=0)))

    (building / "labels.json").write_text(json.dumps(names, ensure_ascii=False) + "\n", encoding="utf-8")
    (building / "release.json").write_text(json.dumps({
        "runtime": RUNTIME,
        "accelerator": ACCELERATOR,
        "input": list(input_shape),
        "classes": outputs,
        "top1": metrics["top1"],
        "top8": metrics["top8"],
    }, indent=2) + "\n", encoding="utf-8")
    shutil.rmtree(out, ignore_errors=True)
    building.rename(out)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build release/ for nando-web from the trained model.")
    parser.add_argument("--model", default="models/kanji.keras", help="the trained model from python -m kanji_ml.train")
    parser.add_argument("--data", default="data/prepared", help="the folder with labels.json the model was trained on")
    args = parser.parse_args()
    metrics_file = Path(args.model).with_name("metrics.json")
    metrics = json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file.exists() else {}
    build(Path(args.model), Path(args.data) / "labels.json", ROOT / "release", metrics)
    print(f"wrote {ROOT / 'release'}:", ", ".join(sorted(p.name for p in (ROOT / "release").iterdir())))
