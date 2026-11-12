"""Exports the trained model for the three browser runtimes step 6 of nando-kanji-ml#1 compares.

    python -m kanji_ml.export

Writes bench/models/ (not in git):
- tfjs/, tfjs_f16/: TensorFlow.js graph models
- kanji.tflite, kanji_f16.tflite: LiteRT
- kanji.onnx, kanji_f16.onnx: ONNX Runtime Web
and bench/reference.json: Keras's top-8 for every fixture Drawing, which the
bench page checks each runtime against, and the list of models exported.

The LiteRT and ONNX exports must match Keras's top 8 on every fixture, or the
export fails and leaves the previous bench/models/ as it was. TF.js only runs
in the browser, so the bench page checks it.

TF.js needs its own venv, because tensorflowjs 4.22 pins an older TensorFlow
(see the README's "Export and bench" section). The other converters run in the
training venv.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from kanji_ml.fixtures import OUT as FIXTURES
from kanji_ml.model import build_model
from kanji_ml.strokes import drawing_to_model_input

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "bench"
TFJS_CONVERTER = ROOT / ".venv-tfjs" / "bin" / "tensorflowjs_converter"
TOP = 8


def inference_model(trained: Path) -> keras.Model:
    """The trained model rebuilt in float32. Training's mixed precision leaves float16 ops that LiteRT can't run."""
    model = keras.models.load_model(trained)
    rebuilt = build_model(model.output_shape[-1])
    rebuilt.set_weights(model.get_weights())
    return rebuilt


def fixture_inputs() -> tuple[list[str], np.ndarray]:
    cases = json.loads(FIXTURES.read_text(encoding="utf-8"))
    names = [case["name"] for case in cases]
    x = np.stack([drawing_to_model_input(case["strokes"]) for case in cases])[..., None].astype(np.float32)
    return names, x


def top(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-scores, axis=1, kind="stable")[:, :TOP]


def check_matches_keras(name: str, scores: np.ndarray, keras_top: np.ndarray) -> None:
    """Raises unless every Drawing's top 8, in order, is the same as Keras's."""
    same = int((top(scores) == keras_top).all(axis=1).sum())
    if same < len(keras_top):
        raise ValueError(f"{name}'s top 8 matches Keras on only {same}/{len(keras_top)} fixture Drawings")


def _run(*command: str) -> None:
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError(f"{Path(command[0]).name} {' '.join(command[1:3])} failed ({done.returncode}):\n"
                           f"{done.stderr.strip()[-3000:]}")


def to_litert(saved_model: Path, out: Path, float16: bool) -> None:
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model))
    if float16:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    out.write_bytes(converter.convert())


def litert_scores(path: Path, x: np.ndarray) -> np.ndarray:
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    scores = []
    for image in x:  # one at a time, like the app
        interpreter.set_tensor(interpreter.get_input_details()[0]["index"], image[None])
        interpreter.invoke()
        scores.append(interpreter.get_tensor(interpreter.get_output_details()[0]["index"])[0])
    return np.stack(scores)


def _onnx(saved_model: Path, out: Path) -> None:
    _run(sys.executable, "-m", "tf2onnx.convert", "--saved-model", str(saved_model), "--output", str(out), "--opset", "17")


def _onnx_float16(model: Path, out: Path) -> None:
    import onnx
    from onnxconverter_common import float16

    # Inputs and outputs stay float32, so the app passes the same array to either model.
    onnx.save(float16.convert_float_to_float16(onnx.load(model), keep_io_types=True), out)


def _onnx_scores(path: Path, x: np.ndarray) -> np.ndarray:
    import onnxruntime

    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    return np.stack([session.run(None, {name: image[None]})[0][0] for image in x])  # one at a time, like the app


CHECKS = {
    "kanji.tflite": litert_scores,
    "kanji_f16.tflite": litert_scores,
    "kanji.onnx": _onnx_scores,
    "kanji_f16.onnx": _onnx_scores,
}


def export(trained: Path) -> dict[str, dict]:
    models = BENCH / "models"
    building = BENCH / ".models-building"  # swapped in for models/ only once every export is built and checked
    shutil.rmtree(building, ignore_errors=True)
    building.mkdir(parents=True)
    work = building / ".saved_model"
    model = inference_model(trained)
    # A fixed batch of 1: the app recognises one Drawing at a time, and WebGPU runtimes want fixed shapes.
    model.export(work, format="tf_saved_model", verbose=False,
                 input_signature=[tf.TensorSpec((1, *model.input_shape[1:]), tf.float32)])
    names, x = fixture_inputs()
    keras_top = top(model.predict(x, verbose=0))

    to_litert(work, building / "kanji.tflite", float16=False)
    to_litert(work, building / "kanji_f16.tflite", float16=True)
    _onnx(work, building / "kanji.onnx")
    _onnx_float16(building / "kanji.onnx", building / "kanji_f16.onnx")
    if TFJS_CONVERTER.exists():
        _run(str(TFJS_CONVERTER), "--input_format=tf_saved_model", "--output_format=tfjs_graph_model",
             str(work), str(building / "tfjs"))
        _run(str(TFJS_CONVERTER), "--input_format=tf_saved_model", "--output_format=tfjs_graph_model",
             "--quantize_float16=*", str(work), str(building / "tfjs_f16"))
    shutil.rmtree(work)

    report = {}
    for name, scores in CHECKS.items():
        check_matches_keras(name, scores(building / name, x), keras_top)
        report[name] = {"bytes": (building / name).stat().st_size, "top8_same_as_keras": f"{len(names)}/{len(names)}"}
    for name in ("tfjs", "tfjs_f16"):
        if (building / name).exists():
            size = sum(f.stat().st_size for f in (building / name).iterdir())
            report[name] = {"bytes": size, "top8_same_as_keras": "checked in the browser"}

    shutil.rmtree(models, ignore_errors=True)
    building.rename(models)
    labels = json.loads((ROOT / "data" / "prepared" / "labels.json").read_text(encoding="utf-8"))
    exported = [f"models/{name}" for name in CHECKS] + [f"models/{name}/model.json" for name in ("tfjs", "tfjs_f16")
                                                        if (models / name).exists()]
    (BENCH / "reference.json").write_text(json.dumps({
        "labels": labels,
        "exported": exported,
        "fixtures": [{"name": n, "input": image[..., 0].astype(int).tolist(), "keras_top8": [int(i) for i in t]}
                     for n, image, t in zip(names, x, keras_top)],
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export the model for TensorFlow.js, LiteRT and ONNX Runtime Web.")
    parser.add_argument("--model", default="models/kanji.keras", help="the trained model from python -m kanji_ml.train")
    args = parser.parse_args()
    if not TFJS_CONVERTER.exists():
        print(f"no {TFJS_CONVERTER}: skipping TensorFlow.js (see the README's \"Export and bench\" section)")
    for name, info in export(Path(args.model)).items():
        print(f"{name:18s} {info['bytes'] / 1e6:5.2f} MB   top-8 same as Keras: {info['top8_same_as_keras']}")
