"""The release build. TensorFlow isn't installed in CI, so these run on the training machine only."""

import json

import numpy as np
import pytest

keras = pytest.importorskip("keras")

from kanji_ml import release  # noqa: E402
from kanji_ml.model import build_model  # noqa: E402

KANJI = list("一二三四五六七八九十")


def small_trained_model(path):
    """A 10-kanji model whose classes score far apart, so float16 keeps Keras's order."""
    keras.utils.set_random_seed(0)
    model = build_model(classes=len(KANJI))
    dense = model.layers[-1]
    kernel, bias = dense.get_weights()
    dense.set_weights([kernel * 50, bias + np.arange(len(KANJI)) * 5])
    model.save(path)
    return path


def test_writes_the_float16_litert_model_its_labels_and_what_it_was_released_with(tmp_path):
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps(KANJI, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "release"

    release.build(small_trained_model(tmp_path / "kanji.keras"), labels, out, metrics={"top1": 0.99, "top8": 0.999})

    from ai_edge_litert.interpreter import Interpreter
    interpreter = Interpreter(model_path=str(out / "kanji.tflite"))
    assert list(interpreter.get_input_details()[0]["shape"]) == [1, 48, 48, 1]
    assert list(interpreter.get_output_details()[0]["shape"]) == [1, len(KANJI)]
    assert json.loads((out / "labels.json").read_text(encoding="utf-8")) == KANJI
    assert json.loads((out / "release.json").read_text(encoding="utf-8")) == {
        "runtime": "@litertjs/core@2.6.0",
        "accelerator": "wasm",
        "input": [1, 48, 48, 1],
        "classes": len(KANJI),
        "top1": 0.99,
        "top8": 0.999,
    }


def test_refuses_labels_that_dont_match_the_models_outputs(tmp_path):
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps(KANJI[:-1], ensure_ascii=False), encoding="utf-8")  # 9 labels for 10 outputs

    with pytest.raises(ValueError, match="9 labels.*10 outputs"):
        release.build(small_trained_model(tmp_path / "kanji.keras"), labels, tmp_path / "release", metrics={})
