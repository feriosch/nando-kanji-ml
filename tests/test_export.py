"""The export's guards. TensorFlow isn't installed in CI, so these run on the training machine only."""

import sys

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from kanji_ml import export  # noqa: E402


def test_a_failing_converter_reports_its_own_error_message():
    failing = "import sys; sys.stderr.write('Unsupported op: FancyConv'); sys.exit(3)"

    with pytest.raises(RuntimeError, match="Unsupported op: FancyConv"):
        export._run(sys.executable, "-c", failing)


def test_an_export_whose_top_8_differs_from_keras_on_any_drawing_is_refused():
    keras_scores = np.arange(20, dtype=np.float32)[None].repeat(3, axis=0)  # every drawing ranks class 19 first
    exported = keras_scores.copy()
    exported[2, [12, 13]] = exported[2, [13, 12]]  # the third drawing swaps its 7th and 8th guesses

    export.check_matches_keras("kanji.tflite", keras_scores, export.top(keras_scores))  # the same scores pass
    with pytest.raises(ValueError, match=r"kanji\.tflite.*2/3"):
        export.check_matches_keras("kanji.tflite", exported, export.top(keras_scores))
