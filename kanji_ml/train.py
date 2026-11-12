"""Trains the model on data/prepared/ and reports the shipping gate's top-1 and top-8.

    python -m kanji_ml.train

The last VALIDATION_FILES ETL9G training files are held out for early stopping,
so validation, like the test set, is on writers the model hasn't seen. The
test set is used once, at the end. Never runs in CI: it needs the GPU and data.
"""

import argparse
import json
import time
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from kanji_ml import augment
from kanji_ml.dataset import hold_out_last_files
from kanji_ml.evaluate import GATE_TOP_8, top_k_accuracy
from kanji_ml.model import build_model

VALIDATION_FILES = 2  # 8 writers, about 4% of the training samples
BATCH = 256
EPOCHS = 40  # at most: early stopping usually ends training sooner
LEARNING_RATE = 1e-3  # Adam's usual start
PLATEAU_FACTOR, PLATEAU_EPOCHS = 0.3, 2  # cut the learning rate to 30% when validation loss stalls for 2 epochs...
MIN_LEARNING_RATE = 1e-5  # ...but no lower than this
STOP_EPOCHS = 5  # stop when validation loss hasn't improved for this many epochs, keeping the best weights
SMALLEST_LIMIT = 1000  # --limit below this leaves too few validation samples to pick the best epoch


def _batches(x: np.ndarray, y: np.ndarray, augmented: bool = False) -> tf.data.Dataset:
    """Batches of (float32 images with a channel axis, labels), shuffled and augmented for training.

    Shuffling covers every sample: train.npz is in file order, so a smaller buffer
    would fill each batch from only a few files' writers.
    """
    data = tf.data.Dataset.from_tensor_slices((x[..., None], y))
    if augmented:
        data = data.shuffle(len(x), reshuffle_each_iteration=True)
    data = data.batch(BATCH).map(lambda images, labels: (tf.cast(images, tf.float32), labels),
                                 num_parallel_calls=tf.data.AUTOTUNE)
    if augmented:
        geometric = augment.geometric()
        data = data.map(lambda images, labels: (augment.vary_stroke_width(geometric(images, training=True)), labels),
                        num_parallel_calls=tf.data.AUTOTUNE)
    return data.prefetch(tf.data.AUTOTUNE)


def train(data_dir: Path, out_dir: Path, limit: int | None = None) -> dict:
    keras.mixed_precision.set_global_policy("mixed_float16")
    labels = json.loads((data_dir / "labels.json").read_text(encoding="utf-8"))
    train_npz, test_npz = np.load(data_dir / "train.npz"), np.load(data_dir / "test.npz")
    (x_train, y_train), (x_val, y_val) = hold_out_last_files(train_npz["x"], train_npz["y"], len(labels), VALIDATION_FILES)
    x_test, y_test = test_npz["x"], test_npz["y"]
    if limit is not None:  # a quick check that everything runs
        x_train, y_train, x_val, y_val = x_train[:limit], y_train[:limit], x_val[:limit // 10], y_val[:limit // 10]
        x_test, y_test = x_test[:limit // 10], y_test[:limit // 10]

    model = build_model(len(labels))
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", keras.metrics.SparseTopKCategoricalAccuracy(k=8, name="top8")],
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "kanji.keras"
    start = time.time()
    model.fit(
        _batches(x_train, y_train, augmented=True),
        validation_data=_batches(x_val, y_val),
        epochs=EPOCHS,
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=PLATEAU_FACTOR, patience=PLATEAU_EPOCHS,
                                              min_lr=MIN_LEARNING_RATE),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=STOP_EPOCHS, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_loss", save_best_only=True),
            keras.callbacks.CSVLogger(out_dir / "history.csv"),
        ],
        verbose=2,
    )

    scores = model.predict(_batches(x_test, y_test), verbose=0)
    result = {
        "top1": top_k_accuracy(scores, y_test, 1),
        "top8": top_k_accuracy(scores, y_test, 8),
        "test_samples": int(len(y_test)),
        "parameters": int(model.count_params()),
        "minutes": round((time.time() - start) / 60, 1),
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the kanji model and report top-1 and top-8 on the test set.")
    parser.add_argument("--data", default="data/prepared", help="folder with train.npz, test.npz and labels.json")
    parser.add_argument("--out", default="models", help="where to write kanji.keras, history.csv and metrics.json")
    parser.add_argument("--limit", type=int, help=f"train on only this many samples (at least {SMALLEST_LIMIT}), "
                                                  "to check that everything runs")
    args = parser.parse_args()
    if args.limit is not None and args.limit < SMALLEST_LIMIT:
        parser.error(f"--limit must be at least {SMALLEST_LIMIT}")

    result = train(Path(args.data), Path(args.out), args.limit)
    print(f"test: top-1 {result['top1']:.2%}  top-8 {result['top8']:.2%}  ({result['test_samples']} samples, "
          f"{result['parameters']:,} parameters, {result['minutes']} min)")
    print(f"gate (top-8 >= {GATE_TOP_8:.0%}):", "PASS" if result["top8"] >= GATE_TOP_8 else "FAIL")
