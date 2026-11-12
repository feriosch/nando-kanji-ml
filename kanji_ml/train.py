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

from kanji_ml.evaluate import top_k_accuracy
from kanji_ml.model import build_model

SAMPLES_PER_FILE = 4 * 2965  # 4 writers per ETL9G file, each writing every kanji once
VALIDATION_FILES = 2
GATE_TOP_8 = 0.95  # ADR 0005: ship only if top-8 >= 95%


def _augmenter() -> keras.Sequential:
    """Small changes a finger could make: a slight tilt, size and offset. Blank space fills with 0."""
    return keras.Sequential([
        keras.layers.RandomRotation(10 / 360, fill_mode="constant"),
        keras.layers.RandomZoom(0.1, fill_mode="constant"),
        keras.layers.RandomTranslation(0.06, 0.06, fill_mode="constant"),
    ])


def _dataset(x, y, batch, augment=None, shuffle=False):
    data = tf.data.Dataset.from_tensor_slices((x[..., None], y))
    if shuffle:
        data = data.shuffle(50_000, reshuffle_each_iteration=True)
    data = data.batch(batch).map(lambda a, b: (tf.cast(a, tf.float32), b), num_parallel_calls=tf.data.AUTOTUNE)
    if augment is not None:
        data = data.map(lambda a, b: (augment(a, training=True), b), num_parallel_calls=tf.data.AUTOTUNE)
    return data.prefetch(tf.data.AUTOTUNE)


def train(data_dir: Path, out_dir: Path, epochs: int, batch: int, limit: int | None = None) -> dict:
    keras.mixed_precision.set_global_policy("mixed_float16")
    labels = json.loads((data_dir / "labels.json").read_text(encoding="utf-8"))
    train_npz, test_npz = np.load(data_dir / "train.npz"), np.load(data_dir / "test.npz")
    x, y = train_npz["x"], train_npz["y"]
    split = len(x) - VALIDATION_FILES * SAMPLES_PER_FILE
    x_train, y_train, x_val, y_val = x[:split], y[:split], x[split:], y[split:]
    x_test, y_test = test_npz["x"], test_npz["y"]
    if limit:  # a quick check that everything runs
        x_train, y_train, x_val, y_val = x_train[:limit], y_train[:limit], x_val[:limit // 10], y_val[:limit // 10]
        x_test, y_test = x_test[:limit // 10], y_test[:limit // 10]

    model = build_model(len(labels))
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", keras.metrics.SparseTopKCategoricalAccuracy(k=8, name="top8")],
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    best = out_dir / "kanji.keras"
    start = time.time()
    model.fit(
        _dataset(x_train, y_train, batch, augment=_augmenter(), shuffle=True),
        validation_data=_dataset(x_val, y_val, batch),
        epochs=epochs,
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-5),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(best, monitor="val_loss", save_best_only=True),
            keras.callbacks.CSVLogger(out_dir / "history.csv"),
        ],
        verbose=2,
    )

    scores = model.predict(_dataset(x_test, y_test, batch), verbose=0)
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
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--limit", type=int, help="train on only this many samples, to check that everything runs")
    args = parser.parse_args()

    r = train(Path(args.data), Path(args.out), args.epochs, args.batch, args.limit)
    print(f"test: top-1 {r['top1']:.2%}  top-8 {r['top8']:.2%}  ({r['test_samples']} samples, "
          f"{r['parameters']:,} parameters, {r['minutes']} min)")
    print("gate (top-8 >= 95%):", "PASS" if r["top8"] >= GATE_TOP_8 else "FAIL")
