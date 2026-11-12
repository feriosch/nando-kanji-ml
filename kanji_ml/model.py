"""The recognition model: HanCharPI's conv-net idea, shrunk for the browser.

Three conv blocks and global average pooling replace HanCharPI's
Flatten -> Dense(2048), so the model has under 2M parameters and exports to a
few MB (ADR 0005). It takes the preprocessed 48x48 image as it is, 0-255.
"""

import keras
from keras import layers

SIZE = 48
FILTERS = (64, 128, 256)  # one block each, halving the image: 48 -> 24 -> 12 -> 6


def build_model(classes: int) -> keras.Model:
    inputs = keras.Input((SIZE, SIZE, 1))
    x = layers.Rescaling(1 / 255)(inputs)
    for filters in FILTERS:
        for _ in range(2):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(classes, activation="softmax", dtype="float32")(x)
    return keras.Model(inputs, outputs, name="kanji")
