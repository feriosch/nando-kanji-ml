"""The recognition model: HanCharPI's conv-net idea, shrunk for the browser.

Three conv blocks and global average pooling replace HanCharPI's
Flatten -> Dense(2048), so the model has under 2M parameters (ADR 0005). As
float32 weights that's about 7.6 MB; step 6 of nando-kanji-ml#1 measures the
exported size per runtime. It takes the preprocessed 48x48 image as it is, 0-255.
"""

import keras
from keras import layers

from kanji_ml.preprocess import SIZE

FILTERS = (64, 128, 256)  # one block each, halving the image: 48 -> 24 -> 12 -> 6
CONVOLUTIONS_PER_BLOCK = 2  # 3x3 each, so a block sees 5x5 before pooling
DROPOUT = 0.3  # of the pooled features, during training only


def build_model(classes: int) -> keras.Model:
    inputs = keras.Input((SIZE, SIZE, 1))
    x = layers.Rescaling(1 / 255)(inputs)
    for filters in FILTERS:
        for _ in range(CONVOLUTIONS_PER_BLOCK):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(DROPOUT)(x)
    # float32 even when training in mixed precision, so the probabilities don't lose precision
    outputs = layers.Dense(classes, activation="softmax", dtype="float32")(x)
    return keras.Model(inputs, outputs, name="kanji")
