"""Training augmentation: small changes a finger could make, applied to batches of 48x48x1 images, 0-255.

Stroke width varies only mildly. Drawings are drawn to match ETL's typical
stroke (the spec on nando-kanji-ml#1), and ETL's own writers already span
2.6-4.6 pixels, so this only keeps the model from leaning on one exact width.
"""

import keras
import tensorflow as tf

ROTATION = 10 / 360  # up to 10 degrees either way, as a fraction of a full turn
ZOOM = 0.1  # up to 10% bigger or smaller
SHIFT = 0.06  # up to 6% of the image (about 3 pixels) in each direction: where the kanji sits after cropping


def geometric() -> keras.Sequential:
    """A slight tilt, size and offset. Blank space fills with 0, the background."""
    return keras.Sequential([
        keras.layers.RandomRotation(ROTATION, fill_mode="constant"),
        keras.layers.RandomZoom(ZOOM, fill_mode="constant"),
        keras.layers.RandomTranslation(SHIFT, SHIFT, fill_mode="constant"),
    ])


def _blend_with(images, neighbourhood):
    """Half the image, half its 3x3 neighbourhood: strokes change by about a pixel in all."""
    return 0.5 * images + 0.5 * neighbourhood


def thicker(images):
    return _blend_with(images, tf.nn.max_pool2d(images, 3, 1, "SAME"))


def thinner(images):
    return _blend_with(images, -tf.nn.max_pool2d(-images, 3, 1, "SAME"))


def vary_stroke_width(images):
    """Each image, at random: a third unchanged, a third thicker, a third thinner."""
    choice = tf.random.uniform([tf.shape(images)[0], 1, 1, 1])
    return tf.where(choice < 1 / 3, thicker(images), tf.where(choice < 2 / 3, thinner(images), images))
