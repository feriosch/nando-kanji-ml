import numpy as np

from kanji_ml.evaluate import top_k_accuracy

SCORES = np.array([
    [0.10, 0.60, 0.20, 0.10],  # best guesses: 1, 2, then 0 and 3 tied
    [0.50, 0.05, 0.40, 0.05],  # 0, 2, ...
    [0.20, 0.30, 0.10, 0.40],  # 3, 1, 0, 2
])
LABELS = np.array([1, 2, 2])


def test_top_1_counts_the_samples_whose_best_guess_is_right():
    assert top_k_accuracy(SCORES, LABELS, 1) == 1 / 3  # only the first


def test_top_k_counts_the_samples_whose_label_is_among_the_k_best_guesses():
    assert top_k_accuracy(SCORES, LABELS, 2) == 2 / 3  # the first two
    assert top_k_accuracy(SCORES, LABELS, 3) == 2 / 3  # the third's label is its 4th guess
    assert top_k_accuracy(SCORES, LABELS, 4) == 1.0


def test_a_label_tied_at_the_kth_place_counts_as_found():
    scores = np.array([[0.50, 0.25, 0.25]])  # the label, class 2, ties class 1 for second place

    assert top_k_accuracy(scores, np.array([2]), 2) == 1.0
    assert top_k_accuracy(scores, np.array([2]), 1) == 0.0
