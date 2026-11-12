"""The shipping gate: top-8 accuracy on the ETL9G test set must reach GATE_TOP_8 (ADR 0005)."""

import numpy as np

GATE_TOP_8 = 0.95


def top_k_accuracy(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    """The share of samples whose label is among their k best-scored classes.

    A label is among them if fewer than k classes score strictly higher. A tie
    at the k-th place counts as found.
    """
    label_scores = scores[np.arange(len(labels)), labels]
    higher = (scores > label_scores[:, None]).sum(axis=1)
    return float(np.mean(higher < k))
