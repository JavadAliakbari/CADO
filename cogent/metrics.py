"""Evaluation metric used in the experiments."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def clustering_accuracy(pred, true) -> float:
    """Best one-to-one matching accuracy between predicted and true labels
    (Hungarian matching; predicted clusters left unmatched count as errors)."""
    pred = np.asarray(pred)
    true = np.asarray(true)
    _, pi = np.unique(pred, return_inverse=True)
    _, ti = np.unique(true, return_inverse=True)
    C = np.zeros((pi.max() + 1, ti.max() + 1), dtype=int)
    np.add.at(C, (pi, ti), 1)
    r, c = linear_sum_assignment(-C)
    return float(C[r, c].sum()) / len(true)
