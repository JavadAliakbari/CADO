"""
cogent -- CoGEnT-style atomic conditional-gradient solver for the unified
graph + node-information problem (14) of the paper (App. F, Algorithm 1).

    from cogent import CoGEnT, LabelAwareLoss, clustering_accuracy

    loss = LabelAwareLoss(X, y_obs, train_mask, n_classes=K, nu=2.0)
    res = CoGEnT(loss, A, gamma=2.0, mu=0.1).solve()    # enhancement="bcd" (default) or "fw"
    acc = clustering_accuracy(res["y_pred"], y_true)

Any convex node loss can be used by subclassing `NodeLoss` (cogent/losses.py).
"""

from .losses import (
    LabelAwareLoss,
    NodeLoss,
    SquaredLoss,
    project_rows_simplex,
    simplex_argmin_linear,
)
from .metrics import clustering_accuracy
from .solver import CoGEnT, cogent_solve

__all__ = [
    "CoGEnT",
    "cogent_solve",
    "NodeLoss",
    "SquaredLoss",
    "LabelAwareLoss",
    "clustering_accuracy",
    "project_rows_simplex",
    "simplex_argmin_linear",
]
