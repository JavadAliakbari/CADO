"""
The solver only touches f_v through the NodeLoss interface (cogent/losses.py).
This test runs it with losses that provide NO closed forms, so the generic
code paths (numerical argmin, generic Theta-step) are used:

  * GenericSquared : 1/2 ||x_v - theta||^2 through the generic code path
                     (must reproduce SquaredLoss, which has closed forms),
  * BallHuber      : sum_c huber_delta(x_vc - theta_c) + indicator(||theta|| <= R)
                     (non-quadratic, constrained Theta_f).

Run:  pytest tests/   or   python tests/test_general_loss.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cogent import NodeLoss, SquaredLoss, clustering_accuracy, cogent_solve  # noqa: E402


class GenericSquared(NodeLoss):
    """Squared loss that implements only the three required methods."""

    smoothness = 1.0

    def __init__(self, X):
        self.X = X
        self.n, self.dim = X.shape

    def values(self, Theta):
        return 0.5 * ((self.X[:, None, :] - Theta[None]) ** 2).sum(-1)

    def pointwise(self, Theta_rows):
        return 0.5 * ((self.X - Theta_rows) ** 2).sum(-1)

    def grad_atoms(self, Theta, Wt):
        return np.einsum("vi,vid->id", Wt, Theta[None] - self.X[:, None, :])


class BallHuber(NodeLoss):
    """Huber loss with Theta_f = {||theta|| <= radius} (robust to outliers)."""

    smoothness = 1.0

    def __init__(self, X, delta=1.0, radius=10.0):
        self.X, self.delta, self.radius = X, delta, radius
        self.n, self.dim = X.shape

    def _h(self, r):
        a = np.abs(r)
        return np.where(a <= self.delta, 0.5 * r**2, self.delta * (a - 0.5 * self.delta))

    def values(self, Theta):
        return self._h(self.X[:, None, :] - Theta[None]).sum(-1)

    def pointwise(self, Theta_rows):
        return self._h(self.X - Theta_rows).sum(-1)

    def grad_atoms(self, Theta, Wt):
        g = np.clip(Theta[None] - self.X[:, None, :], -self.delta, self.delta)
        return np.einsum("vi,vid->id", Wt, g)

    def project(self, Theta):
        nrm = np.linalg.norm(Theta, axis=1, keepdims=True)
        return Theta * np.minimum(1.0, self.radius / np.maximum(nrm, 1e-12))


def instance(seed, K=4, m=20, p=0.2, sigma=0.9):
    """K Gaussian clusters with 4 outliers and an SBM-like graph."""
    rng = np.random.default_rng(seed)
    ang = 2 * np.pi * np.arange(K) / K
    C = 3.0 * np.c_[np.cos(ang), np.sin(ang)]
    y = np.repeat(np.arange(K), m)
    X = C[y] + sigma * rng.standard_normal((K * m, 2))
    X[rng.choice(K * m, 4, replace=False)] += rng.normal(0, 8, (4, 2))  # outliers
    same = y[:, None] == y[None, :]
    A = np.where(same, rng.random((K * m,) * 2) > p, rng.random((K * m,) * 2) < p).astype(float)
    A = np.triu(A, 1)
    return X, y, A + A.T


def test_generic_loss(verbose=False):
    for seed in range(3):
        X, y, A = instance(seed)
        r1 = cogent_solve(SquaredLoss(X), A, gamma=2.0, mu=0.05, seed=seed)
        r2 = cogent_solve(GenericSquared(X), A, gamma=2.0, mu=0.05, seed=seed)
        r3 = cogent_solve(BallHuber(X), A, gamma=2.0, mu=0.05, seed=seed)
        if verbose:
            print(
                f"seed={seed}  squared(closed form): acc={clustering_accuracy(r1['assign'], y):.3f} "
                f"phi={r1['objective']:.4f} | squared(generic): acc={clustering_accuracy(r2['assign'], y):.3f} "
                f"phi={r2['objective']:.4f} | Huber+ball combined: acc={clustering_accuracy(r3['assign'], y):.3f} "
                f"k={r3['k']}"
            )
        assert abs(r1["objective"] - r2["objective"]) <= 1e-3 * max(1.0, abs(r1["objective"]))
        assert np.all(np.linalg.norm(r3["Theta"], axis=1) <= 10.0 + 1e-9)


if __name__ == "__main__":
    test_generic_loss(verbose=True)
    print("OK: generic code path reproduces the closed-form solver; constrained Huber loss runs.")
