"""
Both enhancement options ("bcd" and "fw") on a small instance of the Fig. 3 setup:

  * Phi never increases across outer iterations (forward step accepted only if it
    lowers Phi, enhancement steps never increase Phi, truncation rule (123) keeps
    Phi <= Phi of the previous iteration);
  * the graph-only, node-only and combined regimes all run;
  * on an easy instance (small rho, small p_max) the combined regime recovers the
    classes with both options.

Run:  pytest tests/   or   python tests/test_enhancements.py
"""

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from cogent import CoGEnT, LabelAwareLoss  # noqa: E402
from experiments.synthetic import construct_controlled_graph, make_clusters_fixed_rho  # noqa: E402

K, M = 3, 12


def _instance(rho=1.0, p=0.17, seed=0):
    rng = np.random.default_rng(seed)
    ang = 2 * np.pi * np.arange(K) / K
    X, y = make_clusters_fixed_rho(rho, 4.0 * np.c_[np.cos(ang), np.sin(ang)], M, rng)
    A = construct_controlled_graph(y, p, rng=rng)
    train = np.zeros(K * M, dtype=bool)
    for c in range(K):
        train[rng.choice(np.flatnonzero(y == c), 3, replace=False)] = True
    R = float(np.linalg.norm(X, axis=1).max())
    return LabelAwareLoss(X, y, train, K, nu=2.0, a_radius=R), A, y, train


def test_both_enhancements(verbose=False):
    loss, A, y, train = _instance()
    for enh in ("bcd", "fw"):
        for mu in (np.inf, 0.0, 0.1):
            res = CoGEnT(loss, A, gamma=2.0, mu=mu, enhancement=enh, seed=0).solve()
            h = res["hist"]
            acc = float(np.mean(res["y_pred"][~train] == y[~train]))
            if verbose:
                print(f"enhancement={enh:3s} mu={mu:<4}  k={res['k']:2d}  Phi={res['objective']: .5f}  "
                      f"outer its={len(h) - 1:2d}  test acc={acc:.3f}")
            assert np.all(np.diff(h) <= 1e-9 * np.maximum(1.0, np.abs(h[:-1]))), "Phi increased"
            if mu == 0.1:
                assert acc == 1.0, f"combined regime failed on an easy instance ({enh})"


if __name__ == "__main__":
    test_both_enhancements(verbose=True)
    print("OK: both enhancement options decrease Phi monotonically and recover the easy instance.")
