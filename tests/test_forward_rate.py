"""
Numerical check of the forward-step rate (115)-(116) of App. F.

1. Soft iterate (no tied node parameters).  For a random (Omega, Theta), a random
   nonnegative unit vector e and a random atom parameter theta, compare

       rate = -wg * e^T Abar0 e + sum_v e_v^2 d_v(theta)                      (115)
       d_v(theta) = wf (f_v(theta) - l_v) + ws <sum_u g_vu, theta - theta_v>
                    + wg (Abar0 L)_vv                                         (116)

   with the one-sided finite difference of the exact objective (104) along path
   (114): Omega(t) = [(1 - t e^2) * Omega, t e^2].  They must agree.

2. Hard iterate (tied node parameters), full move of a node set S (u = 1_S).
   (115)-(116) alone misses the cost of separating tied pairs;
   CoGEnT.exact_rate adds it and must agree with the finite difference.

Run:  pytest tests/   or   python tests/test_forward_rate.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cogent import CoGEnT, LabelAwareLoss, SquaredLoss  # noqa: E402


def _problem():
    rng = np.random.default_rng(0)
    n, d, K = 30, 2, 3
    X = rng.normal(size=(n, d))
    A = (rng.random((n, n)) < 0.3).astype(float)
    A = np.triu(A, 1)
    A = A + A.T
    y = rng.integers(0, K, n)
    train = rng.random(n) < 0.5
    losses = [("squared", SquaredLoss(X)), ("label-aware (22)", LabelAwareLoss(X, y, train, K, nu=1.5))]
    return A, losses


def _soft_case(loss, A, gamma, mu, k, seed):
    rng = np.random.default_rng(seed)
    n = loss.n
    s = CoGEnT(loss, A, gamma, mu)
    Om = rng.dirichlet(np.ones(k), size=n)
    Th = loss.project(rng.normal(size=(k, loss.dim)))
    s._set_state(Om, Th)
    e = np.abs(rng.normal(size=n)) * (rng.random(n) < 0.6)
    e /= np.linalg.norm(e)
    theta = loss.project(rng.normal(size=(1, loss.dim)))[0]

    G, base = s._rate_terms()
    d = s._d_of_theta(theta, G, base)
    rate = -s.wg * e @ s.Abar0 @ e + np.sum(e**2 * d)

    phi0 = s.objective()

    def phi(t):
        tt = CoGEnT(loss, A, gamma, mu)
        tt._set_state(
            np.hstack([(1 - t * e[:, None] ** 2) * Om, t * e[:, None] ** 2]),
            np.vstack([Th, theta[None]]),
        )
        return tt.objective()

    fd = [(phi(h) - phi0) / h for h in (1e-4, 1e-5, 1e-6)]
    return rate, fd


def _tied_case(loss, A, mu, seed):
    rng = np.random.default_rng(100 + seed)
    n = loss.n
    s = CoGEnT(loss, A, 2.0, mu)
    Om = np.eye(3)[rng.integers(0, 3, n)]  # hard rows: many tied node parameters
    Th = loss.project(rng.normal(size=(3, loss.dim)))
    s._set_state(Om, Th)
    S = rng.random(n) < 0.3
    theta = loss.project(rng.normal(size=(1, loss.dim)))[0]
    G, base = s._rate_terms()
    lin = s._rate_of_set(S, s._d_of_theta(theta, G, base), s._graph_B())
    exact = s.exact_rate(S, theta)
    phi0 = s.objective()
    u = S.astype(float)

    def phi(t):
        tt = CoGEnT(loss, A, 2.0, mu)
        tt._set_state(np.hstack([(1 - t * u)[:, None] * Om, (t * u)[:, None]]), np.vstack([Th, theta[None]]))
        return tt.objective()

    fd = (phi(1e-6) - phi0) / 1e-6
    return fd, lin, exact


def test_rate_soft(verbose=False):
    A, losses = _problem()
    worst = 0.0
    for name, loss in losses:
        for mu in (0.3, 2.0, np.inf):
            for seed in range(3):
                rate, fd = _soft_case(loss, A, gamma=2.0, mu=mu, k=4, seed=seed)
                rel = abs(fd[-1] - rate) / max(1.0, abs(rate))
                worst = max(worst, rel)
                if verbose:
                    print(
                        f"{name:18s} mu={mu:<5} seed={seed}  rate(115)={rate: .6f}  "
                        f"finite-diff={fd[0]: .6f} {fd[1]: .6f} {fd[2]: .6f}  rel.err={rel:.1e}"
                    )
    if verbose:
        print(f"max relative error: {worst:.2e}")
    assert worst < 1e-4, "rate formula (115)-(116) does not match the objective"


def test_rate_tied(verbose=False):
    A, losses = _problem()
    worst_lin, worst_exact = 0.0, 0.0
    for name, loss in losses:
        for mu in (0.3, np.inf):
            for seed in range(3):
                fd, lin, exact = _tied_case(loss, A, mu, seed)
                worst_lin = max(worst_lin, abs(fd - lin) / max(1.0, abs(fd)))
                worst_exact = max(worst_exact, abs(fd - exact) / max(1.0, abs(fd)))
                if verbose:
                    print(
                        f"{name:18s} mu={mu:<5} seed={seed}  finite-diff={fd: .5f}  "
                        f"(115) only={lin: .5f}  with tie term={exact: .5f}"
                    )
    if verbose:
        print(f"max relative error: (115) only {worst_lin:.2e},  with tie term {worst_exact:.2e}")
    assert worst_exact < 1e-4, "tie-aware rate does not match the objective"
    assert worst_lin > 1e-2, "expected (115)-(116) alone to miss the tie term on hard rows"


if __name__ == "__main__":
    print("1. soft iterate, path (114) with a random unit vector e:")
    test_rate_soft(verbose=True)
    print("OK: (115)-(116) is the exact one-sided derivative of (104) along (114).\n")
    print("2. hard iterate (tied theta_v), full move of a node set S:")
    test_rate_tied(verbose=True)
    print("OK: with ties, the exact rate of a full move is (115)-(116) plus the tie-separation term.")
