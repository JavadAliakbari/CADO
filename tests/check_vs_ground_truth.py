"""
Is a failure of a regime a failure of the convex program or of the solver?
(slow; not collected by pytest)

For instances of the Fig. 3 setup, compare the objective Phi reached by the
solver with Phi at the ground-truth partition (Omega = one-hot class indicators,
Theta = minimizer of the characteristic problem (5) for that Omega, computed by
the same Theta-step run to convergence).  If Phi(solver) <= Phi(truth), the
solver found a point at least as good as the perfect-recovery candidate, so any
recovery failure on that instance is a property of the program, not of the
optimizer.

Run:  python tests/check_vs_ground_truth.py [--enhancement fw]
"""

import argparse
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import experiments.fig3_sweeps as F  # noqa: E402
from cogent import CoGEnT, LabelAwareLoss  # noqa: E402


def phi_truth(loss, A, mu, y):
    s = CoGEnT(loss, A, F.GAMMA, mu)
    Om = np.eye(F.K)[y]
    Th0 = np.vstack(
        [loss.weighted_argmin((y == c).astype(float), np.zeros(loss.dim)) for c in range(F.K)]
    )
    s._set_state(Om, Th0)
    s._theta_step(iters=500)
    return s.objective()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--enhancement", choices=["bcd", "fw"], default="bcd")
    F.ENHANCEMENT = ap.parse_args().enhancement
    print(f"enhancement: {F.ENHANCEMENT}")
    print(" p_max  rho   mu    seed |  k   test acc |  Phi(solver)   Phi(truth)   solver - truth")
    worst = -np.inf
    for p, rho in [(0.20, 4.0), (0.28, 4.0), (0.36, 4.0), (0.24, 2.0), (0.24, 6.0)]:
        for mu in [np.inf, 0.03, 1.0]:
            for seed in range(3):
                X, y, A, train = F.make_instance(p, rho, seed)
                R = float(np.linalg.norm(X, axis=1).max())  # same Theta_f as F.solve
                loss = LabelAwareLoss(X, y, train, F.K, nu=F.NU, a_radius=R)
                r = F.solve(X, y, A, train, mu, seed)
                pt = phi_truth(loss, A, mu, y)
                gap = r["objective"] - pt
                worst = max(worst, gap / max(1.0, abs(pt)))
                acc = F.test_accuracy(r, y, train)
                print(
                    f" {p:.2f}  {rho:4.1f}  {mu:<5}  {seed}   | {r['k']:3d}   {acc:.3f}   | "
                    f"{r['objective']:11.4f}  {pt:11.4f}   {gap:+.4f}"
                )
    print(f"\nlargest relative excess of the solver over the ground-truth partition: {worst:+.2e}")
    print("(<= 0 means the solver always reached a point at least as good as the true partition)")
