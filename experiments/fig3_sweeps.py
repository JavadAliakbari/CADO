"""
Fig. 3 (center and right panels) of the paper with the CoGEnT-style solver of
App. F (package `cogent`).  All three regimes are the SAME convex program (14)
solved by the SAME algorithm; only mu changes:

    node-information only :  mu = inf      (graph term dropped)
    graph only            :  mu -> 0       (node terms dropped; Theta decoded at the end)
    combined              :  0 < mu < inf  (mu chosen by cross-validation, see below)

Setup of Sec. 6: K = 5 classes, 25 nodes per class in R^2, class centroids on a
circle of radius 4, 5 observed labels per class, label-aware node loss (22) with
nu = 2, gamma = 2, prediction phi(theta_v) = argmax_c pi_c, accuracy on the
unlabeled (test) nodes.  Features have exact cluster diameter rho and the graph
has exact misconnection rate p_max (experiments/synthetic.py).

    center panel : accuracy vs p_max   at rho   = RHO_FIXED (4.0)
    right panel  : accuracy vs rho     at p_max = P_FIXED   (0.36)

Choice of mu for the combined regime (--mu-select):
    "cv"     (default) mu is chosen per instance from MU_GRID by 3-fold
             cross-validation on the observed training labels only (no test
             information); the reported accuracy is that of the full-label solve
             with the chosen mu.
    "oracle" best test accuracy over MU_GRID (an upper envelope, for reference;
             it is also drawn as a dashed line in the "cv" plot).
    "fixed"  a single mu (MU_FIXED).

Run from the repository root:

    python experiments/fig3_sweeps.py                      # enhancement "bcd", 20 trials
    python experiments/fig3_sweeps.py --enhancement fw --trials 10
    python experiments/fig3_sweeps.py --trials 1 --out results/smoke     # quick check

Outputs (in --out, default results/enhancement_<enhancement>):
    fig3_sweeps.png, fig3_center_vs_pmax.csv, fig3_right_vs_rho.csv, fig3_raw.npy (every job)
"""

import os

# one BLAS thread per worker process (the sweep is parallel over processes)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from cogent import LabelAwareLoss, cogent_solve  # noqa: E402
from experiments.synthetic import (  # noqa: E402
    construct_controlled_graph,
    empirical_pmax,
    make_clusters_fixed_rho,
)

# ------------------------------------------------------------------
# Setup (Sec. 6)
# ------------------------------------------------------------------
K = 5
N_PER_CLASS = 25
R_RING = 4.0
TRAIN_PER_CLASS = 5  # observed labels per class
GAMMA = 2.0
NU = 2.0

RHO_FIXED = 4.0  # center panel: accuracy vs p_max
P_FIXED = 0.36  # right panel: accuracy vs rho (graph only is partially informative here, ~0.5)
P_GRID = np.arange(1, 12) / N_PER_CLASS  # realized p_max = m/N exactly, 0.04 ... 0.44
RHO_GRID = np.arange(1.0, 6.01, 0.5)

MU_GRID = [0.01, 0.03, 0.1, 1.0, 10.0, 100.0]
MU_FIXED = 0.03
MU_SELECT = "cv"
CV_FOLDS = 3

N_TRIALS = 20
# max_atoms = n (no cap); max_outer = 2n outer iterations
SOLVER_KW = dict(max_atoms=K * N_PER_CLASS, max_outer=2 * K * N_PER_CLASS)
# enhancement step: "bcd" (row block-coordinate descent + majorize-minimize on Theta)
# or "fw" (Frank-Wolfe on rows and atoms with line search, as written in App. F.3)
ENHANCEMENT = "bcd"
INNER_CYCLES = {"bcd": 3, "fw": 3}  # enhancement cycles per outer iteration
# random node order in the regular graph construction (a different graph per seed);
# False gives the deterministic banded graph of the earlier runs
RANDOM_GRAPH = True


# ------------------------------------------------------------------
# Instances
# ------------------------------------------------------------------
def centroids():
    ang = 2.0 * np.pi * np.arange(K) / K
    return R_RING * np.c_[np.cos(ang), np.sin(ang)]


def make_instance(p, rho, seed):
    """Features with diameter rho, graph with p_max = p, and a train/test split.

    Random streams: the split depends only on `seed`, the features on (seed, rho),
    the graph on (seed, p).  So along the rho axis the graph and the split are
    fixed (graph-only accuracy is flat), and along the p axis the features are fixed."""
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(K), N_PER_CLASS)
    train = np.zeros(K * N_PER_CLASS, dtype=bool)
    for c in range(K):
        idx = np.flatnonzero(labels == c)
        train[rng.choice(idx, TRAIN_PER_CLASS, replace=False)] = True
    rng_x = np.random.default_rng([seed, int(round(1000 * rho))])
    X, y = make_clusters_fixed_rho(rho, centroids(), N_PER_CLASS, rng_x)
    rng_g = np.random.default_rng([seed, int(round(1000 * p)), 99]) if RANDOM_GRAPH else None
    A = construct_controlled_graph(y, p, rng=rng_g)
    return X, y, A, train


def solve(X, y, A, train, mu, seed):
    """Solve (14) for one mu.  Theta_f = {||a|| <= R} x simplex with R = max_v ||x_v||:
    the bound is needed by the Frank-Wolfe atom step (the LMO over a in R^d is
    unbounded) and is used for both enhancement options so they solve the same problem."""
    R = float(np.linalg.norm(X, axis=1).max())
    loss = LabelAwareLoss(X, y, train, K, nu=NU, a_radius=R)
    return cogent_solve(
        loss, A, gamma=GAMMA, mu=mu, seed=seed, enhancement=ENHANCEMENT,
        inner_cycles=INNER_CYCLES[ENHANCEMENT], **SOLVER_KW,
    )


def test_accuracy(res, y, train):
    """Accuracy of y_pred = phi(theta_v) on the unlabeled nodes."""
    return float(np.mean(res["y_pred"][~train] == y[~train]))


def cv_select_mu(X, y, A, train, seed):
    """Choose mu from MU_GRID by CV_FOLDS-fold cross-validation over the training
    labels (stratified folds).  For every fold, the fold's labels are hidden, (14)
    is solved, and the hidden labels are predicted.  Ties are broken by
    `select_from_cv`."""
    rng = np.random.default_rng([seed, 7])
    tr = np.flatnonzero(train)
    fold = np.empty(len(tr), dtype=int)
    for c in range(K):
        idx = np.flatnonzero(y[tr] == c)
        fold[idx] = rng.permutation(len(idx)) % CV_FOLDS
    scores = []
    for mu in MU_GRID:
        hit = 0
        for f in range(CV_FOLDS):
            fit = train.copy()
            fit[tr[fold == f]] = False
            res = solve(X, y, A, fit, mu, seed)
            hit += int(np.sum(res["y_pred"][tr[fold == f]] == y[tr[fold == f]]))
        scores.append(hit / len(tr))
    return MU_GRID[select_from_cv(scores)], list(scores)


def select_from_cv(scores):
    """Index of the best CV score.  With only 25 labels the CV score often ties;
    ties go to the candidate whose neighbour-averaged score is largest, i.e. to the
    interior of the plateau of best values rather than its edge (then the smaller
    index)."""
    s = np.asarray(scores, dtype=float)
    tied = np.flatnonzero(s >= s.max() - 1e-12)
    smooth = [s[max(i - 1, 0) : i + 2].mean() for i in range(len(s))]
    return int(tied[np.argmax([smooth[i] for i in tied])])


# ------------------------------------------------------------------
# Jobs
# ------------------------------------------------------------------
def job(task):
    """One (regime, p, rho, seed) job.  A "comb" job solves every mu in MU_GRID
    (for the oracle envelope) and runs the cross-validation."""
    kind, p, rho, seed = task
    X, y, A, train = make_instance(p, rho, seed)
    out = dict(kind=kind, p=p, rho=rho, seed=seed)
    t0 = time.time()
    if kind == "node":
        out["acc"] = test_accuracy(solve(X, y, A, train, np.inf, seed), y, train)
    elif kind == "graph":
        out["acc"] = test_accuracy(solve(X, y, A, train, 0.0, seed), y, train)
    else:
        accs = [test_accuracy(solve(X, y, A, train, mu, seed), y, train) for mu in MU_GRID]
        out["acc_per_mu"] = accs
        out["acc_oracle"] = max(accs)
        out["acc_fixed"] = (
            accs[MU_GRID.index(MU_FIXED)]
            if MU_FIXED in MU_GRID
            else test_accuracy(solve(X, y, A, train, MU_FIXED, seed), y, train)
        )
        if MU_SELECT == "cv":
            mu_cv, cv_scores = cv_select_mu(X, y, A, train, seed)
            out["mu_cv"] = mu_cv
            out["cv_scores"] = cv_scores
            out["acc_cv"] = accs[MU_GRID.index(mu_cv)]
        out["pmax_emp"] = empirical_pmax(A, y)
    out["time"] = time.time() - t0
    return out


def build_tasks():
    tasks = []
    for s in range(N_TRIALS):
        # center: accuracy vs p_max at rho = RHO_FIXED (node only does not depend on p)
        tasks.append(("node", float(P_GRID[0]), RHO_FIXED, s))
        for p in P_GRID:
            tasks.append(("graph", float(p), RHO_FIXED, s))
            tasks.append(("comb", float(p), RHO_FIXED, s))
        # right: accuracy vs rho at p_max = P_FIXED
        for r in RHO_GRID:
            tasks.append(("graph", P_FIXED, float(r), s))
            tasks.append(("node", P_FIXED, float(r), s))
            tasks.append(("comb", P_FIXED, float(r), s))
    tasks = list(dict.fromkeys(tasks))
    cost = {"comb": 0, "node": 1, "graph": 2}  # expensive jobs first (load balance)
    return sorted(tasks, key=lambda t: cost[t[0]])


def lookup(results, kind, p, rho, key="acc"):
    v = [r[key] for r in results if r["kind"] == kind and np.isclose(r["p"], p) and np.isclose(r["rho"], rho)]
    return np.array(v, dtype=float)


P_PLOT_MAX = None  # optional upper limit of the p_max axis in the plot/CSV (e.g. 0.32 as in the paper)


def curves(results, axis):
    """mean/std of every curve along the p axis (center) or the rho axis (right)."""
    comb_key = {"cv": "acc_cv", "oracle": "acc_oracle", "fixed": "acc_fixed"}[MU_SELECT]
    xs = P_GRID if axis == "p" else RHO_GRID
    if axis == "p" and P_PLOT_MAX is not None:
        xs = xs[xs <= P_PLOT_MAX + 1e-9]
    rows = []
    for x in xs:
        p, rho = (float(x), RHO_FIXED) if axis == "p" else (P_FIXED, float(x))
        node = lookup(results, "node", P_GRID[0] if axis == "p" else P_FIXED, rho)
        graph = lookup(results, "graph", p, rho)
        comb = lookup(results, "comb", p, rho, comb_key)
        orc = lookup(results, "comb", p, rho, "acc_oracle")
        rows.append([x, node.mean(), node.std(), graph.mean(), graph.std(), comb.mean(), comb.std(), orc.mean(), orc.std()])
    return np.array(rows)


def plot(cen, rig, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lab = {"cv": "Combined ($\\mu$ by CV on train labels)", "oracle": "Combined (best $\\mu$)", "fixed": f"Combined ($\\mu$={MU_FIXED})"}[MU_SELECT]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"enhancement step: {ENHANCEMENT}", fontsize=9)
    for ax, C, xl, title in (
        (axes[0], cen, "graph misconnection rate $p_{\\max}$", f"$\\rho$ = {RHO_FIXED}"),
        (axes[1], rig, "within-class variability $\\rho$", f"$p_{{\\max}}$ = {P_FIXED}"),
    ):
        for (m, s, name, col, mk) in (
            (3, 4, "Graph-only", "tab:green", "s"),
            (1, 2, "Node-only", "tab:blue", "o"),
            (5, 6, lab, "tab:red", "D"),
        ):
            ax.plot(C[:, 0], C[:, m], marker=mk, color=col, label=name)
            ax.fill_between(C[:, 0], C[:, m] - C[:, s], np.minimum(C[:, m] + C[:, s], 1), color=col, alpha=0.12)
        if MU_SELECT != "oracle":
            ax.plot(C[:, 0], C[:, 7], ls="--", lw=1, color="tab:red", alpha=0.6, label="Combined (best $\\mu$ in grid)")
        ax.set_xlabel(xl)
        ax.set_ylabel("test accuracy")
        ax.set_ylim(0, 1.03)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def _init_worker(enhancement, mu_select):
    global ENHANCEMENT, MU_SELECT
    ENHANCEMENT, MU_SELECT = enhancement, mu_select


def main():
    global MU_SELECT, N_TRIALS, ENHANCEMENT, P_PLOT_MAX
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--mu-select", choices=["cv", "oracle", "fixed"], default=MU_SELECT)
    ap.add_argument("--enhancement", choices=["bcd", "fw"], default=ENHANCEMENT)
    ap.add_argument("--out", default=None, help="output directory relative to the repository root "
                    "(default results/enhancement_<enhancement>)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--replot", action="store_true", help="re-plot <out>/fig3_raw.npy without solving")
    ap.add_argument("--p-plot-max", type=float, default=None, help="crop the p_max axis of the plot/CSV")
    args = ap.parse_args()
    MU_SELECT, N_TRIALS, ENHANCEMENT = args.mu_select, args.trials, args.enhancement
    P_PLOT_MAX = args.p_plot_max
    tag = "" if P_PLOT_MAX is None else f"_pmax{P_PLOT_MAX:g}"

    out_dir = os.path.join(_ROOT, args.out or f"results/enhancement_{ENHANCEMENT}")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    if args.replot:
        results = list(np.load(os.path.join(out_dir, "fig3_raw.npy"), allow_pickle=True))
    else:
        tasks = build_tasks()
        print(
            f"{len(tasks)} solver jobs, mu grid {MU_GRID}, mu selection: {MU_SELECT}, "
            f"enhancement: {ENHANCEMENT}",
            flush=True,
        )
        results = []
        with ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=(ENHANCEMENT, MU_SELECT)) as ex:
            for i, r in enumerate(ex.map(job, tasks, chunksize=1)):
                results.append(r)
                if (i + 1) % max(1, len(tasks) // 20) == 0:
                    print(f"  {i + 1}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
        np.save(os.path.join(out_dir, "fig3_raw.npy"), np.array(results, dtype=object), allow_pickle=True)

    cen, rig = curves(results, "p"), curves(results, "rho")
    head = ["x", "node", "node_std", "graph", "graph_std", "comb", "comb_std", "comb_oracle", "comb_oracle_std"]
    for name, C in (("fig3_center_vs_pmax", cen), ("fig3_right_vs_rho", rig)):
        with open(os.path.join(out_dir, name + tag + ".csv"), "w", newline="") as f:
            w = csv.writer(f, delimiter=" ")
            w.writerow(head)
            w.writerows([[f"{v:.6g}" for v in row] for row in C])
    plot(cen, rig, os.path.join(out_dir, f"fig3_sweeps{tag}.png"))

    for name, C in (("center (vs p_max)", cen), ("right (vs rho)", rig)):
        print(f"\n{name}:   x | node | graph | combined | combined-oracle")
        for row in C:
            flag = "" if row[5] >= max(row[1], row[3]) - 1e-9 else "   <-- combined below a baseline"
            print(f"  {row[0]:5.2f} | {row[1]:.3f} | {row[3]:.3f} | {row[5]:.3f} | {row[7]:.3f}{flag}")
    print(f"\nsaved to {out_dir}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
