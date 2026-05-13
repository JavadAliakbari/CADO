"""
src/sweep_p_rho.py

Line-sweep experiment for the synergistic recovery region.

Produces two figures:
  (a) accuracy vs p_max  (with rho fixed)
  (b) accuracy vs rho    (with p_max fixed)

Each figure shows three curves, averaged over N_TRIALS independent runs:
  * feature_only  -- CoGEnT with mu = inf (no graph term)
  * graph_only    -- spectral clustering on A_graph (features ignored)
  * combined      -- CoGEnT with A_graph and finite mu

We use the EXACT-rho cluster generator and the EXACT-p_max graph constructor
from `cogent_empirical_validation.py`, so the realized rho and p_max equal
the targets up to integer rounding (granularity 1/N_PER_CLUSTER).
"""

import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.cluster.vq import kmeans2
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as _mp

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from cogent import cogent_structured_atoms_with_W
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "_cev", os.path.join(_HERE, "cogent_empirical_validation.py")
)
_cev = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_cev)
make_clusters_fixed_rho = _cev.make_clusters_fixed_rho
construct_controlled_graph = _cev.construct_controlled_graph
compute_rho = _cev.compute_rho
empirical_pmax = _cev.empirical_pmax

# ------------------------------------------------------------------
# Setup -- small, fast, clearly shows synergy.
# ------------------------------------------------------------------
K = 5
D = 2
N_PER_CLUSTER = 25  # n = 50 total
R_RING = 4.0  # centroid ring radius (d_min = 2 R sin(pi/K) ≈ 4.70)
GAMMA = 3.0  # gamma/K = 0.6
N_TRIALS = 20

# Fixed point of the OTHER axis is chosen near the boundary of the
# recovery region so neither feature-only nor graph-only succeeds alone,
# but their combination does.
P_FIXED = 0.2  # fixed for the rho-sweep   (above p* ≈ 0.117)
RHO_FIXED = 1.2  # fixed for the p-sweep     (above gamma/K = 0.6)

# --- p_max sweep ---------------------------------------------------------
# Paper definition (eq. 6):  p_max = max_{i,j} ( bar_rho_ij + 2 * delta_ij )
#
# Our `construct_controlled_graph` builds a fully-regular graph, so
# delta_ij = 0 in every block; the formula collapses to
#       p_max = max_{i,j} bar_rho_ij.
# With m mis-edges per node to each other class (and m missing intra edges
# per node), bar_rho_ij = bar_rho_ii = m / N_PER_CLUSTER, hence
#       p_max  =  m / N_PER_CLUSTER     for any m in {0, 1, ..., N}.
# Achievability: a regular graph requires N * (intra_deg) to be even.
#   - Even N: every value m/N is achievable exactly.
#   - Odd N:  only even m hits the target exactly; odd m gets bumped.
M_SWEEP = np.arange(0, 8)  # mis-edges per node
P_SWEEP = M_SWEEP / N_PER_CLUSTER  # corresponding exact p_max
RHO_SWEEP = np.linspace(0.5, 6, 12)


def block_pmax_breakdown(A, labels):
    """Return (bar_rho_max, two_delta_max, p_max) over all blocks (paper eq. 6)."""
    classes = np.unique(labels)
    Kk = len(classes)
    n_v_j = np.zeros((len(labels), Kk))
    for jj, cj in enumerate(classes):
        n_v_j[:, jj] = A[:, labels == cj].sum(axis=1)
    br_max, td_max, p_max = 0.0, 0.0, 0.0
    for ii, ci in enumerate(classes):
        v = np.where(labels == ci)[0]
        ni = len(v)
        for jj, cj in enumerate(classes):
            nj = (labels == cj).sum()
            nbar_v = (ni - n_v_j[v, jj]) if ii == jj else n_v_j[v, jj]
            br = float(nbar_v.sum() / (ni * nj))
            d = float(np.max(np.abs(br - nbar_v / nj)))
            p = br + 2 * d
            br_max = max(br_max, br)
            td_max = max(td_max, 2 * d)
            p_max = max(p_max, p)
    return br_max, td_max, p_max


# Combined regime: mu controls graph weight (1/(mu*n)).  Smaller mu = stronger
# graph term.  We try a small grid and take the BEST mu per cell.
MU_GRID = [0.5]

# Number of parallel worker processes.  Default = all physical cores; override
# with the SWEEP_WORKERS env var (e.g. SWEEP_WORKERS=4 python src/sweep_p_rho.py).
N_WORKERS = max(1, int(os.environ.get("SWEEP_WORKERS", _mp.cpu_count())))

SOLVER_KW = dict(
    max_iters=20,
    init_with_one_atom=False,
    inner_alt_cycles=2,
    pg_max_iter=10,
    pg_step_size=0.05,
    pg_tol=1e-8,
    eps=1e-12,
    truncation=True,
    verbose=False,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def cluster_accuracy(pred, true, K_local=K):
    C = np.zeros((K_local, K_local), int)
    for p, t in zip(pred, true):
        C[int(p) % K_local, int(t) % K_local] += 1
    row, col = linear_sum_assignment(-C)
    return C[row, col].sum() / len(true)


def spectral_cluster(A, K_local=K, seed=0):
    """Normalized-Laplacian spectral clustering (Ng-Jordan-Weiss)."""
    n = A.shape[0]
    deg = A.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    L_sym = np.eye(n) - (d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :])
    # Smallest K eigenvectors of L_sym = largest K of (I - L_sym).
    w, V = np.linalg.eigh(L_sym)
    U = V[:, :K_local]
    norms = np.linalg.norm(U, axis=1, keepdims=True)
    U_n = U / np.maximum(norms, 1e-12)
    centroids, labels = kmeans2(U_n, K_local, minit="++", seed=seed)
    return labels


def make_centroids(seed):
    """K equispaced centroids on a ring of radius R_RING in R^D (first 2 dims)."""
    angles = 2 * np.pi * np.arange(K) / K
    C = np.zeros((K, D))
    C[:, 0] = R_RING * np.cos(angles)
    C[:, 1] = R_RING * np.sin(angles)
    return C


def run_cell(p_target, rho_target, trial_seed):
    """Returns (acc_feature, acc_graph, acc_combined_best, rho_emp, pmax_emp)."""
    rng = np.random.default_rng(trial_seed)
    np_seed = int(rng.integers(0, 2**31))

    centroids = make_centroids(trial_seed)
    X, labels = make_clusters_fixed_rho(rho_target, centroids, N_PER_CLUSTER, rng)
    rho_emp = compute_rho(X, labels)

    A_graph = construct_controlled_graph(labels, p_target)
    pmax_emp = empirical_pmax(A_graph, labels)
    br_max, td_max, _ = block_pmax_breakdown(A_graph, labels)

    # ---- feature-only: CoGEnT, no graph term ----
    res_f = cogent_structured_atoms_with_W(
        X=X, gamma=GAMMA, mu=float("inf"), seed=np_seed, **SOLVER_KW
    )
    acc_f = cluster_accuracy(res_f["W"].argmax(0), labels)

    # ---- graph-only: spectral clustering on A_graph ----
    try:
        pred_g = spectral_cluster(A_graph, K_local=K, seed=np_seed)
        acc_g = cluster_accuracy(pred_g, labels)
    except Exception:
        acc_g = 1.0 / K  # degenerate

    # ---- combined: CoGEnT with graph term, best over MU_GRID ----
    acc_c_best = 0.0
    for mu in MU_GRID:
        res_c = cogent_structured_atoms_with_W(
            X=X, gamma=GAMMA, mu=mu, A_graph=A_graph, seed=np_seed, **SOLVER_KW
        )
        acc_c = cluster_accuracy(res_c["W"].argmax(0), labels)
        if acc_c > acc_c_best:
            acc_c_best = acc_c

    return acc_f, acc_g, acc_c_best, rho_emp, pmax_emp, br_max, td_max


def _run_cell_task(args):
    """Process-pool worker: unpack args, return (i, t, results...)."""
    i, t, p_target, rho_target, seed = args
    af, ag, ac, re, pe, br, td = run_cell(p_target, rho_target, seed)
    return i, t, af, ag, ac, re, pe, br, td


def sweep(axis_name, axis_vals, fixed_val):
    """Sweep one axis; the other is held at fixed_val. Returns mean and std arrays.

    Tasks for all (axis_value, trial) pairs are dispatched in parallel across
    `N_WORKERS` processes.
    """
    n = len(axis_vals)
    means = np.zeros((n, 3))
    stds = np.zeros((n, 3))
    rho_emp_log = np.zeros(n)
    pmax_emp_log = np.zeros(n)

    accs = np.zeros((n, N_TRIALS, 3))
    rho_e = np.zeros((n, N_TRIALS))
    p_e = np.zeros((n, N_TRIALS))
    br_e = np.zeros((n, N_TRIALS))
    td_e = np.zeros((n, N_TRIALS))

    tasks = []
    for i, v in enumerate(axis_vals):
        if axis_name == "p":
            p_target, rho_target = float(v), float(fixed_val)
        else:
            p_target, rho_target = float(fixed_val), float(v)
        for t in range(N_TRIALS):
            seed = 1000 * (i + 1) + t
            tasks.append((i, t, p_target, rho_target, seed))

    n_tasks = len(tasks)
    print(
        f"\n--- Sweeping {axis_name} ({n} values, {N_TRIALS} trials each, "
        f"{n_tasks} tasks, {N_WORKERS} workers) ---"
    )

    done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(_run_cell_task, t) for t in tasks]
        for fut in as_completed(futs):
            i, t, af, ag, ac, re, pe, br, td = fut.result()
            accs[i, t] = [af, ag, ac]
            rho_e[i, t] = re
            p_e[i, t] = pe
            br_e[i, t] = br
            td_e[i, t] = td
            done += 1
            if done % max(1, n_tasks // 20) == 0 or done == n_tasks:
                print(f"  [{done:4d}/{n_tasks}]  ({100.0 * done / n_tasks:5.1f}%)")

    means = accs.mean(axis=1)
    stds = accs.std(axis=1)
    rho_emp_log = rho_e.mean(axis=1)
    pmax_emp_log = p_e.mean(axis=1)
    br_log = br_e.mean(axis=1)
    td_log = td_e.mean(axis=1)

    for i, v in enumerate(axis_vals):
        print(
            f"  {axis_name}={v:.4f}  rho_emp={rho_emp_log[i]:.3f}  "
            f"pmax_emp={pmax_emp_log[i]:.3f}  "
            f"(bar_rho={br_log[i]:.3f} + 2*delta={td_log[i]:.3f})  "
            f"feat={means[i,0]:.3f}  graph={means[i,1]:.3f}  combined={means[i,2]:.3f}"
        )
    return means, stds, rho_emp_log, pmax_emp_log


def plot_sweep(ax, x, means, stds, xlabel, title, fixed_label):
    labels = [
        "Feature only (CoGEnT, μ=∞)",
        "Graph only (spectral)",
        "Combined (best μ)",
    ]
    colors = ["tab:blue", "tab:green", "tab:red"]
    markers = ["o", "s", "D"]
    for k in range(3):
        ax.errorbar(
            x,
            means[:, k],
            yerr=stds[:, k],
            label=labels[k],
            color=colors[k],
            marker=markers[k],
            capsize=3,
            lw=1.6,
            markersize=5,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cluster accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"{title}\n({fixed_label})")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 64)
    print("CoGEnT  --  p / rho line sweep  (graph-only vs feature-only vs combined)")
    print("=" * 64)
    print(f"  K={K}, d={D}, n_per_cluster={N_PER_CLUSTER}, n={K*N_PER_CLUSTER}")
    print(f"  gamma={GAMMA}  ->  gamma/K={GAMMA/K:.3f}")
    print(f"  N_TRIALS={N_TRIALS},  MU_GRID={MU_GRID}")
    print(f"  Parallel workers: {N_WORKERS}")
    print(f"  P_SWEEP   ({len(P_SWEEP)} pts) at rho={RHO_FIXED}")
    print(f"  RHO_SWEEP ({len(RHO_SWEEP)} pts) at p ={P_FIXED}")

    means_p, stds_p, rho_e_p, pmax_e_p = sweep("p", P_SWEEP, RHO_FIXED)
    means_r, stds_r, rho_e_r, pmax_e_r = sweep("rho", RHO_SWEEP, P_FIXED)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    plot_sweep(
        axes[0],
        P_SWEEP,
        means_p,
        stds_p,
        xlabel=r"$p_{\max}$  (target misconnection rate, exact)",
        title=r"Accuracy vs $p_{\max}$",
        fixed_label=rf"$\rho={RHO_FIXED}$, K={K}, n={K*N_PER_CLUSTER}, trials={N_TRIALS}",
    )
    plot_sweep(
        axes[1],
        RHO_SWEEP,
        means_r,
        stds_r,
        xlabel=r"$\rho$  (cluster diameter, exact)",
        title=r"Accuracy vs $\rho$",
        fixed_label=rf"$p_{{\max}}={P_FIXED}$, K={K}, n={K*N_PER_CLUSTER}, trials={N_TRIALS}",
    )
    fig.tight_layout()

    out = os.path.join(_HERE, "sweep_p_rho.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved figure: {out}")

    # ------------------------------------------------------------------
    # CSV export for pgfplots / TikZ.
    # Space-separated, one header row; pgfplots reads via:
    #   \pgfplotstableread[col sep=space, header=true]{sweep_p.dat}\tbl
    #   \addplot table[x=p, y=feat] {\tbl};
    # ------------------------------------------------------------------
    import csv

    def _write_csv(path, header, rows):
        with open(path, "w", newline="") as f:
            w = csv.writer(f, delimiter=" ")
            w.writerow(header)
            for r in rows:
                w.writerow([f"{v:.6g}" for v in r])
        print(f"Saved CSV:    {path}")

    p_header = [
        "p_target",
        "pmax_emp",
        "rho_emp",
        "feat_mean",
        "graph_mean",
        "combined_mean",
        "feat_std",
        "graph_std",
        "combined_std",
    ]
    p_rows = [
        [
            P_SWEEP[i],
            pmax_e_p[i],
            rho_e_p[i],
            means_p[i, 0],
            means_p[i, 1],
            means_p[i, 2],
            stds_p[i, 0],
            stds_p[i, 1],
            stds_p[i, 2],
        ]
        for i in range(len(P_SWEEP))
    ]
    _write_csv(os.path.join(_HERE, "sweep_p.csv"), p_header, p_rows)

    r_header = [
        "rho_target",
        "rho_emp",
        "pmax_emp",
        "feat_mean",
        "graph_mean",
        "combined_mean",
        "feat_std",
        "graph_std",
        "combined_std",
    ]
    r_rows = [
        [
            RHO_SWEEP[i],
            rho_e_r[i],
            pmax_e_r[i],
            means_r[i, 0],
            means_r[i, 1],
            means_r[i, 2],
            stds_r[i, 0],
            stds_r[i, 1],
            stds_r[i, 2],
        ]
        for i in range(len(RHO_SWEEP))
    ]
    _write_csv(os.path.join(_HERE, "sweep_rho.csv"), r_header, r_rows)
