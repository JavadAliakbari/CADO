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
N_PER_CLUSTER = 10  # n = 50 total
R_RING = 4.0  # centroid ring radius (d_min = 2 R sin(pi/K) ≈ 4.70)
GAMMA = 3.0  # gamma/K = 0.6
ALPHA = 0.6  # fully regular graph (per-node deviation = 0)
N_TRIALS = 10

# Fixed point of the OTHER axis is chosen near the boundary of the
# recovery region so neither feature-only nor graph-only succeeds alone,
# but their combination does.
P_FIXED = 0.25  # fixed for the rho-sweep   (above p* ≈ 0.117)
RHO_FIXED = 3.0  # fixed for the p-sweep     (above gamma/K = 0.6)

# Sweep grids -- snap p targets to the 1/N grid so realized p_max is exact.
# With ALPHA=1.0 only block-mean rho_bar contributes; granularity = 1/N.
P_SWEEP = np.arange(1, int(0.5 * N_PER_CLUSTER) + 1) / N_PER_CLUSTER
RHO_SWEEP = np.linspace(0.5, 4.5, 12)

# Combined regime: mu controls graph weight (1/(mu*n)).  Smaller mu = stronger
# graph term.  We try a small grid and take the BEST mu per cell.
MU_GRID = [0.5]

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

    A_graph = construct_controlled_graph(labels, p_target, ALPHA)
    pmax_emp = empirical_pmax(A_graph, labels)

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

    return acc_f, acc_g, acc_c_best, rho_emp, pmax_emp


def sweep(axis_name, axis_vals, fixed_val):
    """Sweep one axis; the other is held at fixed_val. Returns mean and std arrays."""
    n = len(axis_vals)
    means = np.zeros((n, 3))
    stds = np.zeros((n, 3))
    rho_emp_log = np.zeros(n)
    pmax_emp_log = np.zeros(n)

    print(f"\n--- Sweeping {axis_name} ({n} values, {N_TRIALS} trials each) ---")
    for i, v in enumerate(axis_vals):
        if axis_name == "p":
            p_target, rho_target = float(v), float(fixed_val)
        else:
            p_target, rho_target = float(fixed_val), float(v)
        accs = np.zeros((N_TRIALS, 3))
        rho_e = np.zeros(N_TRIALS)
        p_e = np.zeros(N_TRIALS)
        for t in range(N_TRIALS):
            seed = 1000 * (i + 1) + t
            af, ag, ac, re, pe = run_cell(p_target, rho_target, seed)
            accs[t] = [af, ag, ac]
            rho_e[t] = re
            p_e[t] = pe
        means[i] = accs.mean(0)
        stds[i] = accs.std(0)
        rho_emp_log[i] = rho_e.mean()
        pmax_emp_log[i] = p_e.mean()
        print(
            f"  {axis_name}={v:.4f}  rho_emp={rho_emp_log[i]:.3f}  "
            f"pmax_emp={pmax_emp_log[i]:.3f}  "
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
