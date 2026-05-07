"""
src/cogent_empirical_validation.py

Empirical reproduction of the synergistic recovery region of Theorem 5.1
(Fig. 3 in the paper).  We follow the paper's setup as closely as possible:

  * K = 5 balanced clusters, squared loss  f_v = ½||x_v - θ||²  (m_v = 1).
  * d_min ≈ 2.5  (centroids on a ring; same as paper Fig. 3).
  * γ  = 2.0,  so γ/K = 0.4   (node-only ceiling, blue line in Fig. 3).
  * p* = ((K+4) - √((K+4)² - 16)) / 8  ≈ 0.117  (graph-only threshold).

Graph construction (eq. (6) of the paper):
  Instead of sampling an SBM and measuring p_max afterwards, we *construct*
  a regular adjacency matrix that hits a target p_max exactly:
    - intra-class:   each v in C_i has constant degree d_intra = N(1 - p)
                     using a circulant graph on N nodes
    - inter-class:   each v in C_i has constant degree d_inter = N p
                     to every other class C_j (shifted bipartite graph)
  By symmetry δ_ij = 0, so p_max = bar_rho_ij = p_target exactly.
  This avoids the need for large n to keep the SBM fluctuation small.

Per-cell empirical procedure:
  for each (p_target, sigma, trial):
    1. Sample data; compute rho (cluster diameter, Remark 4.5).
    2. Construct deterministic regular graph with p_max = p_target.
    3. Run CoGEnT node-only         (mu = inf).
    4. Run CoGEnT combined for each mu in MU_GRID, take BEST accuracy.
"""

import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from cogent import cogent_structured_atoms_with_W
from cluster_generation import random_sparse_directions

# ============================================================
# Experiment parameters  (paper Fig. 3 setup)
# ============================================================
K = 3
D = 2
# Small graph: we construct adjacency directly so delta_ij = 0 and the
# realized p_max equals the requested target exactly (no SBM fluctuation).
N_PER_CLUSTER = 15  # n = 150  (granularity 1/N ≈ 0.033)
ELL = 1000.0
# Centroid geometry: ring of radius R_RING -> d_min = 2 R_RING sin(pi/K).
# Larger d_min raises the Thm 4.7 upper bound  gamma < m_v * d_min / (2(1-1/K))
# so we can pick a larger gamma -> larger gamma/K -> wider node-only region.
# R_RING=5.0, K=5 -> d_min ≈ 5.88, upper bound on gamma ≈ 3.67.
R_RING = 4.0
RADII = [R_RING] * K
GAMMA = 0.5  # well within Thm 4.7 upper bound -> gamma/K = 0.6
M_V = 1.0

# Per-cell mu sweep -- one panel per mu, plus best/union summary panels.
MU_GRID = [0, 0.2, 0.5]

N_P = 10
N_RHO = 10
N_TRIALS = 5

P_STAR = ((K + 4) - np.sqrt((K + 4) ** 2 - 16)) / 8.0
GAMMA_K = GAMMA / K  # = 0.6 for K=5, gamma=3

# x-axis: target p_max for the constructed graph (covers all regimes).
#   p_max < p* ≈ 0.117  : graph-only recovers
#   0.117 < p_max < 0.25 : combined may recover (depending on rho)
#   p_max > 0.25         : even combined fails (eq. (19))
# Snap p targets to multiples of 1/N so realized p_max equals target exactly
# (granularity is 1/N_PER_CLUSTER).
_p_grid = np.linspace(0.005, 0.25, N_P)
P_VALS = np.round(_p_grid * N_PER_CLUSTER) / N_PER_CLUSTER
P_VALS = np.unique(P_VALS[P_VALS > 0])
N_P = len(P_VALS)  # update N_P to match snapped grid
# y-axis: rho is the cluster diameter (Remark 4.5).  We control it exactly
# by placing all points of a cluster on a sphere of radius rho/2 around the
# centroid (with one antipodal pair to make the diameter exact).
# Range spans from below gamma/K = 0.6 up to ~d_min so all four regimes appear.
d_min_ring = 3.0 * R_RING * np.sin(np.pi / K)
RHO_VALS = np.linspace(0.10, 0.5 * d_min_ring, N_RHO)

# Per-block split between block-average rho_bar and per-node deviation delta:
#     rho_bar_ij = ALPHA      * p_target
#     2 delta_ij = (1 - ALPHA) * p_target
# Set ALPHA = 1 to recover a fully regular graph (delta = 0).
# Set ALPHA = 0 to put all of p_max into per-node deviation.
ALPHA = 0.6

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

N_WORKERS = max(1, (mp.cpu_count() or 2) - 1)


# ============================================================
# Helpers
# ============================================================


def cluster_accuracy(pred, true):
    C = np.zeros((K, K), int)
    for p, t in zip(pred, true):
        C[int(p) % K, int(t) % K] += 1
    row, col = linear_sum_assignment(-C)
    return C[row, col].sum() / len(true)


def compute_rho(X, labels):
    """rho = max_i max_{u,v in C_i} ||x_u - x_v||  (Remark 4.5)."""
    rho = 0.0
    for i in range(K):
        Xi = X[labels == i]
        if len(Xi) >= 2:
            diff = Xi[:, None, :] - Xi[None, :, :]
            dists = np.linalg.norm(diff, axis=-1)
            rho = max(rho, float(dists.max()))
    return rho


def empirical_pmax(A, labels):
    """
    p_max of a realized graph A according to paper eq. (6):

        n_bar_{v,j}  = (n_i - n_{v,i})  if j = i,    else  n_{v,j}
        N_bar_{i,j}  = sum_{v in C_i} n_bar_{v,j}
        bar_rho_{ij} = N_bar_{i,j} / (n_i n_j)
        delta_{ij}   = max_{v in C_i} | bar_rho_{ij} - n_bar_{v,j} / n_j |
        p_{ij}       = bar_rho_{ij} + 2 * delta_{ij}
        p_max        = max_{i,j} p_{ij}
    """
    classes = np.unique(labels)
    Kk = len(classes)
    n_v_j = np.zeros((len(labels), Kk))  # n_{v, C_j}
    for jj, cj in enumerate(classes):
        mask = labels == cj
        n_v_j[:, jj] = A[:, mask].sum(axis=1)

    pmax = 0.0
    for ii, ci in enumerate(classes):
        v_in_ci = np.where(labels == ci)[0]
        n_i = len(v_in_ci)
        for jj, cj in enumerate(classes):
            n_j = (labels == cj).sum()
            if ii == jj:
                nbar_v = n_i - n_v_j[v_in_ci, jj]  # missing intra
            else:
                nbar_v = n_v_j[v_in_ci, jj]  # spurious inter
            Nbar_ij = nbar_v.sum()
            bar_rho_ij = Nbar_ij / (n_i * n_j)
            delta_ij = float(np.max(np.abs(bar_rho_ij - nbar_v / n_j)))
            p_ij = bar_rho_ij + 2.0 * delta_ij
            pmax = max(pmax, float(p_ij))
    return pmax


def make_clusters_fixed_rho(rho, centroids, n_per_cluster, rng):
    """
    Place n_per_cluster points around each centroid so that the cluster
    diameter (max pairwise distance, Remark 4.5) equals `rho` exactly.

    Construction:  put n-2 random points uniformly on the sphere of radius
    rho/2 around the centroid, plus one antipodal pair on that sphere
    (so the diameter is rho and is attained).
    """
    K_local, d = centroids.shape
    X_list = []
    labels = []
    r = 0.5 * rho
    for k in range(K_local):
        # one antipodal pair guarantees the diameter is exactly rho
        u = rng.standard_normal(d)
        u /= np.linalg.norm(u) + 1e-12
        pts = [centroids[k] + r * u, centroids[k] - r * u]
        # remaining points: uniform on sphere of radius r
        m = n_per_cluster - 2
        if m > 0:
            G = rng.standard_normal((m, d))
            G /= np.linalg.norm(G, axis=1, keepdims=True) + 1e-12
            pts.append(centroids[k] + r * G)
        Xi = np.vstack(
            [pts[0][None, :], pts[1][None, :]] + ([pts[2]] if len(pts) == 3 else [])
        )
        X_list.append(Xi)
        labels.extend([k] * n_per_cluster)
    X = np.vstack(X_list)
    labels = np.array(labels, dtype=int)
    return X, labels


def _add_circulant(A, nodes, deg):
    """Add a circulant of degree `deg` among `nodes` (in-place).

    Uses offsets +/-1, +/-2, ..., for the floor(deg/2) closest pairs.
    If `deg` is odd, requires len(nodes) to be even and additionally adds
    the antipodal offset N/2 (a perfect matching).  Net degree = `deg`
    on every node.
    """
    n = len(nodes)
    if deg <= 0 or n < 2:
        return
    half = deg // 2
    for off in range(1, half + 1):
        for k in range(n):
            u, v = nodes[k], nodes[(k + off) % n]
            A[u, v] = A[v, u] = 1.0
    if deg % 2 == 1:
        # antipodal matching -- only valid for even n
        if n % 2 != 0:
            raise ValueError(
                f"Cannot build a {deg}-regular circulant on {n} nodes " f"(odd n*deg)."
            )
        m = n // 2
        for k in range(m):
            u, v = nodes[k], nodes[k + m]
            A[u, v] = A[v, u] = 1.0


def _add_bipartite(A, rows, cols, row_deg):
    """Add a regular bipartite graph: each row-node has `row_deg` edges to cols
    and (when |rows|==|cols|) each col-node also has `row_deg`."""
    Nr, Nc = len(rows), len(cols)
    if row_deg <= 0 or Nr == 0 or Nc == 0:
        return
    for s in range(row_deg):
        for k in range(Nr):
            u, v = rows[k], cols[(k + s) % Nc]
            A[u, v] = A[v, u] = 1.0


def construct_controlled_graph(labels, p_target, alpha=1.0):
    """
    Build a symmetric adjacency `A` so that for every ordered pair of
    classes (i, j), p_{ij} = bar_rho_{ij} + 2 delta_{ij} = p_target,
    up to a granularity of 1/N (with N = nodes per class).

    Construction (delta = 0, fully regular).
      - Inter (i != j): for every v in C_i, exactly m_inter = round(p*N)
        edges to C_j (and vice-versa).  Realized via a shifted regular
        bipartite graph -> n_{v,j} = m_inter for all v -> delta_{ij} = 0.
      - Intra (i == j): n_bar_{v,i} = N - n_{v,i} (paper eq. 6).  We make
        the intra graph (intra_deg)-regular with intra_deg = N - 1 -
        m_intra_missing, where m_intra_missing = max(0, round(p*N) - 1).
        The "-1" accounts for the unavoidable self-exclusion floor 1/N.
        bar_rho_{ii} = (N - intra_deg) / N = (1 + m_intra_missing) / N.
        delta_{ii} = 0 by regularity, so p_{ii} = bar_rho_{ii}.

    Feasibility: a (intra_deg)-regular circulant on N nodes exists when
    N * intra_deg is even.  When N is odd we round intra_deg down to the
    nearest even integer (granularity becomes 2/N for the diagonal, but
    the inter blocks remain at 1/N granularity).

    Irreducible floor: p_max >= 1/N (set by the diagonal self-exclusion).

    The `alpha` argument is accepted for API compatibility but ignored;
    the construction always uses delta = 0.
    """
    classes = np.unique(labels)
    n = len(labels)
    A = np.zeros((n, n), dtype=float)
    idx = {c: np.where(labels == c)[0] for c in classes}
    N = len(idx[classes[0]])

    # ---- inter-class: m_inter = round(p*N), in [0, N] -------------------
    m_inter = int(round(p_target * N))
    m_inter = max(0, min(m_inter, N))

    # ---- intra-class: missing-edges per node = max(0, round(p*N) - 1) ---
    m_intra_missing = max(0, int(round(p_target * N)) - 1)
    m_intra_missing = min(m_intra_missing, N - 1)
    intra_deg = N - 1 - m_intra_missing
    # Adjust for parity (need N*intra_deg even).
    if (N * intra_deg) % 2 != 0:
        intra_deg -= 1  # drop one edge per node (ceil missing to even)
    intra_deg = max(0, intra_deg)

    for c in classes:
        _add_circulant(A, idx[c], intra_deg)

    for ii, ci in enumerate(classes):
        for jj, cj in enumerate(classes):
            if jj <= ii:
                continue
            _add_bipartite(A, idx[ci], idx[cj], m_inter)

    np.fill_diagonal(A, 0.0)
    return A


def run_one(args):
    """Worker: returns metrics for a single (p_target, rho_target, trial) cell."""
    si, pi, trial, p_target, rho_target, seed = args
    rng = np.random.default_rng(seed)
    np_seed = int(rng.integers(0, 2**31))

    # ------- centroids on a ring (sparse direction vectors) -----------------
    dirs = random_sparse_directions(K, D, num_active=2, seed=np_seed)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    centroids = (dirs / norms) * np.asarray(RADII, dtype=float)[:, None]

    # ------- data with EXACT cluster diameter = rho_target ------------------
    X, labels = make_clusters_fixed_rho(rho_target, centroids, N_PER_CLUSTER, rng)
    rho_emp = compute_rho(X, labels)  # ≈ rho_target by construction

    d_min = min(
        np.linalg.norm(centroids[i] - centroids[j])
        for i in range(K)
        for j in range(K)
        if i != j
    )
    R_star = M_V * d_min**2 / 8.0

    # ------- controlled graph: rho_bar_ij + 2 delta_ij = p_target ----------
    A_graph = construct_controlled_graph(labels, p_target, ALPHA)
    pmax_emp = empirical_pmax(A_graph, labels)

    res_n = cogent_structured_atoms_with_W(
        X=X, gamma=GAMMA, mu=float("inf"), seed=np_seed, **SOLVER_KW
    )
    acc_node = cluster_accuracy(res_n["W"].argmax(0), labels)

    accs_per_mu = []
    for mu in MU_GRID:
        res_c = cogent_structured_atoms_with_W(
            X=X, gamma=GAMMA, mu=mu, A_graph=A_graph, seed=np_seed, **SOLVER_KW
        )
        accs_per_mu.append(cluster_accuracy(res_c["W"].argmax(0), labels))

    return (si, pi, trial, acc_node, accs_per_mu, rho_emp, R_star, pmax_emp)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    n_calls = N_P * N_RHO * N_TRIALS * (1 + len(MU_GRID))
    n_tasks = N_P * N_RHO * N_TRIALS

    print("=" * 64)
    print("CoGEnT  --  empirical reproduction of paper Fig. 3 / Thm 5.1")
    print("=" * 64)
    print(f"  K={K}, d={D}, n_per_cluster={N_PER_CLUSTER}, n={K*N_PER_CLUSTER}")
    print(f"  gamma={GAMMA}  ->  gamma/K={GAMMA_K:.3f}")
    print(f"  p* = {P_STAR:.4f}  (Cor. 4.2)")
    print(f"  Combined: best accuracy over mu in {MU_GRID}")
    print(
        f"  Grid: {N_P} x {N_RHO},  {N_TRIALS} trials  "
        f"({n_tasks} tasks, {n_calls} CoGEnT calls)"
    )
    print(f"  Parallel workers: {N_WORKERS}")
    print()

    node_sum = np.zeros((N_RHO, N_P))
    per_mu_sum = [np.zeros((N_RHO, N_P)) for _ in MU_GRID]
    rho_sum = np.zeros((N_RHO, N_P))
    rs_sum = np.zeros((N_RHO, N_P))
    pmax_sum = np.zeros((N_RHO, N_P))

    tasks = []
    for si, rho_t in enumerate(RHO_VALS):
        for pi, p in enumerate(P_VALS):
            for trial in range(N_TRIALS):
                seed = si * N_P * N_TRIALS + pi * N_TRIALS + trial
                tasks.append((si, pi, trial, float(p), float(rho_t), seed))

    done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for fut in as_completed([ex.submit(run_one, t) for t in tasks]):
            si, pi, trial, acc_n, accs_mu, rho_e, rs, pmx = fut.result()
            node_sum[si, pi] += acc_n
            for mi, a in enumerate(accs_mu):
                per_mu_sum[mi][si, pi] += a
            rho_sum[si, pi] += rho_e
            rs_sum[si, pi] += rs
            pmax_sum[si, pi] += pmx
            done += 1
            if done % max(1, n_tasks // 20) == 0 or done == n_tasks:
                print(f"  [{done:4d}/{n_tasks}]  " f"({100.0*done/n_tasks:5.1f}%)")

    acc_node_grid = node_sum / N_TRIALS
    per_mu_grids = [s / N_TRIALS for s in per_mu_sum]
    acc_best_grid = np.maximum.reduce(per_mu_grids)  # best mu per cell
    rho_grid = rho_sum / N_TRIALS
    Rstar_grid = rs_sum / N_TRIALS
    pmax_grid = pmax_sum / N_TRIALS  # realized empirical p_max

    rho_axis = rho_grid.mean(axis=1)  # row-mean rho       (y axis)
    pmax_axis = pmax_grid.mean(axis=0)  # column-mean p_max  (x axis)

    print("\n  Realized p_max per target column (target | mean realized):")
    for ptgt, pmx in zip(P_VALS, pmax_axis):
        print(f"    p_target={ptgt:.3f}  ->  p_max_realized={pmx:.3f}")
    print("\n  Realized rho per row (target | mean realized):")
    for rt, r in zip(RHO_VALS, rho_axis):
        flag = " <-- gamma/K" if abs(r - GAMMA_K) < 0.10 else ""
        print(f"    rho_target={rt:.3f}  rho_realized={r:.3f}{flag}")

    # ============================================================
    # Theory map at the *realized* (p_max, rho) grid
    # ============================================================
    PP, RR = np.meshgrid(pmax_axis, rho_axis)
    standalone = (RR <= GAMMA_K) | (PP <= P_STAR)

    MU_TH = np.logspace(-2, 3, 80)
    MU3 = MU_TH[:, None, None]
    PP3 = PP[None, :, :]
    RR3 = RR[None, :, :]
    RM3 = np.maximum(0.0, RR3 - GAMMA_K)  # r_max

    # ζ(r) = r²/(2 m_v) for m_v-strongly convex squared loss
    Theta3 = K * MU3 * RM3**2 / (2.0 * M_V)
    cond_p = (PP3 < (1.0 - Theta3) / 4.0) & (PP3 < 0.25) & (Theta3 < 1.0)

    slack3 = (1.0 - 4.0 * PP3) - Theta3
    rm_pos = cond_p & (slack3 > 0) & (RM3 > 0)
    _omp = np.where(PP3 < 1, 1.0 - PP3, 1.0)
    _slk = np.where(slack3 > 0, slack3, 1.0)
    bracket = (
        Theta3
        - (1.0 - 4.0 * PP3)
        + (K - 2) * PP3 / _omp
        + (K - 1) * PP3**2 / (_omp**2 * _slk)
    )
    Rmin3 = np.where(rm_pos, bracket / (K * MU3), 0.0)
    RS3 = Rstar_grid[None, :, :]
    thm51_per_mu = (cond_p & (RM3 == 0)) | (rm_pos & (RS3 >= Rmin3))
    thm51 = thm51_per_mu.any(axis=0)  # union over mu
    gain_th = thm51 & ~standalone

    # ============================================================
    # Synergistic-corner marker (in (p_max, rho) units)
    # ============================================================
    pstar_x = float(np.interp(P_STAR, pmax_axis, np.arange(N_P)))
    if rho_axis[0] < GAMMA_K < rho_axis[-1]:
        gammak_y = float(np.interp(GAMMA_K, rho_axis, np.arange(N_RHO)))
    else:
        gammak_y = -1

    xlabels = [f"{p:.3f}" for p in pmax_axis]
    ylabels = [f"{r:.2f}" for r in rho_axis]

    def _fmt(ax):
        ax.set_xticks(range(N_P))
        ax.set_xticklabels(xlabels, rotation=40, ha="right", fontsize=7)
        ax.set_yticks(range(N_RHO))
        ax.set_yticklabels(ylabels, fontsize=7)
        ax.set_xlabel("$p_{\\max}$ (constructed, eq. 6)", fontsize=8)
        ax.set_ylabel("$\\rho$ (cluster diameter)", fontsize=8)
        ax.axvline(pstar_x, color="red", lw=1.3, ls="--", label=f"$p^*$={P_STAR:.3f}")
        if gammak_y >= 0:
            ax.axhline(
                gammak_y,
                color="blue",
                lw=1.3,
                ls="--",
                label=f"$\\gamma/K$={GAMMA_K:.2f}",
            )
            ax.scatter(
                [pstar_x],
                [gammak_y],
                marker="*",
                s=240,
                c="black",
                edgecolors="white",
                linewidths=1.5,
                zorder=5,
                label="Synergistic corner $(p^*,\\gamma/K)$",
            )

    # ============================================================
    # Figure: 2 rows x (len(MU_GRID)) cols
    #   Row 0: combined accuracy heatmap for each mu
    #   Row 1: node-only | best (max over mu) | union (any mu recovers) | theory
    # ============================================================
    NCOLS = max(4, len(MU_GRID))
    fig, axes = plt.subplots(2, NCOLS, figsize=(3.6 * NCOLS, 9.0))

    # ---- Row 0: per-mu combined accuracy ------------------------------------
    for mi, mu in enumerate(MU_GRID):
        ax = axes[0, mi]
        im = ax.imshow(
            per_mu_grids[mi],
            origin="lower",
            aspect="auto",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
        )
        ax.set_title(f"Combined acc, $\\mu$={mu}", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # blank any unused panels in row 0
    for mi in range(len(MU_GRID), NCOLS):
        axes[0, mi].axis("off")

    # ---- Row 1: node-only | best | union | theory ---------------------------
    thresh = 0.65

    im = axes[1, 0].imshow(
        acc_node_grid,
        origin="lower",
        aspect="auto",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
    )
    axes[1, 0].set_title("Node-only accuracy ($\\mu=\\infty$)", fontsize=10)
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im = axes[1, 1].imshow(
        acc_best_grid,
        origin="lower",
        aspect="auto",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
    )
    axes[1, 1].set_title(f"Best combined  (max over $\\mu$)", fontsize=10)
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # union of recovery: cell is "recovered" if ANY mu reaches threshold
    union_bin = np.zeros((N_RHO, N_P), dtype=float)
    for g in per_mu_grids:
        union_bin = np.maximum(union_bin, (g >= thresh).astype(float))
    axes[1, 2].imshow(
        union_bin,
        origin="lower",
        aspect="auto",
        cmap="Greens",
        vmin=0,
        vmax=1,
    )
    axes[1, 2].set_title(
        f"Union: any $\\mu$ recovers (acc$\\geq${thresh:.2f})\n"
        f"recovered = {union_bin.mean():.1%} of cells",
        fontsize=9,
    )

    rgb = np.ones((N_RHO, N_P, 3))
    rgb[standalone] = [0.53, 0.81, 0.98]
    rgb[gain_th] = [0.13, 0.55, 0.13]
    axes[1, 3].imshow(rgb, origin="lower", aspect="auto")
    axes[1, 3].set_title(
        f"Theory regions (paper Fig. 3)\n"
        f"green = synergistic gain ({gain_th.mean():.1%}), "
        f"blue = standalone, white = none",
        fontsize=9,
    )

    # blank any unused panels in row 1
    for mi in range(4, NCOLS):
        axes[1, mi].axis("off")

    # Format only the panels that show heatmaps (skip blanked axes)
    fmt_axes = list(axes[0, : len(MU_GRID)]) + list(axes[1, :4])
    for ax in fmt_axes:
        _fmt(ax)
    axes[0, 0].legend(fontsize=6, loc="lower left")

    fig.suptitle(
        f"CoGEnT empirical reproduction of Theorem 5.1  |  "
        f"K={K}, n={K*N_PER_CLUSTER}, $\\gamma$={GAMMA}, "
        f"$d_{{\\min}}\\approx${2*R_RING*np.sin(np.pi/K):.2f}, "
        f"{N_TRIALS} trials/cell\n"
        f"Synergistic corner $(p^*={P_STAR:.3f},\\,\\gamma/K={GAMMA_K:.2f})$",
        fontsize=11,
    )
    fig.tight_layout()

    out_dir = os.path.join(_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cogent_empirical_validation.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved -> {out_path}")

    # ============================================================
    # Region-wise summary
    # ============================================================
    print("\n" + "=" * 64)
    print(f"Recovery rate (accuracy >= {thresh:.0%}) over realized p_max grid")
    print("=" * 64)

    PP_mask = np.broadcast_to(pmax_axis[None, :], (N_RHO, N_P))
    RR_mask = np.broadcast_to(rho_axis[:, None], (N_RHO, N_P))
    node_mask = RR_mask <= GAMMA_K
    graph_mask = PP_mask <= P_STAR
    neither = ~node_mask & ~graph_mask

    def rate(grid, mask):
        v = grid[mask]
        return (v >= thresh).mean() if v.size else float("nan")

    print(
        f"  Node region   (rho<={GAMMA_K:.2f}):        "
        f"node={rate(acc_node_grid,node_mask):.1%}  "
        f"best={rate(acc_best_grid,node_mask):.1%}"
    )
    print(
        f"  Graph region  (p_max<={P_STAR:.3f}):     "
        f"node={rate(acc_node_grid,graph_mask):.1%}  "
        f"best={rate(acc_best_grid,graph_mask):.1%}"
    )
    print(
        f"  Neither (synergistic candidate):     "
        f"node={rate(acc_node_grid,neither):.1%}  "
        f"best={rate(acc_best_grid,neither):.1%}"
    )
    if gain_th.any():
        print(
            f"  Theory-predicted gain cells:         "
            f"node={rate(acc_node_grid,gain_th):.1%}  "
            f"best={rate(acc_best_grid,gain_th):.1%}  "
            f"(theory says combined recovers)"
        )

    gain = acc_best_grid - acc_node_grid
    print(f"\n  Theory standalone:        {standalone.mean():.1%}  of grid")
    print(f"  Theory Thm 5.1 (any mu):  {thm51.mean():.1%}  of grid")
    print(f"  Theory synergistic gain:  {gain_th.mean():.1%}  of grid")
    print(f"  Empirical gain mean:      {gain.mean():.3f}")
    print(f"  Mean acc node-only:       {acc_node_grid.mean():.3f}")
    print(f"  Mean acc best-combined:   {acc_best_grid.mean():.3f}")
    for mi, mu in enumerate(MU_GRID):
        print(f"  Mean acc mu={mu:<5}:        {per_mu_grids[mi].mean():.3f}")
