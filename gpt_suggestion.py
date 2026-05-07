import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

# ============================================================
# Utility: clustering accuracy up to permutation
# ============================================================


def clustering_accuracy(y_true, y_pred, K):
    """
    Accuracy after best permutation of cluster labels.
    """
    M = np.zeros((K, K), dtype=int)
    for a, b in zip(y_true, y_pred):
        M[a, b] += 1

    row_ind, col_ind = linear_sum_assignment(-M)
    return M[row_ind, col_ind].sum() / len(y_true)


def is_perfect_recovery(y_true, y_pred, K):
    return clustering_accuracy(y_true, y_pred, K) == 1.0


# ============================================================
# Synthetic data generation
# ============================================================


def generate_data(K=3, n_per_cluster=15, sigma_x=0.6, p_in=0.75, p_out=0.15, seed=None):
    """
    Generate Gaussian node features and SBM-like graph (vectorized).
    """
    rng = np.random.default_rng(seed)
    n = K * n_per_cluster

    # Class centers on a circle
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
    radius = 2.5
    centers = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)

    y = np.repeat(np.arange(K), n_per_cluster)
    X = centers[y] + sigma_x * rng.normal(size=(n, 2))

    # Vectorized SBM: build edge probability matrix then sample upper triangle.
    same = y[:, None] == y[None, :]
    P = np.where(same, p_in, p_out)
    iu, ju = np.triu_indices(n, k=1)
    edges = (rng.random(iu.size) < P[iu, ju]).astype(float)
    A = np.zeros((n, n), dtype=float)
    A[iu, ju] = edges
    A = A + A.T

    return X, A, y


# ============================================================
# Method 1: Graph-only nuclear-norm clustering
# ============================================================


def graph_only_nuclear_norm(A, K, lam=None, solver="SCS"):
    """
    Nuclear-norm + L1 graph denoising (Chen-Jalali-Sanghavi-Xu style):

        min_{Y, S}  ||Y||_*  +  lam * ||S||_1
        s.t.        Y + S = A,  0 <= Y <= 1.

    The clean low-rank piece Y is then clustered by K-means on its top-K
    eigenvectors (row-normalized).
    """
    n = A.shape[0]
    if lam is None:
        lam = n / 4

    Y = cp.Variable((n, n), symmetric=True)
    S = cp.Variable((n, n), symmetric=True)
    constraints = [Y + S == A, Y >= 0, Y <= 1]
    prob = cp.Problem(cp.Minimize(cp.normNuc(Y) + lam * cp.norm1(S)), constraints)

    try:
        prob.solve(solver=solver, verbose=False)
    except Exception:
        prob.solve(solver="SCS", verbose=False, max_iters=3000)

    Y_val = Y.value
    if Y_val is None:
        Y_val = A  # fallback: cluster raw adjacency

    # Top-K eigenvectors (Y is symmetric PSD-ish), row-normalize, then K-means.
    Y_sym = 0.5 * (Y_val + Y_val.T)
    w, V = np.linalg.eigh(Y_sym)
    U = V[:, -K:]  # K largest eigenvalues
    norms = np.linalg.norm(U, axis=1, keepdims=True)
    U = U / np.maximum(norms, 1e-12)
    return KMeans(n_clusters=K, random_state=0, n_init=10).fit_predict(U)


# ============================================================
# Method 2: Node-only convex clustering
# ============================================================


def node_only_convex_clustering(X, K, gamma=1.0, solver="CLARABEL"):
    """
    Solve:
        min_theta 1/2 sum_v ||x_v - theta_v||^2
                  + gamma/n sum_{u<v} ||theta_u - theta_v||_2

    Then cluster learned theta values with K-means.
    """
    n, d = X.shape
    Theta = cp.Variable((n, d))

    data_fit = 0.5 * cp.sum_squares(Theta - X)

    # Vectorized fusion penalty: sum of row-wise 2-norms of pairwise diffs.
    iu, ju = np.triu_indices(n, k=1)
    diffs = Theta[iu] - Theta[ju]  # (n_pairs, d)
    fusion = cp.sum(cp.norm(diffs, 2, axis=1))

    obj = data_fit + (gamma / n) * fusion
    prob = cp.Problem(cp.Minimize(obj))

    try:
        prob.solve(solver=solver, verbose=False)
    except Exception:
        prob.solve(solver="SCS", verbose=False, max_iters=3000)

    if Theta.value is None:
        # Fallback: KMeans on raw features
        return KMeans(n_clusters=K, random_state=0, n_init=10).fit_predict(X)

    Z = Theta.value
    return KMeans(n_clusters=K, random_state=0, n_init=10).fit_predict(Z)


# ============================================================
# Method 3: Combined graph-weighted convex clustering
# ============================================================


def combined_graph_weighted_convex_clustering(
    X, A, K, gamma=1.0, graph_weight=3.0, solver="CLARABEL"
):
    """
    Practical convex surrogate:
        min_theta 1/2 sum_v ||x_v - theta_v||^2
                  + gamma/n sum_{u<v} omega_uv ||theta_u - theta_v||_2

    with
        omega_uv = 1 + graph_weight * A_uv.

    This is convex and uses both features and graph.
    """
    n, d = X.shape
    Theta = cp.Variable((n, d))

    data_fit = 0.5 * cp.sum_squares(Theta - X)

    # Vectorized weighted fusion penalty.
    iu, ju = np.triu_indices(n, k=1)
    omega = 1.0 + graph_weight * A[iu, ju]  # (n_pairs,)
    diffs = Theta[iu] - Theta[ju]  # (n_pairs, d)
    fusion = omega @ cp.norm(diffs, 2, axis=1)

    obj = data_fit + (gamma / n) * fusion
    prob = cp.Problem(cp.Minimize(obj))

    try:
        prob.solve(solver=solver, verbose=False)
    except Exception:
        prob.solve(solver="SCS", verbose=False, max_iters=3000)

    if Theta.value is None:
        # Fallback: weighted spectral-like feature clustering
        return KMeans(n_clusters=K, random_state=0, n_init=10).fit_predict(X)

    Z = Theta.value
    return KMeans(n_clusters=K, random_state=0, n_init=10).fit_predict(Z)


# ============================================================
# Run experiment
# ============================================================


def run_experiment(
    K=3,
    n_per_cluster=12,
    p_in=0.75,
    sigma_grid=None,
    pout_grid=None,
    n_trials=10,
    gamma_node=1.4,
    gamma_combined=1.4,
    graph_weight=3.0,
    seed=123,
):
    if sigma_grid is None:
        sigma_grid = np.linspace(0.25, 1.25, 8)

    if pout_grid is None:
        pout_grid = np.linspace(0.02, 0.35, 8)

    graph_rate = np.zeros((len(sigma_grid), len(pout_grid)))
    node_rate = np.zeros((len(sigma_grid), len(pout_grid)))
    combined_rate = np.zeros((len(sigma_grid), len(pout_grid)))

    rng = np.random.default_rng(seed)

    for i, sigma_x in enumerate(sigma_grid):
        for j, p_out in enumerate(pout_grid):
            graph_success = 0
            node_success = 0
            combined_success = 0

            for t in range(n_trials):
                trial_seed = rng.integers(0, 10**9)

                X, A, y = generate_data(
                    K=K,
                    n_per_cluster=n_per_cluster,
                    sigma_x=sigma_x,
                    p_in=p_in,
                    p_out=p_out,
                    seed=trial_seed,
                )

                y_graph = graph_only_nuclear_norm(A, K)
                y_node = node_only_convex_clustering(X, K, gamma=gamma_node)
                y_combined = combined_graph_weighted_convex_clustering(
                    X, A, K, gamma=gamma_combined, graph_weight=graph_weight
                )

                graph_success += is_perfect_recovery(y, y_graph, K)
                node_success += is_perfect_recovery(y, y_node, K)
                combined_success += is_perfect_recovery(y, y_combined, K)

            graph_rate[i, j] = graph_success / n_trials
            node_rate[i, j] = node_success / n_trials
            combined_rate[i, j] = combined_success / n_trials

            print(
                f"sigma={sigma_x:.2f}, p_out={p_out:.2f} | "
                f"graph={graph_rate[i,j]:.2f}, "
                f"node={node_rate[i,j]:.2f}, "
                f"combined={combined_rate[i,j]:.2f}"
            )

    return sigma_grid, pout_grid, graph_rate, node_rate, combined_rate


# ============================================================
# Plotting
# ============================================================


def plot_results(sigma_grid, pout_grid, graph_rate, node_rate, combined_rate):
    """
    Plot recovery heatmaps.
    """
    gain_rate = np.maximum(combined_rate - np.maximum(graph_rate, node_rate), 0)

    panels = [
        ("Graph-only", graph_rate),
        ("Node-only", node_rate),
        ("Combined", combined_rate),
        ("Combined-only gain", gain_rate),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4), constrained_layout=True)

    for ax, (title, data) in zip(axes, panels):
        im = ax.imshow(
            data,
            origin="lower",
            aspect="auto",
            vmin=0,
            vmax=1,
            extent=[
                pout_grid[0],
                pout_grid[-1],
                sigma_grid[0],
                sigma_grid[-1],
            ],
        )
        ax.set_title(title)
        ax.set_xlabel(r"Graph noise $p_{\rm out}$")
        ax.set_ylabel(r"Feature noise $\sigma_x$")

    fig.colorbar(im, ax=axes, shrink=0.85, label="Perfect recovery rate")
    plt.savefig("cogent_synthetic_results.png", dpi=300)
    plt.show()


# ============================================================
# CSV export (long format, pgfplots-friendly)
# ============================================================


def write_results_csv(
    sigma_grid,
    pout_grid,
    graph_rate,
    node_rate,
    combined_rate,
    path="cogent_synthetic_results.csv",
):
    """
    Long-format CSV (one row per cell). Suitable for pgfplots `matrix plot*`:

        \\pgfplotstableread[col sep=space, header=true]{file.csv}\\tbl
        \\addplot[matrix plot*, ...] table[x=p_out, y=sigma, meta=combined] {\\tbl};
    """
    import csv

    gain_rate = np.maximum(combined_rate - np.maximum(graph_rate, node_rate), 0)
    header = ["sigma", "p_out", "graph", "node", "combined", "gain"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter=" ")
        w.writerow(header)
        prev_sigma = None
        for i, s in enumerate(sigma_grid):
            # Blank line between rows is what pgfplots `matrix plot` expects.
            if prev_sigma is not None:
                f.write("\n")
            for j, p in enumerate(pout_grid):
                w.writerow(
                    [
                        f"{s:.6g}",
                        f"{p:.6g}",
                        f"{graph_rate[i, j]:.6g}",
                        f"{node_rate[i, j]:.6g}",
                        f"{combined_rate[i, j]:.6g}",
                        f"{gain_rate[i, j]:.6g}",
                    ]
                )
            prev_sigma = s
    print(f"Saved CSV: {path}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    sigma_grid = np.linspace(0.25, 1.2, 25)
    pout_grid = np.linspace(0.02, 0.36, 25)

    sigma_grid, pout_grid, graph_rate, node_rate, combined_rate = run_experiment(
        K=3,
        n_per_cluster=15,
        p_in=0.75,
        sigma_grid=sigma_grid,
        pout_grid=pout_grid,
        n_trials=25,
        gamma_node=1.4,
        gamma_combined=1.4,
        graph_weight=3.0,
        seed=123,
    )

    write_results_csv(
        sigma_grid,
        pout_grid,
        graph_rate,
        node_rate,
        combined_rate,
        path="cogent_synthetic_results.csv",
    )
    plot_results(sigma_grid, pout_grid, graph_rate, node_rate, combined_rate)
