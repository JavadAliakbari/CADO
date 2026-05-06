import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


def generate_clusters_with_directions(
    d,
    n_per_cluster,
    sigma,
    ell,
    radii,
    directions,
    normalize_data=False,
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)

    radii = np.asarray(radii, dtype=float).reshape(-1)
    K = radii.shape[0]
    if np.any(radii <= ell):
        raise ValueError("Each radius must be strictly larger than ell.")

    directions = np.asarray(directions, dtype=float)
    if directions.shape != (K, d):
        raise ValueError(f"directions must have shape (K, {d}).")

    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    dirs = directions / norms
    centroids = dirs * radii[:, None]

    X_list = []
    labels = []
    for k in range(K):
        noise = sigma * np.random.randn(n_per_cluster, d)
        X_list.append(centroids[k] + noise)
        labels.extend([k] * n_per_cluster)

    X = np.vstack(X_list)
    labels = np.array(labels, dtype=int)

    if normalize_data:
        xnorms = np.linalg.norm(X, axis=1, keepdims=True)
        xnorms[xnorms == 0] = 1.0
        X = X / xnorms

    return X, labels, centroids, dirs


def random_sparse_directions(K, d, num_active=2, seed=None):
    if seed is not None:
        np.random.seed(seed)

    if num_active > d:
        raise ValueError("num_active must be <= d")

    directions = np.zeros((K, d))
    used_patterns = set()

    for k in range(K):
        while True:
            vec = np.zeros(d)
            coords = np.random.choice(d, size=num_active, replace=False)
            values = np.random.choice([-1, 1], size=num_active)
            vec[coords] = values
            key = tuple(vec.tolist())
            if key not in used_patterns:
                used_patterns.add(key)
                directions[k] = vec
                break

    return directions


def solve_A_hard_assignments_cvxpy(
    X,
    labels,
    # ell,
    gamma,
    solver=None,
    verbose=False,
):
    """
    Solve

        min_{a_1,...,a_K}
            sum_k sum_{v in C_k} ||x_v - a_k||_2^2
            + gamma * sum_{k<l} n_k n_l ||a_k - a_l||_2
        s.t. ||a_k||_2 <= ell

    Parameters
    ----------
    X : (n, d) array
    labels : (n,) integer labels in {0, ..., K-1}
    ell : float
        Radius of l2 ball
    gamma : float
        Fusion weight
    solver : str or None
        Optional CVXPY solver, e.g. "ECOS", "SCS", "CLARABEL"
    verbose : bool

    Returns
    -------
    result : dict with keys
        A_opt            : (K, d) optimal representatives
        objective_value  : optimal objective
        fit_value        : data-fit part
        fusion_value     : fusion part without gamma
        cluster_sizes    : (K,)
        empirical_means  : (K, d)
        status           : CVXPY solve status
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels, dtype=int)

    n, d = X.shape
    K = int(labels.max()) + 1

    cluster_indices = [np.where(labels == k)[0] for k in range(K)]
    cluster_sizes = np.array([len(idx) for idx in cluster_indices], dtype=int)

    if np.any(cluster_sizes == 0):
        raise ValueError("Every cluster must contain at least one point.")

    # Variable: one representative per cluster
    A = cp.Variable((K, d))

    fit_terms = []
    for k in range(K):
        Xk = X[cluster_indices[k], :]  # (n_k, d)
        # Factor of 1/2 matches eq.(15): (1/2) sum_{i,v} W_iv ||x_v - a_i||^2
        fit_terms.append(
            0.5 * cp.sum_squares(Xk - cp.reshape(A[k, :], (1, d), order="C"))
        )

    fusion_terms = []
    for k in range(K):
        for l in range(k + 1, K):
            weight_kl = cluster_sizes[k] * cluster_sizes[l]
            fusion_terms.append(weight_kl * cp.norm(A[k, :] - A[l, :], 2))

    # constraints = [cp.norm(A[k, :], 2) <= ell for k in range(K)]

    objective = cp.Minimize(cp.sum(fit_terms) + gamma * cp.sum(fusion_terms))
    problem = cp.Problem(objective)

    if solver is None:
        problem.solve(verbose=verbose)
    else:
        problem.solve(solver=solver, verbose=verbose)

    A_opt = A.value
    if A_opt is None:
        raise RuntimeError(f"Solver failed. Status: {problem.status}")

    # Compute values again in numpy — (1/2)*fit to match eq.(15)
    fit_value = 0.0
    for k in range(K):
        Xk = X[cluster_indices[k], :]
        diff = Xk - A_opt[k][None, :]
        fit_value += 0.5 * float(np.sum(diff**2))

    fusion_value = 0.0
    for k in range(K):
        for l in range(k + 1, K):
            fusion_value += (
                cluster_sizes[k]
                * cluster_sizes[l]
                * float(np.linalg.norm(A_opt[k] - A_opt[l]))
            )

    empirical_means = np.vstack([X[idx].mean(axis=0) for idx in cluster_indices])

    return {
        "A_opt": A_opt,
        "objective_value": float(problem.value),
        "fit_value": fit_value,
        "fusion_value": fusion_value,
        "cluster_sizes": cluster_sizes,
        "empirical_means": empirical_means,
        "status": problem.status,
    }


def build_W_from_labels(labels, K=None):
    """
    Build hard one-hot assignment matrix W of shape (K, n),
    where column v is the one-hot cluster assignment of point v.
    """
    labels = np.asarray(labels, dtype=int)
    n = labels.size
    if K is None:
        K = int(labels.max()) + 1

    W = np.zeros((K, n), dtype=float)
    W[labels, np.arange(n)] = 1.0
    return W


def objective_from_WA(X, W, A, gamma):
    """
    Eq.(15) objective:
        (1/2) sum_{i,v} W_iv ||x_v - a_i||^2 + gamma * sum_{u<v} ||M_u - M_v||
    """
    W = np.asarray(W, dtype=float)
    A = np.asarray(A, dtype=float)
    D = (
        np.sum(A**2, axis=1, keepdims=True)
        - 2.0 * (A @ X.T)
        + np.sum(X**2, axis=1)[np.newaxis, :]
    )
    fit = 0.5 * float(np.sum(W * D))

    M = W.T @ A
    n = X.shape[0]
    fusion = 0.0
    for u in range(n):
        for v in range(u + 1, n):
            fusion += float(np.linalg.norm(M[u] - M[v]))

    return fit + gamma * fusion


def plot_solution_2d(X, labels, A_opt=None, centroids=None, ell=None, title=""):
    if X.shape[1] != 2:
        return

    plt.figure(figsize=(7, 7))
    for k in np.unique(labels):
        idx = labels == k
        plt.scatter(X[idx, 0], X[idx, 1], alpha=0.5, label=f"cluster {k}")

    if centroids is not None:
        plt.scatter(
            centroids[:, 0],
            centroids[:, 1],
            marker="x",
            s=200,
            linewidths=3,
            color="black",
            label="true centroids",
        )

    if A_opt is not None:
        plt.scatter(
            A_opt[:, 0],
            A_opt[:, 1],
            marker="o",
            s=180,
            edgecolors="red",
            facecolors="none",
            linewidths=2,
            label="optimized A",
        )

    if ell is not None:
        circle = plt.Circle(
            (0, 0), ell, color="red", fill=False, linestyle="--", linewidth=2
        )
        plt.gca().add_patch(circle)

    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)
    plt.axis("equal")
    plt.title(title)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)

    d = 2
    K_true = 3
    n_per_cluster = 500
    sigma = 0.10
    ell = 1.0
    gamma = 0.6 / n_per_cluster  # try also 0.0, 0.001, 0.01, 0.05

    radii = [1.5, 1.7, 2.0]
    directions = random_sparse_directions(K_true, d, num_active=1, seed=seed)

    X, labels, centroids, dirs = generate_clusters_with_directions(
        d=d,
        n_per_cluster=n_per_cluster,
        sigma=sigma,
        ell=ell,
        radii=radii,
        directions=directions,
        normalize_data=False,
        seed=seed,
    )

    result = solve_A_hard_assignments_cvxpy(
        X=X,
        labels=labels,
        ell=ell,
        gamma=gamma,
        solver=None,  # you can set "ECOS", "SCS", or "CLARABEL"
        verbose=False,
    )

    A_opt = result["A_opt"]
    W_true = build_W_from_labels(labels, K=K_true)
    obj_check = objective_from_WA(X, W_true, A_opt, gamma)

    print("Status:", result["status"])
    print("Cluster sizes:", result["cluster_sizes"])
    print("Optimal A:")
    print(np.array2string(A_opt, precision=4, suppress_small=True))
    print("Empirical means (gamma=0 solution before ball projection):")
    print(np.array2string(result["empirical_means"], precision=4, suppress_small=True))
    print("Fit value:", result["fit_value"])
    print("Fusion value:", result["fusion_value"])
    print("Optimal objective:", result["objective_value"])
    print("Objective recomputed from W,A:", obj_check)

    plot_solution_2d(
        X=X,
        labels=labels,
        A_opt=A_opt,
        centroids=centroids,
        ell=ell,
        title=f"Optimal A for hard assignments, gamma={gamma}",
    )
