import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Data generation
# ============================================================


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


def plot_data(X, labels, centroids, ell):
    plt.figure(figsize=(6, 6))
    for k in np.unique(labels):
        idx = labels == k
        plt.scatter(X[idx, 0], X[idx, 1], alpha=0.6)

    if centroids is not None:
        C = np.asarray(centroids, dtype=float)
        plt.scatter(C[:, 0], C[:, 1], marker="x", s=200, color="black")

    if ell is not None:
        circle = plt.Circle(
            (0, 0), ell, color="red", fill=False, linewidth=2, linestyle="--"
        )
        plt.gca().add_patch(circle)

    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)
    plt.axis("equal")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Data with true centroids and ell-ball")


def plot_solution(
    X, labels, A_opt=None, centroids=None, ell=None, res=None, gamma_val=None
):
    plt.figure(figsize=(7, 6))
    for k in np.unique(labels):
        idx = labels == k
        plt.scatter(X[idx, 0], X[idx, 1], alpha=0.4)

    if centroids is not None:
        C = np.asarray(centroids, dtype=float)
        plt.scatter(
            C[:, 0],
            C[:, 1],
            marker="x",
            s=200,
            color="black",
            label="True centroids",
        )

    if A_opt is not None:
        plt.scatter(
            A_opt[:, 0],
            A_opt[:, 1],
            marker="o",
            s=180,
            edgecolors="green",
            facecolors="none",
            linewidths=2,
            label="optimized A",
        )

    if ell is not None:
        circle = plt.Circle(
            (0, 0), ell, color="red", fill=False, linewidth=2, linestyle="--"
        )
        plt.gca().add_patch(circle)

    A = res["A_atoms"]
    if A.size:
        plt.scatter(
            A[:, 0], A[:, 1], marker="x", s=160, color="red", label="Learned atoms"
        )

    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)
    plt.axis("equal")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f"Learned atoms (gamma={gamma_val})")
    plt.legend()
