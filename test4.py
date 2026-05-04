import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Solver (projected subgradient)
# ----------------------------


def project_to_ball(v, ell):
    nrm = np.linalg.norm(v)
    if nrm <= ell:
        return v
    return v * (ell / (nrm + 1e-12))


def solve_thetas_for_gamma(
    xbars,
    n,
    ell,
    gamma,
    theta_init=None,
    max_iter=4000,
    lr0=0.3,
    eps=1e-6,
    tol=1e-8,
    verbose=False,
):
    """
    Solve:
      sum_i (n_i/2)||theta_i - xbar_i||^2 + gamma * sum_{i<j}(n_i+n_j)||theta_i-theta_j||
    s.t. ||theta_i||<=ell
    """
    K, d = xbars.shape

    # warm start
    if theta_init is None:
        theta = np.vstack([project_to_ball(xbars[i].copy(), ell) for i in range(K)])
    else:
        theta = theta_init.copy()

    pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]

    def objective(th):
        quad = 0.0
        for i in range(K):
            quad += 0.5 * n[i] * np.sum((th[i] - xbars[i]) ** 2)
        fuse = 0.0
        for i, j in pairs:
            fuse += (n[i] + n[j]) * np.linalg.norm(th[i] - th[j])
        return quad + gamma * fuse

    prev_obj = objective(theta)

    for t in range(1, max_iter + 1):
        lr = lr0 / np.sqrt(t)  # diminishing step

        grad = np.zeros_like(theta)

        # quadratic gradient
        for i in range(K):
            grad[i] += n[i] * (theta[i] - xbars[i])

        # fusion subgradient
        for i, j in pairs:
            diff = theta[i] - theta[j]
            dist = np.linalg.norm(diff)
            if dist < eps:
                continue
            u = diff / dist
            w = n[i] + n[j]
            grad[i] += gamma * w * u
            grad[j] -= gamma * w * u

        # step + projection
        theta_new = theta - lr * grad
        for i in range(K):
            theta_new[i] = project_to_ball(theta_new[i], ell)

        obj = objective(theta_new)

        if abs(prev_obj - obj) <= tol * (1 + abs(prev_obj)):
            theta = theta_new
            prev_obj = obj
            break

        theta, prev_obj = theta_new, obj

        if verbose and (t % 500 == 0):
            print(f"iter {t}  obj={obj:.6f}")

    return theta, prev_obj


def min_pairwise_distance(theta):
    K = theta.shape[0]
    beta = np.inf
    for i in range(K):
        for j in range(i + 1, K):
            beta = min(beta, np.linalg.norm(theta[i] - theta[j]))
    return float(beta)


# ----------------------------
# Data generation (2D symmetric means)
# ----------------------------


def generate_circle_means(K, radius):
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
    mu = np.c_[radius * np.cos(angles), radius * np.sin(angles)]
    return mu


def sample_clusters(mu, sigma, n_per_cluster, seed=0):
    rng = np.random.default_rng(seed)
    K, d = mu.shape
    X = []
    labels = []
    for i in range(K):
        Xi = rng.normal(loc=mu[i], scale=sigma, size=(n_per_cluster, d))
        X.append(Xi)
        labels.append(np.full(n_per_cluster, i, dtype=int))
    return np.vstack(X), np.concatenate(labels)


def cluster_means_from_samples(X, labels, K):
    xbars = np.zeros((K, X.shape[1]))
    n = np.zeros(K, dtype=int)
    for i in range(K):
        Xi = X[labels == i]
        n[i] = Xi.shape[0]
        xbars[i] = Xi.mean(axis=0)
    return xbars, n


# ----------------------------
# Run + visualize for many gamma
# ----------------------------


def run_demo(
    K=5,
    n_per_cluster=500,
    sigma=0.5,
    m=6.0,  # ||mu_i|| (true mean radius)
    ell=2.0,  # feasible radius
    gammas=None,
    seed=1,
    max_iter=5000,
    lr0=0.4,
):
    if gammas is None:
        gammas = np.linspace(0.0, 2, 50)

    mu = generate_circle_means(K, m)
    X, labels = sample_clusters(mu, sigma, n_per_cluster, seed=seed)
    xbars, n = cluster_means_from_samples(X, labels, K)

    theta = None  # warm start
    thetas, betas, radii, objs = [], [], [], []

    for g in gammas:
        theta, obj = solve_thetas_for_gamma(
            xbars, n, ell, g, theta_init=theta, max_iter=max_iter, lr0=lr0
        )
        thetas.append(theta.copy())
        betas.append(min_pairwise_distance(theta))
        radii.append(np.mean(np.linalg.norm(theta, axis=1)))
        objs.append(obj)

    thetas = np.array(thetas)

    # ---- Plot 1: show centers for a few gammas
    pick = (
        [0, len(gammas) // 2, len(gammas) - 1]
        if len(gammas) >= 3
        else list(range(len(gammas)))
    )
    plt.figure(figsize=(10, 4))

    for p, idx in enumerate(pick, 1):
        ax = plt.subplot(1, len(pick), p)
        ax.scatter(X[:, 0], X[:, 1], s=6, alpha=0.15)
        ax.scatter(
            mu[:, 0], mu[:, 1], marker="x", s=80, linewidths=2, label="true means μ"
        )
        ax.scatter(xbars[:, 0], xbars[:, 1], marker="^", s=60, label="sample means x̄")
        ax.scatter(thetas[idx, :, 0], thetas[idx, :, 1], s=70, label="optimized θ*")

        circle = plt.Circle((0, 0), ell, fill=False)
        ax.add_artist(circle)

        ax.set_aspect("equal", "box")
        ax.set_title(f"γ = {gammas[idx]:.3g}\nβ = {betas[idx]:.3g}")
        ax.set_xlim(-m * 1.2, m * 1.2)
        ax.set_ylim(-m * 1.2, m * 1.2)
        if p == 1:
            ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    # plt.show()

    # ---- Plot 2: trajectories of each θ_i as γ increases
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.scatter(mu[:, 0], mu[:, 1], marker="x", s=80, linewidths=2, label="μ")
    circle = plt.Circle((0, 0), ell, fill=False)
    ax.add_artist(circle)

    for i in range(K):
        ax.plot(thetas[:, i, 0], thetas[:, i, 1], marker="o", linewidth=1, markersize=3)

    ax.set_aspect("equal", "box")
    ax.set_title("Centroid paths as γ increases")
    ax.set_xlim(-m * 1.2, m * 1.2)
    ax.set_ylim(-m * 1.2, m * 1.2)
    plt.tight_layout()
    # plt.show()

    # ---- Plot 3: β(γ), mean radius(γ), objective(γ)
    plt.figure(figsize=(10, 3))

    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(gammas, betas, marker="o")
    ax1.set_title("β vs γ")
    ax1.set_xlabel("γ")
    ax1.set_ylabel("β")

    ax2 = plt.subplot(1, 3, 2)
    ax2.plot(gammas, radii, marker="o")
    ax2.set_title("mean ||θ*|| vs γ")
    ax2.set_xlabel("γ")
    ax2.set_ylabel("mean radius")

    ax3 = plt.subplot(1, 3, 3)
    ax3.plot(gammas, objs, marker="o")
    ax3.set_title("objective vs γ")
    ax3.set_xlabel("γ")
    ax3.set_ylabel("obj")

    plt.tight_layout()
    plt.show()

    return gammas, thetas, np.array(betas), np.array(radii), np.array(objs)


# ---- Run it
gammas, thetas, betas, radii, objs = run_demo()
