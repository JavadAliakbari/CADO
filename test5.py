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
    max_iter=6000,
    lr0=0.35,
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

    if theta_init is None:
        theta = np.vstack([project_to_ball(xbars[i].copy(), ell) for i in range(K)])
    else:
        theta = theta_init.copy()

    pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]

    def objective(th):
        quad = 0.0
        for i in range(K):
            quad += 0.5 * np.sum((th[i] - xbars[i]) ** 2) / n[i]
        fuse = 0.0
        for i, j in pairs:
            fuse += (n[i] + n[j]) * np.linalg.norm(th[i] - th[j])
        return quad + gamma * fuse

    prev_obj = objective(theta)

    for t in range(1, max_iter + 1):
        lr = lr0 / np.sqrt(t)

        grad = np.zeros_like(theta)

        # quadratic part
        for i in range(K):
            grad[i] += (theta[i] - xbars[i]) / n[i]

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

        theta_new = theta - lr * grad
        for i in range(K):
            theta_new[i] = project_to_ball(theta_new[i], ell)

        obj = objective(theta_new)

        if abs(prev_obj - obj) <= tol * (1 + abs(prev_obj)):
            theta = theta_new
            prev_obj = obj
            break

        theta, prev_obj = theta_new, obj

        if verbose and (t % 800 == 0):
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
# Data generation (2D symmetric means on circle)
# ----------------------------


def generate_circle_means(K, radius):
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
    mu = np.c_[radius * np.cos(angles), radius * np.sin(angles)]
    return mu, angles


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
# Theoretical curves for symmetric layout
# ----------------------------


def unit_directions_from_mu(mu):
    norms = np.linalg.norm(mu, axis=1, keepdims=True) + 1e-12
    return mu / norms


def min_unit_distance(u):
    K = u.shape[0]
    dmin = np.inf
    for i in range(K):
        for j in range(i + 1, K):
            dmin = min(dmin, np.linalg.norm(u[i] - u[j]))
    return float(dmin)


def S_double_sum_unit(u):
    # S = sum_i sum_{j!=i} ||u_i - u_j|| = 2 * sum_{i<j} ...
    K = u.shape[0]
    S = 0.0
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            S += np.linalg.norm(u[i] - u[j])
    return float(S)


def theory_radius_beta_mu(gammas, K, m, ell, u, n0):
    """
    Symmetric ansatz: theta_i = r u_i, mu_i = m u_i, equal sizes n0
    r_free = m - gamma * S/K
    r_th  = clip(r_free, 0, ell)
    beta_th = d_u_min * r_th
    mu_th (secant) = n0 * (m - r_th)/r_th  (when r_th>0)
    """
    d_u_min = min_unit_distance(u)
    S = S_double_sum_unit(u)

    r_free = m - gammas * (S / K)
    r_th = np.minimum(ell, np.maximum(0.0, r_free))

    beta_th = d_u_min * r_th

    mu_th = np.zeros_like(gammas)
    mask = r_th > 1e-12
    mu_th[mask] = n0 * (m - r_th[mask]) / r_th[mask]
    mu_th[~mask] = np.nan  # undefined when collapsed to 0

    # proof-style gamma bound (pointwise, using current beta_th and mu_th)
    n_total = K * n0
    gamma_bound_pointwise = (beta_th * mu_th) / (2.0 * n_total)

    # also show the "boundary regime" constant bound using r=ell
    # for circle evenly spaced, d_u_min = 2 sin(pi/K), this becomes sin(pi/K)(m-ell)/K
    gamma_bound_boundary = (d_u_min * ell) * (n0 * (m - ell) / ell) / (2.0 * (K * n0))

    # threshold where r_th leaves boundary (r_th=ell -> interior)
    # solve ell = m - gamma * S/K  => gamma = (m-ell)*K/S
    gamma_leave_boundary = (m - ell) * K / (S + 1e-12)

    return (
        r_th,
        beta_th,
        mu_th,
        gamma_bound_pointwise,
        gamma_bound_boundary,
        gamma_leave_boundary,
    )


# ----------------------------
# Run + visualize
# ----------------------------


def run_demo_with_theory(
    K=8,
    n_per_cluster=250,
    sigma=0.5,
    m=6.0,  # ||mu_i|| (true mean radius)
    ell=2.0,  # feasible radius
    gammas=None,
    seed=1,
    max_iter=7000,
    lr0=0.35,
):
    if gammas is None:
        gammas = np.linspace(0.0, 3.0, 16)
    gammas = np.array(gammas, dtype=float)

    mu, angles = generate_circle_means(K, m)
    X, labels = sample_clusters(mu, sigma, n_per_cluster, seed=seed)
    xbars, n = cluster_means_from_samples(X, labels, K)

    # empirical solutions (warm start across gammas)
    theta = None
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
    betas = np.array(betas)
    radii = np.array(radii)
    objs = np.array(objs)

    # theory curves (use the intended symmetric geometry)
    u = unit_directions_from_mu(mu)
    n0 = int(n_per_cluster)  # equal sizes in this demo
    (
        r_th,
        beta_th,
        mu_th,
        gamma_bound_pointwise,
        gamma_bound_boundary,
        gamma_leave_boundary,
    ) = theory_radius_beta_mu(gammas, K, m, ell, u, n0)

    L = np.max(n)  # for R bounds
    R_th_beta = 0.5 * L * (beta_th**2)  # from Step-1 style R <= (L/2) d^2
    R_uniform = 2.0 * L * (ell**2)  # crude uniform R <= 2 L ell^2

    # ---- Plot 1: show centers for a few gammas
    pick = (
        [0, len(gammas) // 2, len(gammas) - 1]
        if len(gammas) >= 3
        else list(range(len(gammas)))
    )
    plt.figure(figsize=(16, 9))

    for p, idx in enumerate(pick, 1):
        ax = plt.subplot(1, len(pick), p)
        ax.scatter(X[:, 0], X[:, 1], s=6, alpha=0.15)
        ax.scatter(
            mu[:, 0], mu[:, 1], marker="x", s=80, linewidths=2, label="true means μ"
        )
        ax.scatter(xbars[:, 0], xbars[:, 1], marker="^", s=60, label="sample means x̄")
        ax.scatter(thetas[idx, :, 0], thetas[idx, :, 1], s=70, label="optimized θ*")

        # theory markers for this gamma: points at r_th * u_i
        theta_th_points = r_th[idx] * u
        ax.scatter(
            theta_th_points[:, 0],
            theta_th_points[:, 1],
            marker="s",
            s=45,
            label="theory θ_th",
        )

        circle = plt.Circle((0, 0), ell, fill=False)
        ax.add_artist(circle)

        ax.set_aspect("equal", "box")
        ax.set_title(
            f"γ={gammas[idx]:.3g}\nβ(emp)={betas[idx]:.3g}, β(th)={beta_th[idx]:.3g}"
        )
        ax.set_xlim(-m * 1.2, m * 1.2)
        ax.set_ylim(-m * 1.2, m * 1.2)
        if p == 1:
            ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    # plt.show()

    # ---- Plot 2: trajectories of each θ_i as γ increases + theory circle
    plt.figure(figsize=(16, 9))
    ax = plt.gca()
    ax.scatter(mu[:, 0], mu[:, 1], marker="x", s=80, linewidths=2, label="μ")

    circle = plt.Circle((0, 0), ell, fill=False)
    ax.add_artist(circle)

    # empirical paths
    for i in range(K):
        ax.plot(thetas[:, i, 0], thetas[:, i, 1], marker="o", linewidth=1, markersize=3)

    # theory path for radius (draw as a circle of changing radius by plotting the theory points across gammas)
    for idx in range(len(gammas)):
        th_pts = r_th[idx] * u
        ax.scatter(th_pts[:, 0], th_pts[:, 1], s=10, alpha=0.15)

    ax.set_aspect("equal", "box")
    ax.set_title("Empirical centroid paths + faint theory points")
    ax.set_xlim(-m * 1.2, m * 1.2)
    ax.set_ylim(-m * 1.2, m * 1.2)
    plt.tight_layout()
    # plt.show()

    # ---- Plot 3: add theoretical bounds/curves
    plt.figure(figsize=(20, 7))

    # β vs γ
    ax1 = plt.subplot(1, 4, 1)
    ax1.plot(gammas, betas, marker="o", label="β (emp)")
    ax1.plot(gammas, beta_th, marker="s", label="β (theory)")
    ax1.axvline(gamma_leave_boundary, linestyle="--", label="γ leave boundary (theory)")
    ax1.set_title("β vs γ")
    ax1.set_xlabel("γ")
    ax1.set_ylabel("β")
    ax1.legend(fontsize=8)

    # mean radius vs γ
    ax2 = plt.subplot(1, 4, 2)
    ax2.plot(gammas, radii, marker="o", label="mean ||θ*|| (emp)")
    ax2.plot(gammas, r_th, marker="s", label="r_th (theory)")
    ax2.axvline(gamma_leave_boundary, linestyle="--")
    ax2.set_title("radius vs γ")
    ax2.set_xlabel("γ")
    ax2.set_ylabel("radius")
    ax2.legend(fontsize=8)

    # R bounds vs γ
    ax3 = plt.subplot(1, 4, 3)
    ax3.plot(gammas, R_th_beta, marker="s", label="R_th = (L/2)β_th^2")
    ax3.axhline(R_uniform, linestyle="--", label="R_uniform = 2 L ell^2")
    ax3.set_title("R bounds vs γ")
    ax3.set_xlabel("γ")
    ax3.set_ylabel("R bound")
    ax3.legend(fontsize=8)

    # gamma bound lines
    ax4 = plt.subplot(1, 4, 4)
    ax4.plot(
        gammas, gamma_bound_pointwise, marker="s", label="β_th μ_th / (2n) (pointwise)"
    )
    ax4.axhline(gamma_bound_boundary, linestyle="--", label="boundary approx γ_max")
    ax4.plot(gammas, gammas, linestyle=":", label="y=x")
    ax4.set_title("Proof-style γ bound (theory)")
    ax4.set_xlabel("γ")
    ax4.set_ylabel("bound value")
    ax4.legend(fontsize=8)

    plt.tight_layout()
    # plt.show()

    # ---- Objective vs γ (empirical)
    plt.figure(figsize=(4.8, 3.3))
    plt.plot(gammas, objs, marker="o")
    plt.title("objective vs γ (empirical)")
    plt.xlabel("γ")
    plt.ylabel("obj")
    plt.tight_layout()
    plt.show()

    return {
        "gammas": gammas,
        "thetas_emp": thetas,
        "betas_emp": betas,
        "radii_emp": radii,
        "objs_emp": objs,
        "mu": mu,
        "xbars": xbars,
        # theory
        "r_th": r_th,
        "beta_th": beta_th,
        "mu_th": mu_th,
        "gamma_bound_pointwise": gamma_bound_pointwise,
        "gamma_bound_boundary": gamma_bound_boundary,
        "gamma_leave_boundary": gamma_leave_boundary,
        "R_th_beta": R_th_beta,
        "R_uniform": R_uniform,
    }


# ---- Run
out = run_demo_with_theory(
    K=5,
    n_per_cluster=1000,
    sigma=0.5,
    m=6.0,
    ell=2.0,
    gammas=np.linspace(0.0, 2.0, 25),
    seed=1,
)
