import numpy as np

import matplotlib.pyplot as plt


def generate_clusters_with_directions(
    d,
    n_per_cluster,
    sigma,
    ell,
    radii,  # length K, each > ell
    directions,  # shape (K, d), will be normalized row-wise
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

    # normalize directions
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    dirs = directions / norms

    centroids = dirs * radii[:, None]  # outside ell-ball by construction

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
    """
    Generate K direction vectors in R^d.
    Each row activates `num_active` random coordinates.

    - Rows are different
    - Values are ±1
    - Generator later normalizes them

    Requires: num_active <= d
    """

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

            # ensure rows are different
            key = tuple(vec)
            if key not in used_patterns:
                used_patterns.add(key)
                directions[k] = vec
                break

    return directions


# ============================================================
# Projections
# ============================================================


def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """
    Euclidean projection onto simplex {x >= 0, sum x = 1}.
    """
    v = np.asarray(v, dtype=float).reshape(-1)
    if v.size == 1:
        return np.array([1.0], dtype=float)

    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - 1.0))[0]
    if rho.size == 0:
        w = np.maximum(v, 0.0)
        s = w.sum()
        return w / s if s > 0 else np.ones_like(w) / len(w)

    rho = int(rho[-1])
    theta = (cssv[rho] - 1.0) / (rho + 1)
    return np.maximum(v - theta, 0.0)


def project_nonneg_l1_ball(w: np.ndarray, tau: float) -> np.ndarray:
    """
    Projection onto {w >= 0, sum w = tau} (boundary of nonnegative L1 ball).
    Projects onto the scaled simplex with sum equal to tau.
    """
    w = np.asarray(w, dtype=float).reshape(-1)
    if w.size == 1:
        return np.array([tau], dtype=float)

    # Project onto simplex {x >= 0, sum x = tau}
    u = np.sort(w)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - tau))[0]
    if rho.size == 0:
        return np.ones_like(w) * (tau / len(w))
    rho = int(rho[-1])
    theta = (cssv[rho] - tau) / (rho + 1)
    return np.maximum(w - theta, 0.0)


def project_to_l2_ball(a: np.ndarray, ell: float, eps: float = 1e-12) -> np.ndarray:
    """
    Projection onto {a: ||a||_2 <= ell}.
    """
    a = np.asarray(a, dtype=float).reshape(-1)
    nrm = float(np.linalg.norm(a))
    if nrm <= ell:
        return a
    return a * (ell / max(nrm, eps))


# ============================================================
# Model: M = sum_i lambda_i s_i a_i^T
# S: (n,k) columns s_i in simplex Delta_n
# A_atoms: (k,d) rows a_i in l2 ball Theta = B_ell
# lam: (k,) nonnegative, sum <= tau
# ============================================================


def build_M(S: np.ndarray, A_atoms: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """
    M = (S * lam) @ A_atoms, where lam broadcasts over columns of S.
    """
    if S.size == 0:
        n = int(A_atoms.shape[0]) if A_atoms.ndim == 2 else 0
        return np.zeros((n, 0), dtype=float)
    return (S * lam.reshape(1, -1)) @ A_atoms


# ============================================================
# SON term: objective and subgradient wrt rows of M
# Objective uses sum_{u<v} ||mu_u - mu_v||
# Subgradient row v is sum_{u != v} (mu_v - mu_u)/||mu_v - mu_u||
# ============================================================


def son_value_all_pairs(M: np.ndarray) -> float:
    n = M.shape[0]
    val = 0.0
    for u in range(n):
        for v in range(u + 1, n):
            val += float(np.linalg.norm(M[u] - M[v]))
    return val


def son_subgrad_all_pairs(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Returns G_son (n,d) such that:
      G_son[v] in partial_{mu_v} sum_{u<v} ||mu_u - mu_v||_2
    """
    M = np.asarray(M, dtype=float)
    n, d = M.shape
    G = np.zeros((n, d), dtype=float)

    for v in range(n):
        acc = np.zeros(d, dtype=float)
        for u in range(n):
            if u == v:
                continue
            diff = M[v] - M[u]
            nd = float(np.linalg.norm(diff))
            if nd > eps:
                acc += diff / nd
        G[v] = acc
    return G


def objective_full(X: np.ndarray, M: np.ndarray, gamma: float) -> float:
    data = float(np.mean((X - M) ** 2))
    son = son_value_all_pairs(M)
    return data + gamma * son


def q_rows_full(
    X: np.ndarray, M: np.ndarray, gamma: float, eps: float = 1e-12
) -> np.ndarray:
    """
    q_v = partial_{mu_v} [ sum_v ||x_v - mu_v||^2 + gamma * sum_{u<v} ||mu_u - mu_v|| ]
        = 2(mu_v - x_v) + gamma * SON_subgrad_row
    """
    G_son = son_subgrad_all_pairs(M, eps=eps)
    return 2.0 * (M - X) + gamma * G_son


# ============================================================
# Forward step (Theta = l2 ball)
# Atom is s_new a_new^T with s_new = e_v* (one-hot) and a_new = -ell * q_v/||q_v||
# ============================================================


def forward_atom_onehot(
    Q: np.ndarray, ell: float, eps: float = 1e-12, iter_: int = 10
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Returns (s_new, a_new)
    """
    a = np.random.randn(Q.shape[1])  # dummy initialization, will be overwritten
    for i in range(iter_):
        H = -np.diagonal(Q @ a)
        s = np.linalg.eigh(H)[1][:, -1]  # one-hot at v* with max H[v]
        g = Q.T @ s
        a = project_to_l2_ball(-g, ell, eps=eps)
    return s, a


# ============================================================
# Warm start on lambda with global budget tau
# lam <- (1-rho) lam, lam_new <- rho*tau, then project to {lam>=0, sum<=tau}
# ============================================================


def warm_start_lambda(lam: np.ndarray, tau: float, rho: float) -> np.ndarray:
    lam = np.asarray(lam, dtype=float).reshape(-1)
    lam = (1.0 - rho) * lam
    lam = np.append(lam, rho * tau)
    lam = project_nonneg_l1_ball(lam, tau)
    return lam


def line_search_rho(
    X: np.ndarray,
    M_old: np.ndarray,
    s_new: np.ndarray,
    a_new: np.ndarray,
    tau: float,
    gamma: float,
    n_grid: int = 20,
    refine_iters: int = 2,
    eps: float = 1e-12,
) -> float:
    """
    Find rho in [0, 1] that minimizes:
        f(rho * tau * s_new @ a_new^T + (1 - rho) * M_old)

    Uses grid search followed by golden section refinement.

    Parameters:
    -----------
    X : (n, d) data matrix
    M_old : (n, d) current model matrix
    s_new : (n,) new atom's simplex vector (one-hot)
    a_new : (d,) new atom's dictionary vector
    tau : global lambda budget
    gamma : regularization parameter for SON term
    n_grid : number of grid points for initial search
    refine_iters : number of golden section refinement iterations

    Returns:
    --------
    rho_opt : optimal rho in [0, 1]
    """
    atom_matrix = tau * np.outer(s_new, a_new)  # new atom contribution at rho=1

    def eval_obj(rho: float) -> float:
        M_trial = (1.0 - rho) * M_old + rho * atom_matrix
        return objective_full(X, M_trial, gamma)

    # Grid search for initial bracket
    rhos = np.linspace(0.0, 1.0, n_grid)
    objs = [eval_obj(r) for r in rhos]
    best_idx = int(np.argmin(objs))
    return rhos[best_idx]

    # Bracket around best grid point
    # lo = rhos[max(0, best_idx - 1)]
    # hi = rhos[min(n_grid - 1, best_idx + 1)]

    # # Golden section refinement
    # golden = (np.sqrt(5.0) - 1.0) / 2.0

    # for _ in range(refine_iters):
    #     if hi - lo < eps:
    #         break

    #     d = golden * (hi - lo)
    #     r1 = hi - d
    #     r2 = lo + d

    #     f1 = eval_obj(r1)
    #     f2 = eval_obj(r2)

    #     if f1 < f2:
    #         hi = r2
    #     else:
    #         lo = r1

    # rho_opt = 0.5 * (lo + hi)
    # return float(rho_opt)


# ============================================================
# Enhancement step: surrogate based on linearization of SON
# Y = X - (gamma/2) * S_son, where S_son is a SON subgradient at current M
# Then fit Y by alternating closed-form updates per atom:
#   a_i = Pi_{B_ell}( R_i^T s_i / ||s_i||^2 )
#   s_i = Pi_{Delta_n}( R_i a_i / ||a_i||^2 )
#   lam_i = max(0, (s_i^T R_i a_i)/(||s_i||^2 ||a_i||^2))
# Finally project lam onto {lam>=0, sum<=tau}
# ============================================================


def enhancement_alternating_surrogate(
    X: np.ndarray,
    S: np.ndarray,
    A_atoms: np.ndarray,
    lam: np.ndarray,
    gamma: float,
    tau: float,
    ell: float,
    cycles: int = 10,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns updated (M, S, A_atoms, lam). (M can be recomputed by caller)
    """
    n, d = X.shape
    k = lam.shape[0]

    if k == 0:
        return np.zeros((n, d), dtype=float), S, A_atoms, lam

    M = build_M(S, A_atoms, lam)
    S_son = son_subgrad_all_pairs(M, eps=eps)
    Y = X - 0.5 * gamma * S_son

    lam = project_nonneg_l1_ball(lam, tau)

    for _ in range(int(cycles)):
        M = build_M(S, A_atoms, lam)

        for i in range(k):
            si = S[:, i].copy()
            ai = A_atoms[i, :].copy()
            li = float(lam[i])

            Ri = Y - M + li * np.outer(si, ai)

            si_norm2 = float(np.dot(si, si))
            if si_norm2 < eps:
                si_norm2 = eps

            ai_update = (Ri.T @ si) / si_norm2
            ai_update = project_to_l2_ball(ai_update, ell, eps=eps)

            ai_norm2 = float(np.dot(ai_update, ai_update))
            if ai_norm2 < eps:
                ai_norm2 = eps

            si_update = (Ri @ ai_update) / ai_norm2
            si_update = project_to_simplex(si_update)

            num = float(si_update.T @ (Ri @ ai_update))
            den = float(np.dot(si_update, si_update) * np.dot(ai_update, ai_update))
            den = max(den, eps)
            li_update = max(0.0, num / den)

            S[:, i] = si_update
            A_atoms[i, :] = ai_update
            lam[i] = li_update

            M = M - li * np.outer(si, ai) + lam[i] * np.outer(S[:, i], A_atoms[i, :])

        lam = project_nonneg_l1_ball(lam, tau)

    M = build_M(S, A_atoms, lam)
    return M, S, A_atoms, lam


# ============================================================
# Backward search truncation with threshold
# T = (1-kappa) obj_prev + kappa obj_post
# Try removing one atom at a time, re-run enhancement for a few cycles, accept if obj <= T
# ============================================================


def backward_search_truncation(
    X: np.ndarray,
    S: np.ndarray,
    A_atoms: np.ndarray,
    lam: np.ndarray,
    gamma: float,
    tau: float,
    ell: float,
    obj_prev: float,
    obj_post: float,
    kappa: float = 0.5,
    max_remove: int | None = None,
    enh_cycles_after_remove: int = 5,
    search_max_atoms: int | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, list[dict]]:
    if not (0.0 < kappa <= 1.0):
        raise ValueError("kappa must be in (0, 1].")

    rng = np.random.default_rng(seed)
    threshold = (1.0 - kappa) * float(obj_prev) + kappa * float(obj_post)

    cur_S = S
    cur_A = A_atoms
    cur_lam = lam

    cur_M = build_M(cur_S, cur_A, cur_lam)
    cur_obj = objective_full(X, cur_M, gamma)

    removed_log: list[dict] = []
    removed = 0

    while cur_lam.shape[0] > 1:
        k = cur_lam.shape[0]
        candidates = np.arange(k)
        if search_max_atoms is not None and search_max_atoms < k:
            candidates = rng.choice(candidates, size=search_max_atoms, replace=False)

        best_obj = np.inf
        best_j = None
        best_pack = None

        for j in candidates:
            S_try = np.delete(cur_S, j, axis=1)
            A_try = np.delete(cur_A, j, axis=0)
            lam_try = np.delete(cur_lam, j, axis=0)

            lam_try = project_nonneg_l1_ball(lam_try, tau)

            M_try, S_try, A_try, lam_try = enhancement_alternating_surrogate(
                X=X,
                S=S_try,
                A_atoms=A_try,
                lam=lam_try,
                gamma=gamma,
                tau=tau,
                ell=ell,
                cycles=enh_cycles_after_remove,
            )
            obj_try = objective_full(X, M_try, gamma)

            if obj_try < best_obj:
                best_obj = float(obj_try)
                best_j = int(j)
                best_pack = (S_try, A_try, lam_try, M_try)

        if best_j is None:
            break

        if best_obj <= threshold:
            removed_log.append({"removed_index": best_j, "obj_after": best_obj})
            cur_S, cur_A, cur_lam, cur_M = best_pack
            cur_obj = best_obj
            removed += 1
            if max_remove is not None and removed >= max_remove:
                break
        else:
            break

    return cur_S, cur_A, cur_lam, float(cur_obj), removed_log


# ============================================================
# Main CoGEnT-style algorithm for your structured atoms
# ============================================================


def cogent_structured_atoms(
    X: np.ndarray,
    gamma: float,
    ell: float,
    tau: float,
    max_iters: int = 50,
    init_with_one_atom: bool = True,
    enh_cycles: int = 10,
    trunc_every: int = 1,
    trunc_kappa: float = 0.5,
    trunc_max_remove: int | None = 1,
    trunc_search_max_atoms: int | None = None,
    eps: float = 1e-12,
    verbose: bool = True,
    seed: int | None = None,
) -> dict:
    """
    CoGEnT-style loop for:
      min_M  sum_v ||x_v - mu_v||^2 + gamma * sum_{u<v} ||mu_u - mu_v||
      s.t.   M = sum_i lambda_i s_i a_i^T,
             lambda_i >= 0, sum_i lambda_i <= tau,
             s_i in Delta_n, a_i in B_ell.

    Forward step uses exact linear oracle over atoms (s in simplex, a in ball),
    which reduces to one-hot s at v* and a aligned with -q_{v*}.
    Enhancement uses SON-linearized surrogate with closed-form alternating updates.
    Truncation uses CoGEnT backward search with threshold.
    """
    if seed is not None:
        np.random.seed(seed)

    X = np.asarray(X, dtype=float)
    n, d = X.shape

    S = np.zeros((n, 0), dtype=float)
    A_atoms = np.zeros((0, d), dtype=float)
    lam = np.zeros((tau,), dtype=float)

    if init_with_one_atom:
        s0 = np.ones(n, dtype=float) / n
        a0 = project_to_l2_ball(np.mean(X, axis=0), ell, eps=eps)
        S = s0.reshape(n, 1)
        A_atoms = a0.reshape(1, d)
        lam = np.array([tau], dtype=float)
        lam = project_nonneg_l1_ball(lam, tau)

    obj_hist = []
    info_hist = []

    for t in range(int(max_iters)):
        M = build_M(S, A_atoms, lam) if lam.size > 0 else np.zeros((n, d), dtype=float)
        obj_prev = objective_full(X, M, gamma)

        Q = q_rows_full(X, M, gamma, eps=eps)

        s_new, a_new = forward_atom_onehot(Q, ell=ell, eps=eps)

        # Line search to find optimal rho
        rho_opt = line_search_rho(
            X=X,
            M_old=M,
            s_new=s_new,
            a_new=a_new,
            tau=tau,
            gamma=gamma,
            n_grid=20,
            refine_iters=3,
            eps=eps,
        )

        S = np.column_stack([S, s_new]) if S.size else s_new.reshape(n, 1)
        A_atoms = (
            np.vstack([A_atoms, a_new.reshape(1, d)])
            if A_atoms.size
            else a_new.reshape(1, d)
        )

        lam = warm_start_lambda(lam, tau=tau, rho=rho_opt)
        M = build_M(S, A_atoms, lam) if lam.size > 0 else np.zeros((n, d), dtype=float)

        # M_post, S, A_atoms, lam = enhancement_alternating_surrogate(
        #     X=X,
        #     S=S,
        #     A_atoms=A_atoms,
        #     lam=lam,
        #     gamma=gamma,
        #     tau=tau,
        #     ell=ell,
        #     cycles=enh_cycles,
        #     eps=eps,
        # )
        obj_post = objective_full(X, M, gamma)

        obj_final = obj_post
        removed_log = []
        threshold = (1.0 - trunc_kappa) * obj_prev + trunc_kappa * obj_post

        if (
            trunc_every is not None
            and trunc_every > 0
            and ((t + 1) % trunc_every == 0)
            and lam.size > 1
        ):
            S, A_atoms, lam, obj_final, removed_log = backward_search_truncation(
                X=X,
                S=S,
                A_atoms=A_atoms,
                lam=lam,
                gamma=gamma,
                tau=tau,
                ell=ell,
                obj_prev=obj_prev,
                obj_post=obj_post,
                kappa=trunc_kappa,
                max_remove=trunc_max_remove,
                enh_cycles_after_remove=max(1, enh_cycles // 2),
                search_max_atoms=trunc_search_max_atoms,
                seed=seed,
            )

        obj_hist.append(float(obj_final))
        info_hist.append(
            {
                "iter": int(t),
                "q_norm_max": float(np.max(np.linalg.norm(Q, axis=1))),
                "num_atoms": int(lam.size),
                "sum_lam": float(lam.sum()),
                "obj_prev": float(obj_prev),
                "obj_post": float(obj_post),
                "obj_final": float(obj_final),
                "threshold": float(threshold),
                "removed": removed_log,
                "rho_opt": float(rho_opt),
            }
        )

        if verbose:
            print(
                f"iter={t} K={lam.size} "
                f"obj_prev={obj_prev:.3f} obj_post={obj_post:.3f} obj_final={obj_final:.3f} "
                f"sum_lam={lam.sum():.3f} rho={rho_opt:.4f}"
            )

    M_final = (
        build_M(S, A_atoms, lam) if lam.size > 0 else np.zeros((n, d), dtype=float)
    )

    return {
        "M": M_final,
        "S": S,
        "A_atoms": A_atoms,
        "lam": lam,
        "obj_hist": np.array(obj_hist, dtype=float),
        "info_hist": info_hist,
    }


def plot_M_solutions(X, sol, result, K_true, labels, centroids, ell, gamma, gamma_val):
    # plot the two solutions:
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for k in range(K_true):
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
    plt.scatter(
        result["dictt"][:, 0],
        result["dictt"][:, 1],
        color="red",
        marker="x",
        label="CoGEnT atoms",
    )
    plt.title(f"CoGEnT solution (gamma={gamma_val})")
    plt.legend()


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
    # plt.show()


# ============================================================
# Minimal test (small n due to all-pairs SON)
# ============================================================

if __name__ == "__main__":

    seed = 0
    np.random.seed(seed=seed)

    d = 2
    ell = 1.0
    K_true = 3
    n_per_cluster = 100  # keep small because SON is all pairs (n^2)
    n = K_true * n_per_cluster
    sigma = 0.1

    radii = [1.5, 1.7, 2.0]  # all > ell (centroids outside the ell-ball)
    directions = random_sparse_directions(K_true, d, num_active=1, seed=seed)

    X, labels, centroids, _ = generate_clusters_with_directions(
        d=d,
        n_per_cluster=n_per_cluster,
        sigma=sigma,
        ell=ell,
        radii=radii,
        directions=directions,
        normalize_data=False,
        seed=seed,
    )

    plot_data(X, labels, centroids, ell)

    res = cogent_structured_atoms(
        X=X,
        gamma=0.0 / n,  # start with 0, then increase (e.g., 0.05 / n, 0.1 / n, ...)
        ell=ell,
        tau=n,
        max_iters=15,
        enh_cycles=8,
        trunc_every=1,
        trunc_kappa=0.7,
        trunc_max_remove=1,
        verbose=True,
        seed=0,
    )

    print("Final K:", res["lam"].size)
    print("Final objective:", res["obj_hist"][-1] if res["obj_hist"].size else None)
    print("Final atoms A:")
    print(np.array2string(res["A_atoms"], precision=2, suppress_small=True))
