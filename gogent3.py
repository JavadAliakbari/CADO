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


def project_to_l2_ball(a: np.ndarray, ell: float, eps: float = 1e-12) -> np.ndarray:
    a = np.asarray(a, dtype=float).reshape(-1)
    nrm = float(np.linalg.norm(a))
    if nrm <= ell:
        return a
    return a * (ell / max(nrm, eps))


# ============================================================
# Model in W-form
#
# W : (k, n), each column w_v lies in simplex over active atoms
# A : (k, d), each row a_i lies in l2 ball
# M = W^T A
# ============================================================

import numpy as np


def remove_atom_and_reproject_W(W: np.ndarray, j: int) -> np.ndarray:
    """
    Remove atom j (row j of W), then re-project each column onto the simplex.

    Parameters
    ----------
    W : (k, n) ndarray
        Current assignment matrix. Each column should lie on the simplex.
    j : int
        Row index to remove.

    Returns
    -------
    W_new : (k-1, n) ndarray
        Reduced and reprojected assignment matrix.
    """
    if W.ndim != 2:
        raise ValueError("W must be a 2D array.")
    k, n = W.shape
    if not (0 <= j < k):
        raise IndexError("j is out of range.")

    W_new = np.delete(W, j, axis=0)

    # Reproject each column to keep feasibility: w_v >= 0, sum_i w_iv = 1
    for v in range(n):
        W_new[:, v] = project_to_simplex(W_new[:, v])

    return W_new


def truncation_backward_W(
    X: np.ndarray,
    W: np.ndarray,
    A_atoms: np.ndarray,
    gamma_obj: float,
    obj_prev: float,
    obj_post: float,
    eta: float = 0.5,
    ell: float | None = None,
    trunc_refine_cycles: int = 2,
    pg_max_iter: int = 100,
    pg_step_size: float = 1e-2,
    pg_tol: float = 1e-6,
    eps: float = 1e-12,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, float, list[dict]]:
    """
    Backward truncation for the W-based CoGEnT variant.

    Threshold rule:
        F = eta * obj_prev + (1 - eta) * obj_post

    At each truncation step:
      1) Try removing each atom j.
      2) Reproject the reduced W column-wise onto the simplex.
      3) Optionally refine (W, A) a few alternating cycles.
      4) Evaluate the true objective.
      5) Accept the best removal if objective <= F.
      6) Repeat until no more removals are accepted.

    Parameters
    ----------
    X : (n, d) ndarray
        Data matrix.
    W : (k, n) ndarray
        Current assignment matrix.
    A_atoms : (k, d) ndarray
        Current atom matrix.
    gamma_obj : float
        Weight of the SON term in the true objective.
    obj_prev : float
        Objective before forward step, i.e. f(x_t).
    obj_post : float
        Objective after forward/enhancement step, i.e. f(x~_{t+1}).
    eta : float
        Threshold parameter in (0, 1]. Paper uses
            F = eta f(x_t) + (1-eta) f(x~_{t+1}).  [oai_citation:1‡1404.5692v2.pdf](sediment://file_00000000770c71f4b24f29bc2cb940ba)
    ell : float | None
        Radius of l2-ball for rows of A. Required if trunc_refine_cycles > 0.
    trunc_refine_cycles : int
        Number of alternating refinement cycles after removing a candidate atom.
        Set to 0 to skip refinement.
    pg_max_iter : int
        Max projected-gradient iterations for W refinement.
    pg_step_size : float
        Step size for projected-gradient W update.
    pg_tol : float
        Tolerance for projected-gradient W update.
    eps : float
        Numerical stability constant.
    verbose : bool
        Print truncation progress.

    Returns
    -------
    W_cur : ndarray
        Possibly reduced W.
    A_cur : ndarray
        Possibly reduced A_atoms.
    obj_cur : float
        Final objective after truncation.
    removed_log : list of dict
        Log of removed atoms.
    """
    if not (0.0 < eta <= 1.0):
        raise ValueError("eta must be in (0, 1].")

    X = np.asarray(X, dtype=float)
    W_cur = np.asarray(W, dtype=float).copy()
    A_cur = np.asarray(A_atoms, dtype=float).copy()

    n, d = X.shape
    threshold = float(eta * obj_prev + (1.0 - eta) * obj_post)

    M_cur = build_M_from_W(W_cur, A_cur, n, d)
    obj_cur = objective_full(X, M_cur, gamma_obj)

    removed_log = []

    if verbose:
        print(f"[TRUNC] threshold = {threshold:.8f}, start obj = {obj_cur:.8f}")

    # Need at least 2 atoms to remove one
    while W_cur.shape[0] > 1:
        k = W_cur.shape[0]

        best_obj = np.inf
        best_j = None
        best_W = None
        best_A = None

        for j in range(k):
            # 1) Remove atom j and reproject W to keep feasibility
            W_try = remove_atom_and_reproject_W(W_cur, j)
            A_try = np.delete(A_cur, j, axis=0)

            # 2) Optional small refinement after removal
            if trunc_refine_cycles > 0:
                if ell is None:
                    raise ValueError(
                        "ell must be provided when trunc_refine_cycles > 0."
                    )

                for _ in range(int(trunc_refine_cycles)):
                    Lc_try = build_Lc_from_WA(W_try, A_try, eps=eps)

                    W_try = update_W_projected_gradient(
                        W_init=W_try,
                        A=A_try,
                        X=X,
                        Lc=Lc_try,
                        gamma_mm=gamma_obj,
                        step_size=pg_step_size,
                        max_iter=pg_max_iter,
                        tol=pg_tol,
                        use_backtracking=True,
                        verbose=False,
                    )

                    A_try = update_A_mm_closed_form(
                        W=W_try,
                        A_current=A_try,
                        X=X,
                        gamma_mm=gamma_obj,
                        ell=ell,
                        eps=eps,
                    )

            # 3) Evaluate true objective after removal
            M_try = build_M_from_W(W_try, A_try, n, d)
            obj_try = objective_full(X, M_try, gamma_obj)

            if obj_try < best_obj:
                best_obj = float(obj_try)
                best_j = int(j)
                best_W = W_try
                best_A = A_try

        if best_j is None:
            break

        # 4) Accept only if threshold condition holds
        if best_obj <= threshold:
            if verbose:
                print(
                    f"[TRUNC] remove atom {best_j}, "
                    f"obj: {obj_cur:.8f} -> {best_obj:.8f}"
                )

            removed_log.append(
                {
                    "removed_index": best_j,
                    "obj_before": float(obj_cur),
                    "obj_after": float(best_obj),
                    "threshold": float(threshold),
                }
            )

            W_cur = best_W
            A_cur = best_A
            obj_cur = best_obj
        else:
            if verbose:
                print(
                    f"[TRUNC] stop: best removal obj={best_obj:.8f} "
                    f"> threshold={threshold:.8f}"
                )
            break

    return W_cur, A_cur, float(obj_cur), removed_log


def build_M_from_W(W: np.ndarray, A_atoms: np.ndarray, n: int, d: int) -> np.ndarray:
    if W.size == 0 or A_atoms.size == 0:
        return np.zeros((n, d), dtype=float)
    return W.T @ A_atoms


def recover_lambda_and_Esq(W: np.ndarray, eps: float = 1e-12):
    """
    Recover lambda_i = sum_v w_{iv} and e_{iv}^2 = w_{iv} / lambda_i.
    """
    if W.size == 0:
        return np.zeros(0, dtype=float), np.zeros_like(W)

    lam = W.sum(axis=1)
    E_sq = np.zeros_like(W)
    active = lam > eps
    E_sq[active, :] = W[active, :] / lam[active, None]
    return lam, E_sq


# ============================================================
# SON objective and subgradient
# ============================================================


def son_value_all_pairs(M: np.ndarray) -> float:
    n = M.shape[0]
    val = 0.0
    for u in range(n):
        for v in range(u + 1, n):
            val += float(np.linalg.norm(M[u] - M[v]))
    return val


def son_subgrad_all_pairs(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
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
    data = float(np.sum((X - M) ** 2))
    son = son_value_all_pairs(M)
    return data + gamma * son


def q_rows_full(
    X: np.ndarray, M: np.ndarray, gamma: float, eps: float = 1e-12
) -> np.ndarray:
    G_son = son_subgrad_all_pairs(M, eps=eps)
    return 2.0 * (M - X) + gamma * G_son


# ============================================================
# Forward atom: choose only a_new
# The new assignment row is initialized later through rho:
#   W_new = [ (1-rho) W ;
#             rho * 1_n^T ]
# so every data point gets rho mass on the new atom.
# ============================================================


def forward_atom_from_Q(Q: np.ndarray, ell: float, eps: float = 1e-12):
    norms = np.linalg.norm(Q, axis=1)
    v_star = int(np.argmax(norms))
    qv = Q[v_star]
    nq = float(np.linalg.norm(qv))
    if nq < eps:
        a_new = np.zeros(Q.shape[1], dtype=float)
    else:
        a_new = -ell * (qv / nq)
    return a_new, v_star


# ============================================================
# rho line search
# W_trial = [ (1-rho) W ;
#             rho * 1_n^T ]
# ============================================================


def augment_W_with_rho(W: np.ndarray, rho: float, n: int) -> np.ndarray:
    rho = float(np.clip(rho, 0.0, 1.0))
    if W.size == 0:
        return np.ones((1, n), dtype=float)
    return np.vstack([(1.0 - rho) * W, rho * np.ones((1, n), dtype=float)])


def line_search_rho_for_W(
    X: np.ndarray,
    W_old: np.ndarray,
    A_old: np.ndarray,
    a_new: np.ndarray,
    gamma_obj: float,
    n_grid: int = 21,
):
    n, d = X.shape

    def eval_obj(rho: float) -> float:
        W_trial = augment_W_with_rho(W_old, rho, n)
        A_trial = (
            np.vstack([A_old, a_new.reshape(1, d)])
            if A_old.size
            else a_new.reshape(1, d)
        )
        M_trial = build_M_from_W(W_trial, A_trial, n, d)
        return objective_full(X, M_trial, gamma_obj)

    rhos = np.linspace(0.0, 1.0, n_grid)
    vals = np.array([eval_obj(r) for r in rhos], dtype=float)
    best_idx = int(np.argmin(vals))
    return float(rhos[best_idx]), vals


# ============================================================
# Build MM Laplacian for W-update
# c_uv = 1 / ( || A^T (w_u - w_v) || + eps )
# ============================================================


def build_Lc_from_WA(
    W: np.ndarray, A_atoms: np.ndarray, eps: float = 1e-12
) -> np.ndarray:
    k, n = W.shape
    Lc = np.zeros((n, n), dtype=float)

    for u in range(n):
        wu = W[:, u]
        for v in range(u + 1, n):
            dv = wu - W[:, v]
            z = A_atoms.T @ dv
            dist = float(np.linalg.norm(z))
            c = 1.0 / max(dist, eps)
            Lc[u, u] += c
            Lc[v, v] += c
            Lc[u, v] -= c
            Lc[v, u] -= c

    return Lc


# ============================================================
# Projected-gradient update for W
# ============================================================


def compute_W_gradient(
    W: np.ndarray,
    A: np.ndarray,
    X: np.ndarray,
    B: np.ndarray,
    Lc: np.ndarray,
    gamma_mm: float,
) -> np.ndarray:
    return 2.0 * (B @ W - A @ X.T) + gamma_mm * (B @ W @ Lc)


def objective_W_mm(
    W: np.ndarray,
    A: np.ndarray,
    X: np.ndarray,
    B: np.ndarray,
    Lc: np.ndarray,
    gamma_mm: float,
) -> float:
    residual = X - W.T @ A
    fit = np.sum(residual**2)
    reg = 0.5 * gamma_mm * np.trace(B @ W @ Lc @ W.T)
    return float(fit + reg)


def update_W_projected_gradient(
    W_init: np.ndarray,
    A: np.ndarray,
    X: np.ndarray,
    Lc: np.ndarray,
    gamma_mm: float,
    step_size: float = 1e-3,
    max_iter: int = 300,
    tol: float = 1e-6,
    use_backtracking: bool = True,
    backtracking_beta: float = 0.5,
    min_step_size: float = 1e-12,
    verbose: bool = False,
) -> np.ndarray:
    W = np.asarray(W_init, dtype=float).copy()
    A = np.asarray(A, dtype=float)
    X = np.asarray(X, dtype=float)
    Lc = np.asarray(Lc, dtype=float)

    k, n = W.shape
    if A.shape[0] != k:
        raise ValueError("A must have shape (k, d) where k = W.shape[0].")
    if X.shape[0] != n:
        raise ValueError("X must have shape (n, d) where n = W.shape[1].")
    if Lc.shape != (n, n):
        raise ValueError("Lc must have shape (n, n).")

    B = A @ A.T

    current_obj = objective_W_mm(W, A, X, B, Lc, gamma_mm)
    # M = build_M_from_W(W, A, n, d)
    # current_obj = objective_full(X, M, gamma_mm)

    for it in range(max_iter):
        grad = compute_W_gradient(W, A, X, B, Lc, gamma_mm)
        step = step_size
        accepted = False

        while True:
            W_candidate = W - step * grad
            for v in range(n):
                W_candidate[:, v] = project_to_simplex(W_candidate[:, v])

            candidate_obj = objective_W_mm(W_candidate, A, X, B, Lc, gamma_mm)
            # M = build_M_from_W(W_candidate, A, n, d)
            # candidate_obj = objective_full(X, M, gamma_mm)

            if (not use_backtracking) or (candidate_obj <= current_obj):
                accepted = True
                break

            step *= backtracking_beta
            if step < min_step_size:
                break

        if not accepted:
            if verbose:
                print(f"[W-PG] stop at iter {it}: step too small.")
            break

        rel_change = np.linalg.norm(W_candidate - W) / max(np.linalg.norm(W), 1e-12)
        W = W_candidate
        current_obj = candidate_obj

        if verbose and (it % 50 == 0 or it == max_iter - 1):
            print(
                f"[W-PG] iter={it:4d}, obj={current_obj:.8e}, "
                f"step={step:.3e}, rel_change={rel_change:.3e}"
            )

        if rel_change < tol:
            break

    return W


# ============================================================
# A-update: MM closed-form, then row-wise l2 projection
# ============================================================


def update_A_mm_closed_form(
    W: np.ndarray,
    A_current: np.ndarray,
    X: np.ndarray,
    gamma_mm: float,
    ell: float,
    eps: float = 1e-12,
    ridge: float = 1e-8,
) -> np.ndarray:
    """
    Closed-form MM update:
      A^+ = (W W^T + (gamma/2) * G + ridge I)^(-1) W X
    where G = sum_{u<v} r_uv d_uv d_uv^T, d_uv = w_u - w_v,
    and r_uv = 1 / ( ||A^T d_uv|| + eps ).
    """
    k, n = W.shape
    d = X.shape[1]

    G = np.zeros((k, k), dtype=float)
    for u in range(n):
        wu = W[:, u]
        for v in range(u + 1, n):
            dv = wu - W[:, v]
            z = A_current.T @ dv
            dist = float(np.linalg.norm(z))
            r_uv = 1.0 / max(dist, eps)
            G += r_uv * np.outer(dv, dv)

    H = W @ W.T + 0.5 * gamma_mm * G + ridge * np.eye(k)
    RHS = W @ X
    A_new = np.linalg.solve(H, RHS)

    for i in range(k):
        A_new[i, :] = project_to_l2_ball(A_new[i, :], ell, eps=eps)

    return A_new


# ============================================================
# Main algorithm in W-form
# ============================================================


def cogent_structured_atoms_with_W(
    X: np.ndarray,
    gamma_obj: float,
    ell: float,
    max_iters: int = 15,
    init_with_one_atom: bool = True,
    rho_grid_size: int = 21,
    inner_alt_cycles: int = 5,
    pg_max_iter: int = 300,
    pg_step_size: float = 1e-2,
    pg_tol: float = 1e-6,
    eps: float = 1e-12,
    verbose: bool = True,
    truncation: bool = True,
    seed: int | None = None,
) -> dict:
    """
    New version:

    1) Find a_new from the forward step.
    2) Find rho_opt and initialize
           W <- [ (1-rho_opt) W ;
                  rho_opt * 1_n^T ].
    3) Append a_new to A.
    4) Update W using projected gradient.
    5) Update A using MM closed form.
    """
    if seed is not None:
        np.random.seed(seed)

    X = np.asarray(X, dtype=float)
    n, d = X.shape

    W = np.zeros((0, n), dtype=float)
    A_atoms = np.zeros((0, d), dtype=float)

    if init_with_one_atom:
        a0 = project_to_l2_ball(np.mean(X, axis=0), ell, eps=eps)
        W = np.ones((1, n), dtype=float)
        A_atoms = a0.reshape(1, d)

    obj_hist = []
    info_hist = []

    for t in range(int(max_iters)):
        M = build_M_from_W(W, A_atoms, n, d)
        obj_prev = objective_full(X, M, gamma_obj)

        Q = q_rows_full(X, M, gamma_obj, eps=eps)
        a_new, v_star = forward_atom_from_Q(Q, ell=ell, eps=eps)

        rho_opt, _ = line_search_rho_for_W(
            X=X,
            W_old=W,
            A_old=A_atoms,
            a_new=a_new,
            gamma_obj=gamma_obj,
            n_grid=rho_grid_size,
        )

        W = augment_W_with_rho(W, rho_opt, n)
        A_atoms = (
            np.vstack([A_atoms, a_new.reshape(1, d)])
            if A_atoms.size
            else a_new.reshape(1, d)
        )

        for _ in range(int(inner_alt_cycles)):
            Lc = build_Lc_from_WA(W, A_atoms, eps=eps)

            W = update_W_projected_gradient(
                W_init=W,
                A=A_atoms,
                X=X,
                Lc=Lc,
                gamma_mm=gamma_obj,
                step_size=pg_step_size,
                max_iter=pg_max_iter,
                tol=pg_tol,
                use_backtracking=True,
                verbose=False,
            )

            A_atoms = update_A_mm_closed_form(
                W=W,
                A_current=A_atoms,
                X=X,
                gamma_mm=gamma_obj,
                ell=ell,
                eps=eps,
            )

        M = build_M_from_W(W, A_atoms, n, d)
        obj_post = objective_full(X, M, gamma_obj)

        lam, E_sq = recover_lambda_and_Esq(W, eps=eps)

        # after you computed obj_prev and obj_post
        if truncation:
            W, A_atoms, obj_final, removed_log = truncation_backward_W(
                X=X,
                W=W,
                A_atoms=A_atoms,
                gamma_obj=gamma_obj,
                obj_prev=obj_prev,
                obj_post=obj_post,
                eta=0.5,  # same role as in the paper threshold
                ell=ell,
                trunc_refine_cycles=2,  # or 0 if you want faster truncation
                pg_max_iter=100,
                pg_step_size=1e-2,
                pg_tol=1e-6,
                verbose=verbose,
            )

        obj_hist.append(float(obj_post))
        info_hist.append(
            {
                "iter": int(t),
                "v_star": int(v_star),
                "rho_opt": float(rho_opt),
                "num_atoms": int(W.shape[0]),
                "obj_prev": float(obj_prev),
                "obj_post": float(obj_post),
                "q_norm_max": float(np.max(np.linalg.norm(Q, axis=1))),
            }
        )

        if verbose:
            print(
                f"iter={t:02d} "
                f"K={W.shape[0]} "
                f"obj_prev={obj_prev:.4f} "
                f"obj_post={obj_post:.4f} "
                f"rho={rho_opt:.4f} "
                f"v_star={v_star}"
            )

    M_final = build_M_from_W(W, A_atoms, n, d)
    lam, E_sq = recover_lambda_and_Esq(W, eps=eps)

    return {
        "M": M_final,
        "W": W,  # shape (k, n)
        "S": W.T,  # compatibility view, shape (n, k)
        "A_atoms": A_atoms,  # shape (k, d)
        "lam": lam,  # lambda_i = sum_v w_iv
        "E_sq": E_sq,  # shape (k, n), e_{iv}^2
        "obj_hist": np.array(obj_hist, dtype=float),
        "info_hist": info_hist,
    }


# ============================================================
# Plot helpers
# ============================================================


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


def plot_solution(X, labels, centroids, ell, res, gamma_val):
    plt.figure(figsize=(7, 6))
    for k in np.unique(labels):
        idx = labels == k
        plt.scatter(X[idx, 0], X[idx, 1], alpha=0.4)

    if centroids is not None:
        C = np.asarray(centroids, dtype=float)
        plt.scatter(
            C[:, 0], C[:, 1], marker="x", s=200, color="black", label="True centroids"
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


# ============================================================
# Minimal test
# ============================================================

if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)

    d = 2
    ell = 1.0
    K_true = 3
    n_per_cluster = 60
    sigma = 0.10

    radii = [1.5, 1.7, 2.0]
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

    gamma_val = 0.01

    res = cogent_structured_atoms_with_W(
        X=X,
        gamma_obj=gamma_val,
        ell=ell,
        max_iters=10,
        init_with_one_atom=True,
        rho_grid_size=21,
        inner_alt_cycles=4,
        pg_max_iter=200,
        pg_step_size=1e-2,
        pg_tol=1e-6,
        verbose=True,
        truncation=False,  # set to True to enable backward truncation
        seed=seed,
    )

    print("\nFinal number of atoms:", res["A_atoms"].shape[0])
    print("Final objective:", res["obj_hist"][-1] if res["obj_hist"].size else None)
    print("Final lambdas:")
    print(np.array2string(res["lam"], precision=4, suppress_small=True))
    print("Final atoms A:")
    print(np.array2string(res["A_atoms"], precision=4, suppress_small=True))

    plot_solution(X, labels, centroids, ell, res, gamma_val)
    plt.show()
