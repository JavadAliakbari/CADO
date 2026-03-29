import numpy as np
import copy as cp
from utils import _fw_lmo_columns_nonneg_unit_ball

# ============================================================
# Simplex projection
# ============================================================


def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """
    Euclidean projection onto the probability simplex:
        {w : w >= 0, sum(w) = 1}
    """
    v = np.asarray(v, dtype=float).ravel()
    n = v.size

    if n == 1:
        return np.array([1.0], dtype=float)

    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1)

    cond = u - cssv / ind > 0
    if not np.any(cond):
        theta = cssv[-1] / n
    else:
        rho = ind[cond][-1]
        theta = cssv[rho - 1] / rho

    return np.maximum(v - theta, 0.0)


def project_columns_to_simplex(W: np.ndarray) -> np.ndarray:
    """
    Vectorized projection of each column of W onto the simplex.
    W: (k, n) - each column is projected to the probability simplex.
    """
    k, n = W.shape
    if k == 1:
        return np.ones((1, n), dtype=float)

    # Sort each column in descending order
    u = np.sort(W, axis=0)[::-1]  # shape (k, n)

    # Cumulative sum along axis 0
    cssv = np.cumsum(u, axis=0)  # shape (k, n)

    # Compute condition: u[i] * (i+1) > cssv[i] - 1
    indices = np.arange(1, k + 1)[:, np.newaxis]  # shape (k, 1)
    cond = u * indices > (cssv - 1.0)  # shape (k, n)

    # Find rho for each column (last True index)
    rho = k - 1 - np.argmax(cond[::-1], axis=0)  # shape (n,)

    # Handle case where no condition is True
    any_true = np.any(cond, axis=0)

    # Compute theta for each column
    theta = (cssv[rho, np.arange(n)] - 1.0) / (rho + 1)  # shape (n,)

    # Project
    W_proj = np.maximum(W - theta[np.newaxis, :], 0.0)

    # Handle columns where no condition was True
    if not np.all(any_true):
        bad_cols = ~any_true
        W_bad = np.maximum(W[:, bad_cols], 0.0)
        sums = W_bad.sum(axis=0, keepdims=True)
        sums = np.where(sums > 0, sums, 1.0)
        W_proj[:, bad_cols] = W_bad / sums

    return W_proj


# ============================================================
# Original W-objective and helpers
# ============================================================


def build_M_from_W(W: np.ndarray, A: np.ndarray) -> np.ndarray:
    return W.T @ A


def objective_W_original(
    W: np.ndarray,
    A: np.ndarray,
    X: np.ndarray,
    gamma_obj: float,
) -> float:
    """
    Original W-subproblem objective (vectorized):

        ||X - W^T A||_F^2 + gamma * sum_{u<v} ||A^T(w_u - w_v)||_2

    W: (k, n)
    A: (k, d)
    X: (n, d)
    """
    M = build_M_from_W(W, A)
    fit = np.sum((X - M) ** 2)

    # Vectorized fusion term computation
    # Compute all pairwise differences using upper triangle indices
    _, n = W.shape
    u_idx, v_idx = np.triu_indices(n, k=1)

    # DW[:, p] = w_u - w_v for each pair p
    DW = W[:, u_idx] - W[:, v_idx]  # (k, P)

    # Z[:, p] = A^T @ (w_u - w_v)
    Z = A.T @ DW  # (d, P)

    # Sum of norms
    fusion = np.sum(np.linalg.norm(Z, axis=0))

    return float(fit + gamma_obj * fusion)


def compute_W_subgradient_original_fast(
    W: np.ndarray,
    A: np.ndarray,
    X: np.ndarray,
    B: np.ndarray | None = None,
    AX: np.ndarray | None = None,
    gamma_obj: float = 0.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Vectorized subgradient wrt W for the original objective.

    W: (k, n), A: (k, d), X: (n, d)
    """
    k, n = W.shape
    if B is None:
        B = A @ A.T
    if AX is None:
        AX = A @ X.T

    # Smooth part
    G = 2.0 * (B @ W - AX)

    # Pair indices
    u_idx, v_idx = np.triu_indices(n, k=1)

    # DW[:, p] = w_u - w_v
    DW = W[:, u_idx] - W[:, v_idx]  # (k, P)

    # Z[:, p] = A^T (w_u - w_v)
    Z = A.T @ DW  # (d, P)

    norms = np.linalg.norm(Z, axis=0)  # (P,)
    alpha = np.zeros_like(norms)
    mask = norms > eps
    alpha[mask] = 1.0 / norms[mask]

    pair_grad = (B @ DW) * alpha[None, :]  # (k, P)

    np.add.at(G, (slice(None), u_idx), gamma_obj * pair_grad)
    np.add.at(G, (slice(None), v_idx), -gamma_obj * pair_grad)

    return G


# ============================================================
# Column gradient for block coordinate methods
# ============================================================


def compute_single_column_subgradient_original(
    W: np.ndarray,
    A: np.ndarray,
    X: np.ndarray,
    gamma_obj: float,
    v: int,
    B: np.ndarray | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Subgradient wrt a single column w_v of W (vectorized).

    Returns g_v in R^k.
    """
    if B is None:
        B = A @ A.T

    # smooth part: 2(B w_v - A x_v)
    g = 2.0 * (B @ W[:, v] - A @ X[v])

    k, n = W.shape
    wv = W[:, v]

    # Vectorized fusion subgradient
    # Compute differences wv - W[:, u] for all u != v
    # Create mask to exclude column v
    mask = np.ones(n, dtype=bool)
    mask[v] = False

    # DW[:, u] = wv - W[:, u] for u != v
    DW = wv[:, np.newaxis] - W[:, mask]  # (k, n-1)

    # Z[:, u] = A^T @ (wv - W[:, u])
    Z = A.T @ DW  # (d, n-1)

    # Compute norms
    norms = np.linalg.norm(Z, axis=0)  # (n-1,)

    # Safe division
    alpha = np.zeros_like(norms)
    valid = norms > eps
    alpha[valid] = 1.0 / norms[valid]

    # Sum contributions: sum_u (B @ (wv - W[:,u])) / ||z_uv||
    # = B @ sum_u (wv - W[:,u]) * alpha[u]
    # = B @ DW @ alpha
    g += gamma_obj * (B @ (DW @ alpha))

    return g


# ============================================================
# 1) Projected gradient / subgradient
# ============================================================


def update_W_projected_gradient_original(
    W_init: np.ndarray,
    A: np.ndarray,
    X: np.ndarray,
    gamma_obj: float,
    step_size: float = 1e-3,
    max_iter: int = 300,
    tol: float = 1e-6,
    eps: float = 1e-12,
    use_backtracking: bool = True,
    backtracking_beta: float = 0.5,
    min_step_size: float = 1e-14,
    verbose: bool = False,
) -> np.ndarray:
    W = np.asarray(W_init, dtype=float).copy()
    A = np.asarray(A, dtype=float)
    X = np.asarray(X, dtype=float)
    B = A @ A.T
    AX = A @ X.T

    for v in range(W.shape[1]):
        W[:, v] = project_to_simplex(W[:, v])

    current_obj = objective_W_original(W, A, X, gamma_obj)

    for it in range(max_iter):
        G = compute_W_subgradient_original_fast(W, A, X, B, AX, gamma_obj, eps=eps)

        step = step_size
        accepted = False

        while True:
            W_candidate = W - step * G
            # Vectorized simplex projection for all columns
            W_candidate = project_columns_to_simplex(W_candidate)

            candidate_obj = objective_W_original(W_candidate, A, X, gamma_obj)

            if (not use_backtracking) or (candidate_obj <= current_obj):
                accepted = True
                break

            step *= backtracking_beta
            if step < min_step_size:
                break

        if not accepted:
            if verbose:
                print(f"[PG] stop at iter {it}: step too small.")
            break

        rel_change = np.linalg.norm(W_candidate - W) / max(np.linalg.norm(W), 1e-12)
        W = W_candidate
        current_obj = candidate_obj

        if verbose and (it % 25 == 0 or it == max_iter - 1):
            print(
                f"[PG] iter={it:4d}, obj={current_obj:.8e}, step={step:.3e}, rel_change={rel_change:.3e}"
            )

        if rel_change < tol:
            break

    return W


# ============================================================
# 2) Block coordinate descent on columns of W
# ============================================================


def update_W_block_coordinate_descent(
    W_init: np.ndarray,
    A: np.ndarray,
    X: np.ndarray,
    gamma_obj: float,
    step_size: float = 1e-2,
    max_epochs: int = 50,
    tol: float = 1e-6,
    eps: float = 1e-12,
    use_backtracking: bool = True,
    backtracking_beta: float = 0.5,
    min_step_size: float = 1e-14,
    randomized: bool = False,
    seed: int | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Block coordinate descent updating one column w_v at a time.

    Each column update is:
        w_v <- Proj_simplex(w_v - eta * g_v)
    with optional backtracking on the full objective.
    """
    rng = np.random.default_rng(seed)
    W = W_init
    k, n = W.shape
    B = A @ A.T

    current_obj = objective_W_original(W, A, X, gamma_obj)

    for epoch in range(max_epochs):
        W_old = W.copy()

        order = np.arange(n)
        if randomized:
            rng.shuffle(order)

        for v in order:
            g_v = compute_single_column_subgradient_original(
                W=W, A=A, X=X, gamma_obj=gamma_obj, v=v, B=B, eps=eps
            )

            step = step_size
            accepted = False
            old_col = W[:, v].copy()

            while True:
                new_col = project_to_simplex(old_col - step * g_v)
                W[:, v] = new_col
                candidate_obj = objective_W_original(W, A, X, gamma_obj)

                if (not use_backtracking) or (candidate_obj <= current_obj):
                    accepted = True
                    current_obj = candidate_obj
                    break

                W[:, v] = old_col
                step *= backtracking_beta
                if step < min_step_size:
                    break

            if not accepted:
                W[:, v] = old_col

        rel_change = np.linalg.norm(W - W_old) / max(np.linalg.norm(W_old), 1e-12)

        if verbose and (epoch % 5 == 0 or epoch == max_epochs - 1):
            print(
                f"[BCD] epoch={epoch:4d}, obj={current_obj:.8e}, rel_change={rel_change:.3e}"
            )

        if rel_change < tol:
            break

    return W


# ============================================================
# 3) Mirror descent on simplex columns
# ============================================================


def mirror_descent_simplex_step(
    w: np.ndarray, g: np.ndarray, step: float, eps: float = 1e-16
) -> np.ndarray:
    """
    Entropic mirror descent step on the simplex:
        w_i^+ proportional to w_i * exp(-step * g_i)
    """
    w_safe = np.maximum(w, eps)
    z = np.log(w_safe) - step * g
    z -= np.max(z)  # numerical stability
    w_new = np.exp(z)
    s = np.sum(w_new)
    if s <= eps:
        return np.ones_like(w_new) / len(w_new)
    return w_new / s


def update_W_mirror_descent_full(
    W_init: np.ndarray,
    A: np.ndarray,
    X: np.ndarray,
    gamma_obj: float,
    step_size: float = 1e-2,
    max_iter: int = 200,
    tol: float = 1e-6,
    eps: float = 1e-12,
    use_backtracking: bool = True,
    backtracking_beta: float = 0.5,
    min_step_size: float = 1e-14,
    verbose: bool = False,
):
    """
    Full-matrix mirror descent for W.

    Each column of W stays on the simplex.
    """
    # W = np.asarray(W_init, dtype=float).copy()
    W = W_init
    B = A @ A.T
    AX = A @ X.T
    # k, n = W.shape

    # make sure W is feasible
    # W = np.maximum(W, eps)
    # W /= W.sum(axis=0, keepdims=True)

    # current_obj = objective_W_original(W, A, X, gamma_obj)

    for it in range(max_iter):
        G = compute_W_subgradient_original_fast(W, A, X, B, AX, gamma_obj, eps=eps)

        # step = step_size
        # accepted = False

        # while True:
        Z = np.log(np.maximum(W, eps)) - step_size * G
        Z -= np.max(Z, axis=0, keepdims=True)  # stability
        W_candidate = np.exp(Z)
        W_candidate /= np.sum(W_candidate, axis=0, keepdims=True)

        # candidate_obj = objective_W_original(W_candidate, A, X, gamma_obj)

        # if (not use_backtracking) or (candidate_obj <= current_obj):
        #     accepted = True
        #     break

        # step *= backtracking_beta
        # if step < min_step_size:
        #     break

        # if not accepted:
        #     if verbose:
        #         print(f"[W-MD-full] stop at iter {it}: step too small.")
        #     break

        rel_change = np.linalg.norm(W_candidate - W) / max(np.linalg.norm(W), 1e-12)
        W = W_candidate
        # current_obj = candidate_obj

        if verbose and (it % 25 == 0 or it == max_iter - 1):
            print(
                # f"[W-MD-full] iter={it:4d}, obj={current_obj:.8e}, "
                f"step={step_size:.3e}, rel_change={rel_change:.3e}"
            )

        if rel_change < tol:
            break

    return W


def update_W_mirror_descent(
    W_init: np.ndarray,
    A: np.ndarray,
    X: np.ndarray,
    gamma_obj: float,
    step_size: float = 1e-2,
    max_epochs: int = 50,
    tol: float = 1e-6,
    eps: float = 1e-12,
    use_backtracking: bool = True,
    backtracking_beta: float = 0.5,
    min_step_size: float = 1e-14,
    randomized: bool = False,
    seed: int | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Mirror descent updating one simplex column at a time.
    """
    rng = np.random.default_rng(seed)

    W = np.asarray(W_init, dtype=float).copy()
    A = np.asarray(A, dtype=float)
    X = np.asarray(X, dtype=float)

    _, n = W.shape
    B = A @ A.T

    # Vectorized initial projection
    W = project_columns_to_simplex(W)

    current_obj = objective_W_original(W, A, X, gamma_obj)

    for epoch in range(max_epochs):
        W_old = W.copy()

        order = np.arange(n)
        if randomized:
            rng.shuffle(order)

        for v in order:
            g_v = compute_single_column_subgradient_original(
                W=W, A=A, X=X, gamma_obj=gamma_obj, v=v, B=B, eps=eps
            )

            step = step_size
            accepted = False
            old_col = W[:, v].copy()

            while True:
                new_col = mirror_descent_simplex_step(old_col, g_v, step, eps=eps)
                W[:, v] = new_col
                candidate_obj = objective_W_original(W, A, X, gamma_obj)

                if (not use_backtracking) or (candidate_obj <= current_obj):
                    accepted = True
                    current_obj = candidate_obj
                    break

                W[:, v] = old_col
                step *= backtracking_beta
                if step < min_step_size:
                    break

            if not accepted:
                W[:, v] = old_col

        rel_change = np.linalg.norm(W - W_old) / max(np.linalg.norm(W_old), 1e-12)

        if verbose and (epoch % 5 == 0 or epoch == max_epochs - 1):
            print(
                f"[MD]  epoch={epoch:4d}, obj={current_obj:.8e}, rel_change={rel_change:.3e}"
            )

        if rel_change < tol:
            break

    return W


# ============================================================
# 4) Frank-Wolfe on simplex columns
# ============================================================


def update_W_frank_wolfe(
    W_init: np.ndarray,
    A: np.ndarray,
    X: np.ndarray,
    gamma_obj: float,
    max_epochs: int = 50,
    tol: float = 1e-6,
    eps: float = 1e-12,
    line_search_grid: int = 21,
    randomized: bool = False,
    seed: int | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Frank-Wolfe updating one simplex column at a time.

    For column v:
      s_v = e_j where j = argmin_i g_i
      direction d = s_v - w_v
      w_v <- (1-alpha) w_v + alpha s_v

    alpha is chosen by grid line search on the full objective.
    """
    rng = np.random.default_rng(seed)

    # W = np.asarray(W_init, dtype=float).copy()
    # A = np.asarray(A, dtype=float)
    # X = np.asarray(X, dtype=float)
    W = W_init

    k, n = W.shape
    B = A @ A.T

    # Vectorized initial projection
    # W = project_columns_to_simplex(W)

    # current_obj = objective_W_original(W, A, X, gamma_obj)

    for epoch in range(max_epochs):
        # W_old = W.copy()

        order = np.arange(n)
        if randomized:
            rng.shuffle(order)

        for v in order:
            g_v = compute_single_column_subgradient_original(
                W=W, A=A, X=X, gamma_obj=gamma_obj, v=v, B=B, eps=eps
            )

            j = int(np.argmin(g_v))
            s_v = np.zeros(k, dtype=float)
            s_v[j] = 1.0

            old_col = W[:, v]
            alpha = 2 / (epoch + 2)  # step size for FW, can also do line search here
            direction = s_v - old_col

            # # Vectorized grid line search on alpha in [0,1]
            # alphas = np.linspace(0.0, 1.0, line_search_grid)
            # objs = np.zeros(line_search_grid)

            # for i, alpha in enumerate(alphas):
            #     W[:, v] = old_col + alpha * direction
            #     objs[i] = objective_W_original(W, A, X, gamma_obj)

            # best_idx = np.argmin(objs)
            # best_alpha = float(alphas[best_idx])
            # best_obj = float(objs[best_idx])
            new_col = old_col + alpha * direction
            W[:, v] = new_col
            # current_obj = best_obj

        # rel_change = np.linalg.norm(W - W_old) / max(np.linalg.norm(W_old), 1e-12)

        # if verbose and (epoch % 5 == 0 or epoch == max_epochs - 1):
        #     print(
        #         f"[FW]  epoch={epoch:4d}, obj={current_obj:.8e}, rel_change={rel_change:.3e}"
        #     )

        # if rel_change < tol:
        #     break

    return W


# def update_W_frank_wolfe(
#     W_init: np.ndarray,
#     A: np.ndarray,
#     X: np.ndarray,
#     gamma_obj: float,
#     max_epochs: int = 50,
#     tol: float = 1e-6,
#     eps: float = 1e-12,
#     line_search_grid: int = 21,
#     randomized: bool = False,
#     seed: int | None = None,
#     verbose: bool = False,
# ) -> np.ndarray:
#     """
#     Frank-Wolfe updating one simplex column at a time.

#     For column v:
#       s_v = e_j where j = argmin_i g_i
#       direction d = s_v - w_v
#       w_v <- (1-alpha) w_v + alpha s_v

#     alpha is chosen by grid line search on the full objective.
#     """
#     rng = np.random.default_rng(seed)

#     W = np.asarray(W_init, dtype=float).copy()
#     A = np.asarray(A, dtype=float)
#     X = np.asarray(X, dtype=float)

#     k, n = W.shape
#     B = A @ A.T

#     # Vectorized initial projection
#     W = project_columns_to_simplex(W)

#     current_obj = objective_W_original(W, A, X, gamma_obj)

#     for epoch in range(max_epochs):
#         W_old = W.copy()

#         order = np.arange(n)
#         if randomized:
#             rng.shuffle(order)

#         for v in order:
#             g_v = compute_single_column_subgradient_original(
#                 W=W, A=A, X=X, gamma_obj=gamma_obj, v=v, B=B, eps=eps
#             )

#             j = int(np.argmin(g_v))
#             s_v = np.zeros(k, dtype=float)
#             s_v[j] = 1.0

#             old_col = W[:, v].copy()
#             direction = s_v - old_col

#             # Vectorized grid line search on alpha in [0,1]
#             alphas = np.linspace(0.0, 1.0, line_search_grid)
#             objs = np.zeros(line_search_grid)

#             for i, alpha in enumerate(alphas):
#                 W[:, v] = old_col + alpha * direction
#                 objs[i] = objective_W_original(W, A, X, gamma_obj)

#             best_idx = np.argmin(objs)
#             best_alpha = float(alphas[best_idx])
#             best_obj = float(objs[best_idx])

#             W[:, v] = old_col + best_alpha * direction
#             current_obj = best_obj

#         rel_change = np.linalg.norm(W - W_old) / max(np.linalg.norm(W_old), 1e-12)

#         if verbose and (epoch % 5 == 0 or epoch == max_epochs - 1):
#             print(
#                 f"[FW]  epoch={epoch:4d}, obj={current_obj:.8e}, rel_change={rel_change:.3e}"
#             )

#         if rel_change < tol:
#             break

#     return W


# ============================================================
# 5) D-reparameterization  (for the coupled graph+feature objective)
#
#   D_iv = sqrt(W_iv),  constraint: ||d_v||_2 = 1, D >= 0
#   L = D^T D  is the Laplacian used in the graph term.
#
#   Gradient w.r.t. D (singularity-free):
#     graph:   4 * mu_graph * D @ (D^T D - A_graph)
#     clust:   2 * mu * max(D, d_floor) * nabla_W_clust
#
#   The graph gradient is a matrix product (D @ R), so it does NOT
#   vanish for small D entries.  The clustering gradient has an
#   element-wise D factor (chain rule: dW/dD = 2D).  A floor on D
#   in that factor prevents dead entries when mu_graph = 0.
#
#   Projection: clamp negatives to 0, normalise to unit norm.
#   Returns W = D^2.
# ============================================================

_D_FLOOR = 1e-2  # floor for the chain-rule factor in clustering gradient


def compute_D_gradient(
    D: np.ndarray,
    A_atoms: np.ndarray,
    X: np.ndarray,
    gamma_obj: float,
    A_graph: np.ndarray,
    mu: float,
    mu_graph: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """Gradient of the coupled objective w.r.t. D.  No 1/sqrt(W) singularity."""
    # Graph term: gradient of -mu_graph * <L, A_bar> w.r.t. D = -2*mu_graph * D @ A_bar
    G_D = -2.0 * mu_graph * (D @ A_graph)

    ## check if the two grads are the same
    # print("G_D vs G_D_new diff:", np.linalg.norm(G_D - G_D_new))

    # Clustering term: 2 * mu * max(D, floor) * nabla_W_clust
    if mu != 0.0:
        W = D * D  # recover W = D^2
        G_W_clust = compute_W_subgradient_original_fast(
            W,
            A_atoms,
            X,
            gamma_obj=gamma_obj,
            eps=eps,
        )
        # D_safe = np.maximum(D, _D_FLOOR)
        # D_safe = D
        G_D += 2.0 * mu * D * G_W_clust

    return G_D


def _project_columns_to_nonneg_sphere(D: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Project each column of D onto {d >= 0, ||d||_2 = 1}."""
    D = np.maximum(D, 0.0)
    norms = np.linalg.norm(D, axis=0, keepdims=True)
    return D / np.maximum(norms, eps)


def update_W_D_reparam(
    W_init: np.ndarray,
    A_atoms: np.ndarray,
    X: np.ndarray,
    gamma_obj: float,
    A_graph: np.ndarray,
    mu: float = 1.0,
    mu_graph: float = 0.0,
    step_size: float = 1e-3,
    max_iter: int = 200,
    tol: float = 1e-6,
    eps: float = 1e-12,
    verbose: bool = False,
) -> np.ndarray:
    """
    Projected gradient descent over D on the non-negative unit sphere.

    D_iv = sqrt(W_iv).  Constraint per column: ||d_v||_2 = 1, D >= 0.

    Returns W = D^2  (columns on the simplex).
    """
    # D = np.sqrt(np.maximum(W_init, eps)).copy()
    D = np.sqrt(W_init)

    for it in range(max_iter):
        G_D = compute_D_gradient(
            D,
            A_atoms,
            X,
            gamma_obj,
            A_graph,
            mu,
            mu_graph,
            eps=eps,
        )

        # step_size = step_size / np.sqrt(it + 1)  # decaying step size
        F = D - step_size * G_D
        # D_new = F / np.linalg.norm(F, axis=0, keepdims=True)  # project to unit sphere
        D_new = _project_columns_to_nonneg_sphere(F, eps=eps)

        rel_change = np.linalg.norm(D_new - D) / max(np.linalg.norm(D), 1e-12)
        D = D_new

        if verbose and (it % 25 == 0 or it == max_iter - 1):
            G_inf = float(np.max(np.abs(G_D)))
            print(
                f"[W-D]  iter={it:4d}, step={step_size:.3e}, "
                f"|G_D|_inf={G_inf:.2f}, rel_change={rel_change:.3e}"
            )

        if rel_change < tol:
            break

    return D * D  # W = D^2


def update_W_D_reparam_SPG(
    W_init: np.ndarray,
    A_atoms: np.ndarray,
    X: np.ndarray,
    gamma_obj: float,
    A_graph: np.ndarray,
    mu: float = 1.0,
    mu_graph: float = 0.0,
    step_size: float = 1e-3,
    max_iter: int = 200,
    tol: float = 1e-6,
    eps: float = 1e-12,
    verbose: bool = False,
    step_min: float = 1e-10,
    step_max: float = 1e3,
) -> np.ndarray:
    """
    SPG-style projected gradient method over D on the non-negative unit sphere.

    D_iv = sqrt(W_iv). Constraint per column: ||d_v||_2 = 1, D >= 0.

    Returns W = D^2  (columns on the simplex).

    Notes
    -----
    - This version assumes compute_D_gradient(...) returns the true gradient.
    - It uses Barzilai-Borwein spectral step sizes.
    - It does not use an explicit line search, so it is a lightweight SPG variant.
    """

    D = np.sqrt(W_init)
    G_D = compute_D_gradient(
        D,
        A_atoms,
        X,
        gamma_obj,
        A_graph,
        mu,
        mu_graph,
        eps=eps,
    )

    alpha = step_size

    for it in range(max_iter):
        # Projected spectral step
        F = D - alpha * G_D
        D_new = _project_columns_to_nonneg_sphere(F, eps=eps)

        rel_change = np.linalg.norm(D_new - D) / max(np.linalg.norm(D), 1e-12)

        # New gradient
        G_D_new = compute_D_gradient(
            D_new,
            A_atoms,
            X,
            gamma_obj,
            A_graph,
            mu,
            mu_graph,
            eps=eps,
        )

        # BB spectral step update
        S = D_new - D
        Y = G_D_new - G_D

        sy = np.sum(S * Y)
        ss = np.sum(S * S)

        if sy > eps and ss > eps:
            alpha = ss / sy
            alpha = float(np.clip(alpha, step_min, step_max))
        else:
            alpha = step_size

        D = cp.deepcopy(D_new)
        G_D = cp.deepcopy(G_D_new)

        if verbose and (it % 25 == 0 or it == max_iter - 1):
            G_inf = float(np.max(np.abs(G_D)))
            print(
                f"[W-D-SPG] iter={it:4d}, alpha={alpha:.3e}, "
                f"|G_D|_inf={G_inf:.2f}, rel_change={rel_change:.3e}"
            )

        if rel_change < tol:
            break

    return D * D


def update_W_D_reparam_FW(
    W_init: np.ndarray,
    A_atoms: np.ndarray,
    X: np.ndarray,
    gamma_obj: float,
    A_graph: np.ndarray,
    mu: float = 1.0,
    mu_graph: float = 0.0,
    max_iter: int = 200,
    tol: float = 1e-6,
    eps: float = 1e-12,
    verbose: bool = False,
    step_rule: str = "open_loop",
    step_size: float = 1.0,
) -> np.ndarray:
    """
    Frank-Wolfe over D with column-wise feasible set

        d_v >= 0, ||d_v||_2 <= 1.

    D_iv = sqrt(W_iv).

    Returns
    -------
    W = D^2

    Notes
    -----
    1. Frank-Wolfe requires a convex feasible set, so this solves the relaxed
       problem with ||d_v||_2 <= 1 rather than ||d_v||_2 = 1.

    2. step_rule:
       - "open_loop": eta_t = 2 / (t + 2)
       - "fixed":     eta_t = step_size
    """
    D = np.sqrt(np.maximum(W_init, 0.0)).copy()

    # Optional: if you want the initial point projected to the ball, do this
    col_norms = np.linalg.norm(D, axis=0, keepdims=True)
    D = D / np.maximum(1.0, col_norms)

    for it in range(max_iter):
        G_D = compute_D_gradient(
            D,
            A_atoms,
            X,
            gamma_obj,
            A_graph,
            mu,
            mu_graph,
            eps=eps,
        )

        # LMO: minimize <G_D, S> over feasible set
        S = _fw_lmo_columns_nonneg_unit_ball(G_D, eps=eps)

        # FW direction
        Delta = S - D

        # Relative change proxy
        delta_norm = np.linalg.norm(Delta)
        rel_change = delta_norm / max(np.linalg.norm(D), 1e-12)

        if verbose and (it % 25 == 0 or it == max_iter - 1):
            G_inf = float(np.max(np.abs(G_D)))
            dual_gap = float(-np.sum(G_D * Delta))  # standard FW gap
            print(
                f"[W-D-FW] iter={it:4d}, "
                f"|G_D|_inf={G_inf:.2f}, "
                f"gap={dual_gap:.3e}, "
                f"rel_change={rel_change:.3e}"
            )

        # Stopping criterion
        dual_gap = float(-np.sum(G_D * Delta))
        if dual_gap < tol and it > 10:
            break

        # Step size
        if step_rule == "open_loop":
            eta = 2.0 / (it + 2.0)
        elif step_rule == "fixed":
            eta = float(step_size)
        else:
            raise ValueError("step_rule must be 'open_loop' or 'fixed'")

        eta = min(max(eta, 0.0), 1.0)

        # FW update
        D = (1.0 - eta) * D + eta * S

    return D * D


# ============================================================
# Unified chooser
# ============================================================


def update_W(
    W_init: np.ndarray,
    A: np.ndarray,
    X: np.ndarray,
    gamma_obj: float,
    method: str = "d",
    step_size: float = 1e-2,
    max_iter: int = 50,
    tol: float = 1e-6,
    eps: float = 1e-12,
    randomized: bool = False,
    seed: int | None = None,
    verbose: bool = False,
    # --- new optional params for the coupled objective ---
    A_graph: np.ndarray | None = None,
    mu: float = 1.0,
    mu_graph: float = 0.0,
    **kwargs,
) -> np.ndarray:
    """
    Unified interface for W update methods.

    method options:
      - "pg"   : projected gradient / subgradient
      - "bcd"  : block coordinate descent
      - "md"   : mirror descent (column-wise)
      - "fmd"  : mirror descent (full-matrix)
      - "fw"   : Frank-Wolfe
      - "d"    : D-reparameterization (for coupled graph+feature objective)
      - "d-spg": D-reparameterization with spectral projected gradient
    """
    method = method.lower()

    if method == "pg":
        return update_W_projected_gradient_original(
            W_init=W_init,
            A=A,
            X=X,
            gamma_obj=gamma_obj,
            step_size=step_size,
            max_iter=max_iter,
            tol=tol,
            eps=eps,
            verbose=verbose,
            **kwargs,
        )

    if method == "bcd":
        return update_W_block_coordinate_descent(
            W_init=W_init,
            A=A,
            X=X,
            gamma_obj=gamma_obj,
            step_size=step_size,
            max_epochs=max_iter,
            tol=tol,
            eps=eps,
            randomized=randomized,
            seed=seed,
            verbose=verbose,
            **kwargs,
        )

    if method == "md":
        return update_W_mirror_descent(
            W_init=W_init,
            A=A,
            X=X,
            gamma_obj=gamma_obj,
            step_size=step_size,
            max_epochs=max_iter,
            tol=tol,
            eps=eps,
            randomized=randomized,
            seed=seed,
            verbose=verbose,
            **kwargs,
        )
    if method == "fmd":
        return update_W_mirror_descent_full(
            W_init=W_init,
            A=A,
            X=X,
            gamma_obj=gamma_obj,
            step_size=step_size,
            max_iter=max_iter,
            tol=tol,
            eps=eps,
            # randomized=randomized,
            # seed=seed,
            verbose=verbose,
            **kwargs,
        )

    if method == "fw":
        return update_W_frank_wolfe(
            W_init=W_init,
            A=A,
            X=X,
            gamma_obj=gamma_obj,
            max_epochs=max_iter,
            tol=tol,
            eps=eps,
            randomized=randomized,
            seed=seed,
            verbose=verbose,
            **kwargs,
        )

    if method == "d":
        if A_graph is None:
            raise ValueError("method='d' requires A_graph to be provided.")
        return update_W_D_reparam(
            W_init=W_init,
            A_atoms=A,
            X=X,
            gamma_obj=gamma_obj,
            A_graph=A_graph,
            mu=mu,
            mu_graph=mu_graph,
            step_size=step_size,
            max_iter=max_iter,
            tol=tol,
            eps=eps,
            verbose=verbose,
        )

    if method == "d-spg":
        if A_graph is None:
            raise ValueError("method='d-spg' requires A_graph to be provided.")
        return update_W_D_reparam_SPG(
            W_init=W_init,
            A_atoms=A,
            X=X,
            gamma_obj=gamma_obj,
            A_graph=A_graph,
            mu=mu,
            mu_graph=mu_graph,
            step_size=step_size,
            max_iter=max_iter,
            tol=tol,
            eps=eps,
            verbose=verbose,
        )

    if method == "d-fw":
        if A_graph is None:
            raise ValueError("method='d-fw' requires A_graph to be provided.")
        return update_W_D_reparam_FW(
            W_init=W_init,
            A_atoms=A,
            X=X,
            gamma_obj=gamma_obj,
            A_graph=A_graph,
            mu=mu,
            mu_graph=mu_graph,
            max_iter=max_iter,
            tol=tol,
            eps=eps,
            verbose=verbose,
        )

    raise ValueError(
        f"Unknown method '{method}'. Use one of: pg, bcd, md, fmd, fw, d, d-spg."
    )


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    n = 40  # number of data points
    d = 4  # data dimension
    k = 5  # number of atoms

    X = rng.normal(size=(n, d))
    A = rng.normal(size=(k, d))

    W0 = rng.uniform(size=(k, n))
    W0 /= W0.sum(axis=0, keepdims=True)

    gamma_obj = 0.05

    print("Running BCD...")
    W_bcd = update_W(
        W_init=W0,
        A=A,
        X=X,
        gamma_obj=gamma_obj,
        method="bcd",
        step_size=1e-2,
        max_iter=20,
        randomized=True,
        seed=0,
        verbose=True,
    )
    print("BCD objective:", objective_W_original(W_bcd, A, X, gamma_obj))

    print("\nRunning Mirror Descent...")
    W_md = update_W(
        W_init=W0,
        A=A,
        X=X,
        gamma_obj=gamma_obj,
        method="md",
        step_size=1e-2,
        max_iter=20,
        randomized=True,
        seed=0,
        verbose=True,
    )
    print("MD objective:", objective_W_original(W_md, A, X, gamma_obj))

    print("\nRunning Frank-Wolfe...")
    W_fw = update_W(
        W_init=W0,
        A=A,
        X=X,
        gamma_obj=gamma_obj,
        method="fw",
        max_iter=20,
        randomized=True,
        seed=0,
        verbose=True,
        line_search_grid=15,
    )
    print("FW objective:", objective_W_original(W_fw, A, X, gamma_obj))
