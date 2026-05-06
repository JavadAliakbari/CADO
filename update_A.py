import numpy as np

# ============================================================
# Row-wise projection onto l2 balls
# ============================================================


def project_to_l2_ball(a: np.ndarray, ell: float, eps: float = 1e-12) -> np.ndarray:
    a = np.asarray(a, dtype=float).reshape(-1)
    nrm = float(np.linalg.norm(a))
    if nrm <= ell:
        return a
    return a * (ell / max(nrm, eps))


def project_rows_to_l2_ball(
    A: np.ndarray, ell: float, eps: float = 1e-12
) -> np.ndarray:
    """
    Project each row of A onto the l2 ball of radius ell (vectorized).
    """
    A = np.asarray(A, dtype=float).copy()
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    # Scale factor: min(1, ell/norm) for each row
    scale = np.minimum(1.0, ell / np.maximum(norms, eps))
    return A * scale


# ============================================================
# Original A-objective and gradients
# ============================================================


def objective_A_original(
    A: np.ndarray,
    W: np.ndarray,
    X: np.ndarray,
    gamma_obj: float,
) -> float:
    """
    f(A) = (1/2) sum_{i,v} W_iv ||x_v - a_i||^2
           + gamma * sum_{u<v} ||(W^T A)_u - (W^T A)_v||_2
    (eq. 15 weighted k-means formulation)

    A: (k, d)
    W: (k, n)
    X: (n, d)
    """
    A = np.asarray(A, dtype=float)
    W = np.asarray(W, dtype=float)
    X = np.asarray(X, dtype=float)

    # D[i,v] = ||x_v - a_i||^2 = ||a_i||^2 - 2 a_i.x_v + ||x_v||^2
    D = (
        np.sum(A**2, axis=1, keepdims=True)  # (k, 1)
        - 2.0 * (A @ X.T)  # (k, n)
        + np.sum(X**2, axis=1)[np.newaxis, :]  # (1, n)
    )
    fit = 0.5 * float(np.sum(W * D))

    # Vectorized fusion term computation
    _, n = W.shape
    u_idx, v_idx = np.triu_indices(n, k=1)

    # DW[:, p] = w_u - w_v for each pair p
    DW = W[:, u_idx] - W[:, v_idx]  # (k, P)

    # Z[:, p] = A^T @ (w_u - w_v)
    Z = A.T @ DW  # (d, P)

    # Sum of norms
    fusion = float(np.sum(np.linalg.norm(Z, axis=0)))

    return fit + gamma_obj * fusion


def compute_A_subgradient_original_fast(
    A: np.ndarray,
    W: np.ndarray,
    X: np.ndarray,
    gamma_obj: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Vectorized subgradient wrt A for the eq.(15) objective.

    Fit gradient:  diag(lambda) A - W X
    where lambda_i = sum_v W_iv  (row sums of W).

    A: (k, d), W: (k, n), X: (n, d)
    """
    A = np.asarray(A, dtype=float)
    W = np.asarray(W, dtype=float)
    X = np.asarray(X, dtype=float)

    lam = W.sum(axis=1)  # (k,)
    G = lam[:, np.newaxis] * A - W @ X  # diag(lam) @ A - W @ X

    _, n = W.shape
    u_idx, v_idx = np.triu_indices(n, k=1)

    DW = W[:, u_idx] - W[:, v_idx]  # (k, P)
    Z = A.T @ DW  # (d, P)

    norms = np.linalg.norm(Z, axis=0)
    U = np.zeros_like(Z)
    mask = norms > eps
    U[:, mask] = Z[:, mask] / norms[mask][None, :]

    G += gamma_obj * (DW @ U.T)
    return G


def compute_single_row_subgradient_original(
    A: np.ndarray,
    W: np.ndarray,
    X: np.ndarray,
    gamma_obj: float,
    i: int,
    WWt: np.ndarray | None = None,
    WX: np.ndarray | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Subgradient wrt a single row a_i of A (vectorized).

    Returns g_i in R^d.
    """
    A = np.asarray(A, dtype=float)
    W = np.asarray(W, dtype=float)
    X = np.asarray(X, dtype=float)

    k, d = A.shape
    _, n = W.shape

    if WX is None:
        WX = W @ X

    # fit gradient for row i: lambda_i * a_i - (WX)_i
    lam_i = float(W[i, :].sum())
    g = lam_i * A[i, :] - WX[i, :]

    # Vectorized fusion term subgradient
    # Get all pairs (u, v) with u < v
    u_idx, v_idx = np.triu_indices(n, k=1)

    # DW[:, p] = w_u - w_v for each pair p
    DW = W[:, u_idx] - W[:, v_idx]  # (k, P)

    # Z[:, p] = A^T @ (w_u - w_v)
    Z = A.T @ DW  # (d, P)

    # Compute norms
    norms = np.linalg.norm(Z, axis=0)  # (P,)

    # Normalized Z
    U = np.zeros_like(Z)
    mask = norms > eps
    U[:, mask] = Z[:, mask] / norms[mask]

    # For row i, the contribution is: sum_{u<v} dv[i] * (z / ||z||)
    # where dv = w_u - w_v
    # = sum_{u<v} DW[i, p] * U[:, p]
    # This is a weighted sum of normalized directions
    g += gamma_obj * (U @ DW[i, :])

    return g


# ============================================================
# 1) Projected gradient / subgradient
# ============================================================


def update_A_projected_gradient(
    A_init: np.ndarray,
    W: np.ndarray,
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
    """
    Unconstrained subgradient descent for the A objective (quadratic loss).
    """
    A = np.asarray(A_init, dtype=float).copy()
    W = np.asarray(W, dtype=float)
    X = np.asarray(X, dtype=float)

    current_obj = objective_A_original(A, W, X, gamma_obj)

    for it in range(max_iter):
        G = compute_A_subgradient_original_fast(A, W, X, gamma_obj, eps=eps)

        step = step_size
        accepted = False

        while True:
            A_candidate = A - step * G
            candidate_obj = objective_A_original(A_candidate, W, X, gamma_obj)

            if (not use_backtracking) or (candidate_obj <= current_obj):
                accepted = True
                break

            step *= backtracking_beta
            if step < min_step_size:
                break

        if not accepted:
            if verbose:
                print(f"[A-PG] stop at iter {it}: step too small.")
            break

        rel_change = np.linalg.norm(A_candidate - A) / max(np.linalg.norm(A), 1e-12)
        A = A_candidate
        current_obj = candidate_obj

        if verbose and (it % 25 == 0 or it == max_iter - 1):
            print(
                f"[A-PG] iter={it:4d}, obj={current_obj:.8e}, step={step:.3e}, rel_change={rel_change:.3e}"
            )

        if rel_change < tol:
            break

    return A


# ============================================================
# 2) Block coordinate descent by rows of A
# ============================================================


def update_A_block_coordinate_descent(
    A_init: np.ndarray,
    W: np.ndarray,
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
    Row-wise block coordinate descent (unconstrained, quadratic loss):
        a_i <- a_i - eta g_i
    """
    rng = np.random.default_rng(seed)

    A = np.asarray(A_init, dtype=float).copy()
    W = np.asarray(W, dtype=float)
    X = np.asarray(X, dtype=float)

    k, _ = A.shape
    current_obj = objective_A_original(A, W, X, gamma_obj)

    WWt = W @ W.T
    WX = W @ X

    for epoch in range(max_epochs):
        A_old = A.copy()

        order = np.arange(k)
        if randomized:
            rng.shuffle(order)

        for i in order:
            g_i = compute_single_row_subgradient_original(
                A=A, W=W, X=X, gamma_obj=gamma_obj, i=i, WWt=WWt, WX=WX, eps=eps
            )

            step = step_size
            accepted = False
            old_row = A[i, :].copy()

            while True:
                new_row = old_row - step * g_i
                A[i, :] = new_row
                candidate_obj = objective_A_original(A, W, X, gamma_obj)

                if (not use_backtracking) or (candidate_obj <= current_obj):
                    accepted = True
                    current_obj = candidate_obj
                    break

                A[i, :] = old_row
                step *= backtracking_beta
                if step < min_step_size:
                    break

            if not accepted:
                A[i, :] = old_row

        rel_change = np.linalg.norm(A - A_old) / max(np.linalg.norm(A_old), 1e-12)

        if verbose and (epoch % 5 == 0 or epoch == max_epochs - 1):
            print(
                f"[A-BCD] epoch={epoch:4d}, obj={current_obj:.8e}, rel_change={rel_change:.3e}"
            )

        if rel_change < tol:
            break

    return A


# ============================================================
# 3) FISTA
# ============================================================


def update_A_fista(
    A_init: np.ndarray,
    W: np.ndarray,
    X: np.ndarray,
    gamma_obj: float,
    step_size: float = 1e-3,
    max_iter: int = 300,
    tol: float = 1e-6,
    eps: float = 1e-12,
    use_backtracking: bool = True,
    backtracking_beta: float = 0.5,
    min_step_size: float = 1e-14,
    restart: bool = True,
    verbose: bool = False,
) -> np.ndarray:
    """
    FISTA-style subgradient method (unconstrained, quadratic loss).
    Works on the original nonsmooth objective, so this is a practical
    accelerated heuristic rather than textbook smooth FISTA.
    """
    A = np.asarray(A_init, dtype=float).copy()
    Y = A.copy()
    t = 1.0

    current_obj = objective_A_original(A, W, X, gamma_obj)

    for it in range(max_iter):
        G = compute_A_subgradient_original_fast(Y, W, X, gamma_obj, eps=eps)

        step = step_size
        accepted = False

        while True:
            A_candidate = Y - step * G
            candidate_obj = objective_A_original(A_candidate, W, X, gamma_obj)

            if (not use_backtracking) or (candidate_obj <= current_obj):
                accepted = True
                break

            step *= backtracking_beta
            if step < min_step_size:
                break

        if not accepted:
            if verbose:
                print(f"[A-FISTA] stop at iter {it}: step too small.")
            break

        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        Y_new = A_candidate + ((t - 1.0) / t_new) * (A_candidate - A)

        if restart:
            # objective-based adaptive restart
            y_obj = objective_A_original(Y_new, W, X, gamma_obj)
            if y_obj > candidate_obj:
                t_new = 1.0
                Y_new = A_candidate.copy()

        rel_change = np.linalg.norm(A_candidate - A) / max(np.linalg.norm(A), 1e-12)
        A = A_candidate
        Y = Y_new
        t = t_new
        current_obj = candidate_obj

        if verbose and (it % 25 == 0 or it == max_iter - 1):
            print(
                f"[A-FISTA] iter={it:4d}, obj={current_obj:.8e}, step={step:.3e}, rel_change={rel_change:.3e}"
            )

        if rel_change < tol:
            break

    return A


# ============================================================
# 4) Quasi-Newton (diagonal / AdaGrad-like projected method)
# ============================================================


def update_A_quasi_newton(
    A_init: np.ndarray,
    W: np.ndarray,
    X: np.ndarray,
    gamma_obj: float,
    step_size: float = 1e-1,
    max_iter: int = 300,
    tol: float = 1e-6,
    eps: float = 1e-8,
    use_backtracking: bool = True,
    backtracking_beta: float = 0.5,
    min_step_size: float = 1e-14,
    verbose: bool = False,
) -> np.ndarray:
    """
    Diagonal quasi-Newton / AdaGrad-like update (unconstrained, quadratic loss).

    Uses a diagonal inverse-Hessian approximation:
        H_t^{-1} ~ 1 / sqrt(sum_{s<=t} g_s^2 + eps)

    This is not exact BFGS, but it is robust and cheap.
    """
    A = np.asarray(A_init, dtype=float).copy()
    current_obj = objective_A_original(A, W, X, gamma_obj)

    G_accum = np.zeros_like(A)

    for it in range(max_iter):
        G = compute_A_subgradient_original_fast(A, W, X, gamma_obj, eps=eps)
        G_accum += G * G

        scaled_grad = G / (np.sqrt(G_accum) + eps)

        step = step_size
        accepted = False

        while True:
            A_candidate = A - step * scaled_grad
            candidate_obj = objective_A_original(A_candidate, W, X, gamma_obj)

            if (not use_backtracking) or (candidate_obj <= current_obj):
                accepted = True
                break

            step *= backtracking_beta
            if step < min_step_size:
                break

        if not accepted:
            if verbose:
                print(f"[A-QN] stop at iter {it}: step too small.")
            break

        rel_change = np.linalg.norm(A_candidate - A) / max(np.linalg.norm(A), 1e-12)
        A = A_candidate
        current_obj = candidate_obj

        if verbose and (it % 25 == 0 or it == max_iter - 1):
            print(
                f"[A-QN] iter={it:4d}, obj={current_obj:.8e}, step={step:.3e}, rel_change={rel_change:.3e}"
            )

        if rel_change < tol:
            break

    return A


# ============================================================
# 5) Closed-form (Newton / preconditioned) update — quadratic case
# ============================================================


def update_A_closed_form(
    A_init: np.ndarray,
    W: np.ndarray,
    X: np.ndarray,
    gamma_obj: float,
    eps: float = 1e-12,
    max_iter: int = 30,
    tol: float = 1e-10,
    verbose: bool = False,
) -> np.ndarray:
    """
    Exact closed-form A update for the eq.(15) weighted k-means objective.

    Setting grad_A = 0 in the new formulation gives a *diagonal* system
    (one independent equation per atom i):

        lambda_i * a_i = (WX)_i - gamma_obj * (W @ G_SON(M))_i

        a_i^* = [(WX)_i - gamma_obj * (W @ G_SON)_i] / lambda_i

    where lambda_i = sum_v W_iv (row sum of W).

    Since G_SON depends on M = W^T A, we iterate a few steps:

        A^{t+1} = diag(lambda)^{-1} [WX - gamma_obj * W @ G_SON(W^T A^t)]

    Warm start: A_0 = diag(lambda)^{-1} WX  (pure weighted centroids).

    Parameters
    ----------
    A_init    : ignored (warm start from centroids is always better)
    W         : (k, n) ndarray
    X         : (n, d) ndarray
    gamma_obj : float
    eps       : float  — floor on lambda_i to avoid division by zero
    max_iter, tol, verbose : convergence controls
    """
    W = np.asarray(W, dtype=float)
    X = np.asarray(X, dtype=float)

    lam = W.sum(axis=1)  # (k,)
    lam_safe = np.maximum(lam, eps)[:, np.newaxis]  # (k, 1)
    WX = W @ X  # (k, d)

    # Warm start: pure weighted centroids (ignores SON)
    A = WX / lam_safe  # (k, d)

    # Precompute pair indices and DW (unchanged across iterations)
    _, n = W.shape
    u_idx, v_idx = np.triu_indices(n, k=1)
    DW = W[:, u_idx] - W[:, v_idx]  # (k, P)

    for it in range(max_iter):
        # SON subgradient in prediction space: G_SON w.r.t. M = W^T A
        M = W.T @ A  # (n, d)
        Z = A.T @ DW  # (d, P)
        norms = np.linalg.norm(Z, axis=0)  # (P,)
        U = np.zeros_like(Z)
        mask = norms > 1e-12
        U[:, mask] = Z[:, mask] / norms[mask]
        # W @ G_SON(M) via chain rule: W @ (DW^T @ U)^T ... = DW @ U^T
        son_correction = DW @ U.T  # (k, d)

        rhs = WX - gamma_obj * son_correction  # (k, d)
        A_new = rhs / lam_safe  # (k, d)

        rel_change = np.linalg.norm(A_new - A) / max(np.linalg.norm(A), 1e-12)
        A = A_new

        if verbose and it % 5 == 0:
            obj = objective_A_original(A, W, X, gamma_obj)
            print(f"[A-CF] iter={it:3d}, obj={obj:.6e}, rel_change={rel_change:.3e}")

        if rel_change < tol:
            break

    return A


# ============================================================
# Canonical-projection utility
# ============================================================


def canonicalize_atoms(
    A: np.ndarray,
    W: np.ndarray,
    eps: float = 1e-4,
) -> np.ndarray:
    """
    Project A to its minimum-norm representative.

    The objective f(W, A) only depends on M = W^T A, not on A itself.
    Many (W, A) pairs achieve the same M, so atoms can drift to arbitrary
    positions during optimisation.

    This function computes the *canonical* (minimum-norm) A that achieves
    the same predictions M = W^T A:

        A_canonical = (W W^T + eps_scaled I)^{-1}  W M
                    = (W W^T + eps_scaled I)^{-1}  W W^T A

    For well-conditioned W (all atoms have substantial weight), this
    places each atom at the weighted mean of its assigned nodes — the
    correct class-centroid position.  It does NOT change M (up to the
    tiny regularisation shift), so the objective is unchanged.

    Parameters
    ----------
    A : (k, d) ndarray  — current (possibly degenerate) atoms
    W : (k, n) ndarray  — assignment matrix
    eps : float         — relative Tikhonov regularisation
    """
    A = np.asarray(A, dtype=float)
    W = np.asarray(W, dtype=float)

    WWT = W @ W.T
    eps_scaled = eps * max(float(np.max(np.diag(WWT))), 1.0)
    WWT_reg = WWT + eps_scaled * np.eye(WWT.shape[0])

    M = W.T @ A  # n x d  — current predictions
    rhs = W @ M  # k x d  = W W^T A
    return np.linalg.solve(WWT_reg, rhs)


# ============================================================
# Unified chooser
# ============================================================


def update_A(
    A_init: np.ndarray,
    W: np.ndarray,
    X: np.ndarray,
    gamma_obj: float,
    method: str = "cf",
    step_size: float = 1e-2,
    max_iter: int = 50,
    tol: float = 1e-6,
    eps: float = 1e-12,
    randomized: bool = False,
    seed: int | None = None,
    verbose: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    Unified interface for A update methods (unconstrained, quadratic loss).

    method options:
      - "cf"   : closed-form Newton/preconditioned step (recommended for
                  unconstrained quadratic loss — gives unique, interpretable
                  atom positions by solving the linear system exactly)
      - "pg"    : gradient / subgradient descent
      - "bcd"   : block coordinate descent by rows
      - "fista" : accelerated subgradient method
      - "qn"    : diagonal quasi-Newton / AdaGrad-like
    """
    method = method.lower()

    if method in ("cf", "closed_form", "closed-form", "newton"):
        return update_A_closed_form(
            A_init=A_init,
            W=W,
            X=X,
            gamma_obj=gamma_obj,
            eps=eps,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
        )

    if method == "pg":
        return update_A_projected_gradient(
            A_init=A_init,
            W=W,
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
        return update_A_block_coordinate_descent(
            A_init=A_init,
            W=W,
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

    if method == "fista":
        return update_A_fista(
            A_init=A_init,
            W=W,
            X=X,
            gamma_obj=gamma_obj,
            step_size=step_size,
            max_iter=max_iter,
            tol=tol,
            eps=eps,
            verbose=verbose,
            **kwargs,
        )

    if method in ("qn", "quasi-newton", "quasinewton"):
        return update_A_quasi_newton(
            A_init=A_init,
            W=W,
            X=X,
            gamma_obj=gamma_obj,
            step_size=step_size,
            max_iter=max_iter,
            tol=tol,
            eps=eps,
            verbose=verbose,
            **kwargs,
        )

    raise ValueError("Unknown method. Use one of: 'cf', 'pg', 'bcd', 'fista', 'qn'.")


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    n = 40
    d = 5
    k = 4

    X = rng.normal(size=(n, d))

    W = rng.uniform(size=(k, n))
    W /= W.sum(axis=0, keepdims=True)

    A0 = rng.normal(size=(k, d))

    gamma_obj = 0.05

    print("Running A-BCD...")
    A_bcd = update_A(
        A_init=A0,
        W=W,
        X=X,
        gamma_obj=gamma_obj,
        method="bcd",
        step_size=1e-2,
        max_iter=20,
        randomized=True,
        seed=0,
        verbose=True,
    )
    print("BCD objective:", objective_A_original(A_bcd, W, X, gamma_obj))

    print("\nRunning A-FISTA...")
    A_fista = update_A(
        A_init=A0,
        W=W,
        X=X,
        gamma_obj=gamma_obj,
        method="fista",
        step_size=1e-3,
        max_iter=50,
        verbose=True,
    )
    print("FISTA objective:", objective_A_original(A_fista, W, X, gamma_obj))

    print("\nRunning A-QN...")
    A_qn = update_A(
        A_init=A0,
        W=W,
        X=X,
        gamma_obj=gamma_obj,
        method="qn",
        step_size=1e-1,
        max_iter=50,
        verbose=True,
    )
    print("QN objective:", objective_A_original(A_qn, W, X, gamma_obj))

    print("\nRunning A-PG...")
    A_pg = update_A(
        A_init=A0,
        W=W,
        X=X,
        gamma_obj=gamma_obj,
        method="pg",
        step_size=1e-3,
        max_iter=50,
        verbose=True,
    )
    print("PG objective:", objective_A_original(A_pg, W, X, gamma_obj))
