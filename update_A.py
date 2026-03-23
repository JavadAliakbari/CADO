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
    f(A) = ||X - W^T A||_F^2 + gamma * sum_{u<v} ||A^T(w_u - w_v)||_2
    (vectorized)

    A: (k, d)
    W: (k, n)
    X: (n, d)
    """
    A = np.asarray(A, dtype=float)
    W = np.asarray(W, dtype=float)
    X = np.asarray(X, dtype=float)

    M = W.T @ A
    fit = float(np.sum((X - M) ** 2))

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
    Vectorized subgradient wrt A for the original objective.

    A: (k, d), W: (k, n), X: (n, d)
    """
    # A = np.asarray(A, dtype=float)
    # W = np.asarray(W, dtype=float)
    # X = np.asarray(X, dtype=float)

    G = 2.0 * (W @ W.T @ A - W @ X)

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

    if WWt is None:
        WWt = W @ W.T
    if WX is None:
        WX = W @ X

    # smooth part row i
    g = 2.0 * (WWt[i, :] @ A - WX[i, :])

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
    ell: float,
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
    Projected subgradient descent for the original A objective.
    """
    A = A_init
    step_size = 0.001

    for it in range(max_iter):
        G = compute_A_subgradient_original_fast(A, W, X, gamma_obj, eps=eps)

        A_candidate = project_rows_to_l2_ball(A - step_size * G, ell, eps=eps)

        rel_change = np.linalg.norm(A_candidate - A) / max(np.linalg.norm(A), 1e-12)
        A = A_candidate

        if rel_change < tol:
            break

    return A


# ============================================================
# 1b) Projected Adam (momentum-based gradient descent)
# ============================================================


def update_A_projected_adam(
    A_init: np.ndarray,
    W: np.ndarray,
    X: np.ndarray,
    gamma_obj: float,
    ell: float,
    step_size: float = 1e-3,
    max_iter: int = 300,
    tol: float = 1e-6,
    eps: float = 1e-12,
    beta1: float = 0.9,
    beta2: float = 0.999,
    adam_eps: float = 1e-8,
    verbose: bool = False,
) -> np.ndarray:
    """
    Projected Adam optimizer for the original A objective.

    Uses Adam's adaptive learning rate with momentum, then projects
    each row of A onto the l2 ball constraint.

    Parameters
    ----------
    beta1 : float
        Exponential decay rate for first moment estimates (momentum).
    beta2 : float
        Exponential decay rate for second moment estimates (RMSprop-like).
    adam_eps : float
        Small constant for numerical stability in Adam update.
    """
    A = A_init.copy()

    # Initialize first and second moment estimates
    m = np.zeros_like(A)  # First moment (momentum)
    v = np.zeros_like(A)  # Second moment (adaptive learning rate)

    for it in range(max_iter):
        G = compute_A_subgradient_original_fast(A, W, X, gamma_obj, eps=eps)

        # Update biased first moment estimate
        m = beta1 * m + (1 - beta1) * G
        # Update biased second raw moment estimate
        v = beta2 * v + (1 - beta2) * (G * G)

        # Compute bias-corrected first moment estimate
        m_hat = m / (1 - beta1 ** (it + 1))
        # Compute bias-corrected second raw moment estimate
        v_hat = v / (1 - beta2 ** (it + 1))

        # Adam update direction
        update = m_hat / (np.sqrt(v_hat) + adam_eps)

        # Gradient step followed by projection
        A_candidate = project_rows_to_l2_ball(A - step_size * update, ell, eps=eps)

        rel_change = np.linalg.norm(A_candidate - A) / max(np.linalg.norm(A), 1e-12)
        A = A_candidate

        if verbose and (it % 25 == 0 or it == max_iter - 1):
            obj = objective_A_original(A, W, X, gamma_obj)
            print(f"[A-Adam] iter={it:4d}, obj={obj:.8e}, rel_change={rel_change:.3e}")

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
    ell: float,
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
    Row-wise block coordinate descent:
        a_i <- Proj_{||a||<=ell}(a_i - eta g_i)
    """
    rng = np.random.default_rng(seed)

    A = np.asarray(A_init, dtype=float).copy()
    W = np.asarray(W, dtype=float)
    X = np.asarray(X, dtype=float)

    k, _ = A.shape
    A = project_rows_to_l2_ball(A, ell, eps=eps)
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
                new_row = project_to_l2_ball(old_row - step * g_i, ell, eps=eps)
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
    ell: float,
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
    FISTA-style projected subgradient method.
    Works on the original nonsmooth objective, so this is a practical
    accelerated heuristic rather than textbook smooth FISTA.
    """
    # A = project_rows_to_l2_ball(np.asarray(A_init, dtype=float).copy(), ell, eps=eps)
    A = A_init
    Y = A.copy()
    t = 1.0
    step_size = 0.001

    current_obj = objective_A_original(A, W, X, gamma_obj)

    for it in range(max_iter):
        G = compute_A_subgradient_original_fast(Y, W, X, gamma_obj, eps=eps)

        # step = step_size
        # accepted = False

        # while True:
        A_candidate = project_rows_to_l2_ball(Y - step_size * G, ell, eps=eps)
        candidate_obj = objective_A_original(A_candidate, W, X, gamma_obj)

        #     if (not use_backtracking) or (candidate_obj <= current_obj):
        #         accepted = True
        #         break

        #     step *= backtracking_beta
        #     if step < min_step_size:
        #         break

        # if not accepted:
        #     if verbose:
        #         print(f"[A-FISTA] stop at iter {it}: step too small.")
        #     break

        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        Y_new = A_candidate + ((t - 1.0) / t_new) * (A_candidate - A)

        if restart:
            # objective-based adaptive restart
            y_obj = objective_A_original(
                project_rows_to_l2_ball(Y_new, ell, eps=eps), W, X, gamma_obj
            )
            if y_obj > candidate_obj:
                t_new = 1.0
                Y_new = A_candidate.copy()

        rel_change = np.linalg.norm(A_candidate - A) / max(np.linalg.norm(A), 1e-12)
        A = A_candidate
        Y = Y_new
        t = t_new
        # current_obj = candidate_obj

        # if verbose and (it % 25 == 0 or it == max_iter - 1):
        #     print(
        #         f"[A-FISTA] iter={it:4d}, obj={current_obj:.8e}, step={step:.3e}, rel_change={rel_change:.3e}"
        #     )

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
    ell: float,
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
    Projected diagonal quasi-Newton / AdaGrad-like update.

    Uses a diagonal inverse-Hessian approximation:
        H_t^{-1} ~ 1 / sqrt(sum_{s<=t} g_s^2 + eps)
    then projects rows of A onto l2 balls.

    This is not exact BFGS, but it is robust and cheap.
    """
    A = project_rows_to_l2_ball(np.asarray(A_init, dtype=float).copy(), ell, eps=eps)
    current_obj = objective_A_original(A, W, X, gamma_obj)

    G_accum = np.zeros_like(A)

    for it in range(max_iter):
        G = compute_A_subgradient_original_fast(A, W, X, gamma_obj, eps=eps)
        G_accum += G * G

        scaled_grad = G / (np.sqrt(G_accum) + eps)

        step = step_size
        accepted = False

        while True:
            A_candidate = project_rows_to_l2_ball(A - step * scaled_grad, ell, eps=eps)
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
# Unified chooser
# ============================================================


def update_A(
    A_init: np.ndarray,
    W: np.ndarray,
    X: np.ndarray,
    gamma_obj: float,
    ell: float,
    method: str = "bcd",
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
    Unified interface for A update methods.

    method options:
      - "pg"    : projected gradient / subgradient
      - "bcd"   : block coordinate descent by rows
      - "fista" : accelerated projected method
      - "qn"    : diagonal quasi-Newton / AdaGrad-like
    """
    method = method.lower()

    if method == "pg":
        return update_A_projected_gradient(
            A_init=A_init,
            W=W,
            X=X,
            gamma_obj=gamma_obj,
            ell=ell,
            step_size=step_size,
            max_iter=max_iter,
            tol=tol,
            eps=eps,
            verbose=verbose,
            **kwargs,
        )

    if method == "pg_adam":
        return update_A_projected_adam(
            A_init=A_init,
            W=W,
            X=X,
            gamma_obj=gamma_obj,
            ell=ell,
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
            ell=ell,
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
            ell=ell,
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
            ell=ell,
            step_size=step_size,
            max_iter=max_iter,
            tol=tol,
            eps=eps,
            verbose=verbose,
            **kwargs,
        )

    raise ValueError(
        "Unknown method. Use one of: 'pg', 'pg_adam', 'bcd', 'fista', 'qn'."
    )


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
    ell = 1.0

    print("Running A-BCD...")
    A_bcd = update_A(
        A_init=A0,
        W=W,
        X=X,
        gamma_obj=gamma_obj,
        ell=ell,
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
        ell=ell,
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
        ell=ell,
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
        ell=ell,
        method="pg",
        step_size=1e-3,
        max_iter=50,
        verbose=True,
    )
    print("PG objective:", objective_A_original(A_pg, W, X, gamma_obj))
