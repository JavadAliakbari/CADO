import numpy as np


def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """
    Euclidean projection of a vector v onto the probability simplex:
        {w : w >= 0, sum(w) = 1}

    Reference method:
    Wang & Carreira-Perpinan, 2013
    """
    v = np.asarray(v, dtype=float).ravel()
    n = v.size

    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1)

    cond = u - cssv / ind > 0
    if not np.any(cond):
        # Fallback, though this should not normally happen
        theta = cssv[-1] / n
    else:
        rho = ind[cond][-1]
        theta = cssv[rho - 1] / rho

    w = np.maximum(v - theta, 0.0)
    return w


def compute_W_gradient(
    W: np.ndarray,
    A: np.ndarray,
    X: np.ndarray,
    B: np.ndarray,
    Lc: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """
    Gradient of
        ||X - W^T A||_F^2 + (gamma/2) tr(B W Lc W^T)

    Shapes:
        W  : (m, n)
        A  : (m, d)
        X  : (n, d)
        B  : (m, m)
        Lc : (n, n)
    """
    return 2.0 * (B @ W - A @ X.T) + gamma * (B @ W @ Lc)


def objective_W(
    W: np.ndarray,
    A: np.ndarray,
    X: np.ndarray,
    B: np.ndarray,
    Lc: np.ndarray,
    gamma: float,
) -> float:
    """
    Evaluate the quadratic MM objective:
        ||X - W^T A||_F^2 + (gamma/2) tr(B W Lc W^T)
    """
    residual = X - W.T @ A
    fit = np.sum(residual**2)
    reg = 0.5 * gamma * np.trace(B @ W @ Lc @ W.T)
    return float(fit + reg)


def update_W_projected_gradient(
    W_init: np.ndarray,
    A: np.ndarray,
    X: np.ndarray,
    Lc: np.ndarray,
    gamma: float,
    step_size: float = 1e-3,
    max_iter: int = 1000,
    tol: float = 1e-6,
    use_backtracking: bool = True,
    backtracking_beta: float = 0.5,
    min_step_size: float = 1e-12,
    verbose: bool = False,
) -> np.ndarray:
    """
    Update W using projected gradient descent for the problem

        min_W ||X - W^T A||_F^2 + (gamma/2) tr(B W Lc W^T)
        s.t. each column w_v lies in the simplex:
             w_v >= 0, 1^T w_v = 1

    Parameters
    ----------
    W_init : (m, n) ndarray
        Initial matrix W.
    A : (m, d) ndarray
        Atom matrix, each row is a_i^T.
    X : (n, d) ndarray
        Data matrix, each row is x_v^T.
    Lc : (n, n) ndarray
        Weighted Laplacian matrix.
    gamma : float
        Regularization strength.
    step_size : float
        Initial gradient step size.
    max_iter : int
        Maximum number of projected gradient iterations.
    tol : float
        Stopping threshold based on relative change in W.
    use_backtracking : bool
        Whether to use monotone backtracking line search.
    backtracking_beta : float
        Multiplicative shrinkage for backtracking.
    min_step_size : float
        Minimum allowed step size during backtracking.
    verbose : bool
        Print progress if True.

    Returns
    -------
    W : (m, n) ndarray
        Updated matrix W whose columns are on the simplex.
    """
    W = np.asarray(W_init, dtype=float).copy()
    A = np.asarray(A, dtype=float)
    X = np.asarray(X, dtype=float)
    Lc = np.asarray(Lc, dtype=float)

    m, n = W.shape
    if A.shape[0] != m:
        raise ValueError("A must have shape (m, d) where m = W.shape[0].")
    if X.shape[0] != n:
        raise ValueError("X must have shape (n, d) where n = W.shape[1].")
    if Lc.shape != (n, n):
        raise ValueError("Lc must have shape (n, n).")

    # B = A A^T
    B = A @ A.T

    # Project initial columns to simplex to ensure feasibility
    for v in range(n):
        W[:, v] = project_to_simplex(W[:, v])

    current_obj = objective_W(W, A, X, B, Lc, gamma)

    for it in range(max_iter):
        grad = compute_W_gradient(W, A, X, B, Lc, gamma)

        step = step_size
        accepted = False

        while True:
            W_candidate = W - step * grad

            # Project each column onto simplex
            for v in range(n):
                W_candidate[:, v] = project_to_simplex(W_candidate[:, v])

            candidate_obj = objective_W(W_candidate, A, X, B, Lc, gamma)

            if not use_backtracking or candidate_obj <= current_obj:
                accepted = True
                break

            step *= backtracking_beta
            if step < min_step_size:
                break

        if not accepted:
            if verbose:
                print(f"Stopped at iter {it}: step size became too small.")
            break

        rel_change = np.linalg.norm(W_candidate - W) / max(np.linalg.norm(W), 1e-12)
        W = W_candidate
        obj_improvement = current_obj - candidate_obj
        current_obj = candidate_obj

        if verbose and (it % 50 == 0 or it == max_iter - 1):
            print(
                f"iter={it:4d}, obj={current_obj:.8e}, "
                f"step={step:.3e}, rel_change={rel_change:.3e}, "
                f"improvement={obj_improvement:.3e}"
            )

        if rel_change < tol:
            if verbose:
                print(f"Converged at iter {it}.")
            break

    return W


if __name__ == "__main__":
    # Example usage
    rng = np.random.default_rng(0)

    n = 20  # number of data points
    d = 5  # feature dimension
    m = 4  # number of active atoms

    X = rng.normal(size=(n, d))
    A = rng.normal(size=(m, d))

    # Build a simple symmetric nonnegative weight matrix, then Laplacian
    C = rng.uniform(0.0, 1.0, size=(n, n))
    C = 0.5 * (C + C.T)
    np.fill_diagonal(C, 0.0)
    D = np.diag(C.sum(axis=1))
    Lc = D - C

    gamma = 0.1

    # Random feasible initialization: each column on simplex
    W0 = rng.uniform(size=(m, n))
    W0 /= W0.sum(axis=0, keepdims=True)

    W_updated = update_W_projected_gradient(
        W_init=W0,
        A=A,
        X=X,
        Lc=Lc,
        gamma=gamma,
        step_size=1e-2,
        max_iter=1000,
        tol=1e-7,
        use_backtracking=True,
        verbose=True,
    )

    print("Column sums:", W_updated.sum(axis=0))
    print("Minimum entry:", W_updated.min())
    print(W_updated)
