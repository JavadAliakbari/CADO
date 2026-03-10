# ============================================================
# Projections
# ============================================================


import numpy as np


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
    # Use argmax on reversed condition to find last True
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


def project_to_l2_ball(a: np.ndarray, ell: float, eps: float = 1e-12) -> np.ndarray:
    a = np.asarray(a, dtype=float).reshape(-1)
    nrm = float(np.linalg.norm(a))
    if nrm <= ell:
        return a
    return a * (ell / max(nrm, eps))
