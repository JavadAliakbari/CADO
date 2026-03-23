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





# ============================================================
# Stochastic Block Model graph generation
# ============================================================


def generate_sbm_graph(
    labels: np.ndarray,
    p: float,
    q: float,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a Stochastic Block Model (SBM) undirected graph.

    Nodes in the same cluster are connected with probability p (intra-cluster).
    Nodes in different clusters are connected with probability q (inter-cluster).

    Parameters
    ----------
    labels : (n,) integer array of cluster labels (0-indexed)
    p      : intra-cluster edge probability  (p > q for assortative structure)
    q      : inter-cluster edge probability
    seed   : random seed for reproducibility

    Returns
    -------
    A     : (n, n)  binary symmetric adjacency matrix, no self-loops
    A_bar : (n, n)  signed affinity matrix —  A_bar[i,j] = +1 if A[i,j]=1,
                    A_bar[i,j] = -1 if A[i,j]=0 (and i != j),
                    diagonal entries are 0.
    """
    rng    = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=int)
    n      = len(labels)

    # Probability matrix: p on diagonal blocks, q elsewhere
    same_cluster = (labels[:, np.newaxis] == labels[np.newaxis, :])  # (n, n) bool
    prob = np.where(same_cluster, float(p), float(q))

    # Sample upper-triangular edges, then symmetrise
    U   = rng.random((n, n))
    raw = (U < prob).astype(float)
    A   = np.triu(raw, k=1)
    A   = A + A.T          # symmetric
    np.fill_diagonal(A, 1.0)   # self-loops = 1  (matches L_{vv} = 1)

    # Signed affinity: 0-entries off-diagonal become -1, diagonal stays +1
    A_bar = np.where(A > 0.5, 1.0, -1.0)

    return A, A_bar


# ============================================================
# Low-rank matrix approximation
# ============================================================


def low_rank_approx(A: np.ndarray, K: int) -> np.ndarray:
    """
    Rank-K approximation of a symmetric matrix A via truncated eigendecomposition.

        A_K = sum_{i=1}^{K}  lambda_i  u_i u_i^T

    where (lambda_i, u_i) are the K largest-magnitude eigenpairs of A.
    Negative eigenvalues are included if they are among the top-K by magnitude,
    so the approximation captures the dominant spectral structure faithfully.

    Parameters
    ----------
    A : (n, n)  symmetric matrix  (adjacency, covariance, etc.)
    K : number of eigenpairs to retain

    Returns
    -------
    A_K : (n, n)  rank-K approximation of A
    """
    A  = np.asarray(A, dtype=float)
    n  = A.shape[0]
    K  = min(int(K), n)

    eigvals, eigvecs = np.linalg.eigh(A)           # ascending order, real

    # Take the K eigenpairs with the largest eigenvalues (top-K, not top-K by |.|)
    # For an adjacency / PSD-like matrix the leading positive eigenvalues carry
    # the cluster structure; sorting descending and taking the first K is correct.
    idx    = np.argsort(eigvals)[::-1][:K]          # indices of K largest
    lam_K  = eigvals[idx]                           # (K,)
    U_K    = eigvecs[:, idx]                        # (n, K)

    return (U_K * lam_K) @ U_K.T                   # sum_i lam_i u_i u_i^T
