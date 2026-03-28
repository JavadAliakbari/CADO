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


def compute_p_max(A: np.ndarray, labels: np.ndarray, verbose: bool = True) -> dict:
    """
    Compute the graph misconnection rate p_max from Theorem 4.5 of
    "Bounds on Perfect Node Classification".

    Given the binary adjacency matrix A (with A_{uv} in {0,1}) and true
    cluster labels, computes:

    For each cluster pair (i, j):
      - n_{v,j} = number of edges from node v to cluster j
      - Misconnection count:
            n^+_{v,j} = n_j - n_{v,j}   if v in C_j   (missing within-cluster edges)
            n^+_{v,j} = n_{v,j}          if v not in C_j (spurious between-cluster edges)
      - rho^+_{i,j} = (1/(n_i*n_j)) * sum_{v in C_i} n^+_{v,j}
      - delta = max over all v in C_i, all j of |rho^+_{i,j} - n^+_{v,j}/n_j|

    Then:  p_max = max_{i != j} rho^+_{i,j} + 2*delta

    Recovery bound (graph-structure-only regime, Theorem 4.5 eq. 10):
      p_max <= 1 / (2 * (1 + n/n_i))

    For equal-sized clusters n_i = n/K:
      p_max <= 1 / (2 * (1 + K))

    Parameters
    ----------
    A : (n, n) binary adjacency matrix (0/1), may have 1s on diagonal
    labels : (n,) integer cluster labels

    Returns
    -------
    dict with keys:
      'p_max' : float — the graph misconnection rate
      'rho_plus' : (K, K) array — rho^+_{i,j}
      'delta' : float — max node-level deviation
      'bound_graph_only' : float — the RHS of eq (10): 1/(2(1+n/n_i))
      'recovery_feasible' : bool — whether p_max <= bound
      'K' : int, 'n' : int, 'cluster_sizes' : array
    """
    labels = np.asarray(labels, dtype=int)
    n = len(labels)
    K = int(np.max(labels)) + 1

    # Cluster membership
    clusters = [np.where(labels == i)[0] for i in range(K)]
    cluster_sizes = np.array([len(c) for c in clusters])

    # Zero out diagonal for edge counting (self-loops don't count)
    A_no_diag = A.copy().astype(float)
    np.fill_diagonal(A_no_diag, 0.0)

    # Compute n_{v,j} = number of edges from node v to cluster j
    # Shape: (n, K)
    n_vj = np.zeros((n, K), dtype=float)
    for j in range(K):
        n_vj[:, j] = A_no_diag[:, clusters[j]].sum(axis=1)

    # Compute n^+_{v,j} (misconnection count)
    # n^+_{v,j} = n_j - n_{v,j} if v in C_j, else n_{v,j}
    n_plus_vj = np.zeros((n, K), dtype=float)
    for j in range(K):
        nj = cluster_sizes[j]
        for v in range(n):
            if labels[v] == j:
                # v is in C_j: missing within-cluster edges
                # subtract 1 because v doesn't connect to itself
                n_plus_vj[v, j] = (nj - 1) - n_vj[v, j]
            else:
                # v not in C_j: spurious between-cluster edges
                n_plus_vj[v, j] = n_vj[v, j]

    # Compute rho^+_{i,j} = (1/(n_i * n_j)) * sum_{v in C_i} n^+_{v,j}
    # and adjust denominator for within-cluster (i==j): use n_i*(n_i-1)
    rho_plus = np.zeros((K, K), dtype=float)
    for i in range(K):
        for j in range(K):
            N_plus_ij = float(np.sum(n_plus_vj[clusters[i], j]))
            if i == j:
                denom = cluster_sizes[i] * (cluster_sizes[i] - 1)
            else:
                denom = cluster_sizes[i] * cluster_sizes[j]
            rho_plus[i, j] = N_plus_ij / max(denom, 1)

    # Compute delta = max |rho^+_{i,j} - n^+_{v,j}/n_j|
    delta = 0.0
    for i in range(K):
        for j in range(K):
            if i == j:
                nj_eff = cluster_sizes[j] - 1
            else:
                nj_eff = cluster_sizes[j]
            nj_eff = max(nj_eff, 1)
            for v in clusters[i]:
                local_rate = n_plus_vj[v, j] / nj_eff
                dev = abs(rho_plus[i, j] - local_rate)
                delta = max(delta, dev)

    # p_max = max_{i != j} rho^+_{i,j} + 2*delta
    rho_offdiag = rho_plus.copy()
    np.fill_diagonal(rho_offdiag, -np.inf)
    p_max = float(np.max(rho_offdiag)) + 2 * delta

    # Recovery bound: graph-structure-only (eq 10)
    # p_max <= 1 / (2*(1 + n/n_i))  for all i
    # Use the most restrictive (smallest n_i gives tightest bound)
    n_i_min = int(np.min(cluster_sizes))
    bound_graph_only = 1.0 / (2.0 * (1.0 + n / n_i_min))

    recovery_feasible = (p_max <= bound_graph_only)

    if verbose:
        print(f"=== Graph Misconnection Analysis ===")
        print(f"n = {n},  K = {K},  cluster sizes = {cluster_sizes}")
        print(f"rho^+_{{i,j}} (off-diagonal = between-cluster misconnection):")
        print(np.array2string(rho_plus, precision=4, suppress_small=True))
        print(f"max rho^+_{{i,j}} (i!=j) = {float(np.max(rho_offdiag)):.6f}")
        print(f"delta (max node-level deviation) = {delta:.6f}")
        print(f"p_max = max rho^+ + 2*delta = {p_max:.6f}")
        print(f"")
        print(f"Graph-only recovery bound (Thm 4.5, eq 10):")
        print(f"  p_max <= 1/(2(1 + n/n_i)) = {bound_graph_only:.6f}")
        print(f"  p_max = {p_max:.6f}  {'<=' if recovery_feasible else '>'} "
              f" {bound_graph_only:.6f}")
        print(f"  Perfect recovery: {'YES (condition satisfied)' if recovery_feasible else 'NO (condition violated)'}")

    return {
        'p_max': p_max,
        'rho_plus': rho_plus,
        'delta': delta,
        'bound_graph_only': bound_graph_only,
        'recovery_feasible': recovery_feasible,
        'K': K,
        'n': n,
        'cluster_sizes': cluster_sizes,
    }


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


def _fw_lmo_columns_nonneg_unit_ball(G: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Linear minimization oracle for each column over

        {d >= 0, ||d||_2 <= 1}.

    For each column g, solves

        s = argmin_{d >= 0, ||d||_2 <= 1} <g, d>.

    Solution:
        if some entries of g are negative:
            s = -(-g)_+ / ||(-g)_+||_2
        else:
            s = 0
    """
    H = np.maximum(-G, 0.0)  # positive where G is negative
    norms = np.linalg.norm(H, axis=0, keepdims=True)

    S = np.zeros_like(G)
    mask = norms > eps
    S[:, mask[0]] = H[:, mask[0]] / norms[:, mask[0]]
    return S