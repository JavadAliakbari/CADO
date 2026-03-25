import numpy as np
import matplotlib.pyplot as plt

try:
    import scipy.sparse
    import scipy.sparse.linalg
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    from optimized_solution import (
        build_W_from_labels,
        objective_from_WA,
        solve_A_hard_assignments_cvxpy,
    )
except ImportError:
    pass

# from update_A import update_A_projected_gradient_original
from update_A import update_A

# from update_W import update_W_projected_gradient_original
from update_W import update_W

from cluster_generation import (
    generate_clusters_with_directions,
    random_sparse_directions,
    plot_data,
    plot_solution,
)
from utils import *

# ============================================================
# Alternating W/A refinement helper
# ============================================================


def alternating_WA_refinement(
    W: np.ndarray,
    A: np.ndarray,
    X: np.ndarray,
    gamma_obj: float,
    ell: float,
    num_cycles: int = 1,
    w_step_size: float = 1e-2,
    A_step_size: float = 1e-3,
    max_iter: int = 100,
    tol: float = 1e-7,
    eps: float = 1e-12,
    w_method: str = "fmd",
    a_method: str = "pg",
    use_backtracking=False,
    verbose: bool = False,
    # --- new: graph parameters (defaults recover feature-only behaviour) ---
    A_graph: np.ndarray | None = None,
    mu: float = 1.0,
    mu_graph: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform alternating optimization cycles on W and A.

    Parameters
    ----------
    W : (k, n) ndarray
        Initial assignment matrix.
    A : (k, d) ndarray
        Initial atom matrix.
    X : (n, d) ndarray
        Data matrix.
    gamma_obj : float
        Regularization parameter.
    ell : float
        Radius of l2-ball constraint for rows of A.
    num_cycles : int
        Number of alternating W/A update cycles.
    w_step_size : float
        Step size for W update.
    A_step_size : float
        Step size for A update.
    max_iter : int
        Max iterations per update.
    tol : float
        Convergence tolerance.
    eps : float
        Numerical stability constant.
    w_method : str
        Method for W update ("fw", "bcd", "pg", "md", "fmd", "d").
    a_method : str
        Method for A update ("fista", "bcd", "pg", "qn").
    verbose : bool
        Print progress.
    A_graph : (n, n) ndarray or None
        Target adjacency matrix for graph term. Required when w_method="d".
    mu : float
        Weight on the clustering term.
    mu_graph : float
        Weight on the graph term ||L - A_graph||^2.

    Returns
    -------
    W : ndarray
        Refined W.
    A : ndarray
        Refined A.
    """
    for _ in range(int(num_cycles)):
        W = update_W(
            W_init=W,
            A=A,
            X=X,
            gamma_obj=gamma_obj,
            method=w_method,
            step_size=w_step_size,
            max_iter=max_iter,
            randomized=True,
            verbose=verbose,
            use_backtracking=use_backtracking,
            seed=0,
            A_graph=A_graph,
            mu=mu,
            mu_graph=mu_graph,
        )

        A = update_A(
            A_init=A,
            W=W,
            X=X,
            gamma_obj=gamma_obj,
            ell=ell,
            method=a_method,
            step_size=A_step_size,
            max_iter=max_iter,
            tol=tol,
            eps=eps,
            mu=mu,
            verbose=verbose,
        )

    return W, A


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

    # Vectorized reprojection of all columns
    W_new = project_columns_to_simplex(W_new)

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
    w_step_size: float = 1e-2,
    A_step_size: float = 1e-3,
    pg_tol: float = 1e-6,
    eps: float = 1e-12,
    verbose: bool = False,
    # --- new: graph parameters (defaults recover feature-only behaviour) ---
    A_graph: np.ndarray | None = None,
    mu: float = 1.0,
    mu_graph: float = 0.0,
    w_method: str = "fmd",
    A_graph_norm_sq: float | None = None,
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
    """
    if not (0.0 < eta <= 1.0):
        raise ValueError("eta must be in (0, 1].")

    # X = np.asarray(X, dtype=float)
    W_cur = W
    # W_cur = np.asarray(W, dtype=float).copy()
    # A_cur = np.asarray(A_atoms, dtype=float).copy()
    A_cur = A_atoms

    n, d = X.shape
    threshold = float(eta * obj_prev + (1.0 - eta) * obj_post)

    M_cur = build_M_from_W(W_cur, A_cur, n, d)
    obj_cur = objective_coupled(
        X, M_cur, gamma_obj, W_cur, A_graph=A_graph, mu=mu, mu_graph=mu_graph, eps=eps,
        A_graph_norm_sq=A_graph_norm_sq,
    )

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

        for j in reversed(range(k)):
            # 1) Remove atom j and reproject W to keep feasibility
            W_try = remove_atom_and_reproject_W(W_cur, j)
            A_try = np.delete(A_cur, j, axis=0)

            # 2) Optional small refinement after removal
            if trunc_refine_cycles > 0:
                if ell is None:
                    raise ValueError(
                        "ell must be provided when trunc_refine_cycles > 0."
                    )

                W_try, A_try = alternating_WA_refinement(
                    W=W_try,
                    A=A_try,
                    X=X,
                    gamma_obj=gamma_obj,
                    ell=ell,
                    num_cycles=trunc_refine_cycles,
                    w_step_size=w_step_size,
                    A_step_size=A_step_size,
                    max_iter=pg_max_iter,
                    tol=pg_tol,
                    eps=eps,
                    verbose=False,
                    A_graph=A_graph,
                    mu=mu,
                    mu_graph=mu_graph,
                    w_method=w_method,
                )

            # 3) Evaluate true objective after removal
            M_try = build_M_from_W(W_try, A_try, n, d)
            obj_try = objective_coupled(
                X,
                M_try,
                gamma_obj,
                W_try,
                A_graph=A_graph,
                mu=mu,
                mu_graph=mu_graph,
                eps=eps,
                A_graph_norm_sq=A_graph_norm_sq,
            )

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
    """Vectorized computation of sum of pairwise l2 norms."""
    # Compute all pairwise differences using broadcasting: M[i] - M[j]
    # diff[i,j] = M[i] - M[j], shape (n, n, d)
    diff = M[:, np.newaxis, :] - M[np.newaxis, :, :]
    # Compute norms, shape (n, n)
    norms = np.linalg.norm(diff, axis=2)
    # Sum upper triangle (excluding diagonal)
    return float(np.sum(np.triu(norms, k=1)))


def son_subgrad_all_pairs(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Vectorized computation of SON subgradient."""
    M = np.asarray(M, dtype=float)
    n, d = M.shape

    # Compute all pairwise differences: diff[v,u] = M[v] - M[u], shape (n, n, d)
    diff = M[:, np.newaxis, :] - M[np.newaxis, :, :]

    # Compute norms, shape (n, n)
    norms = np.linalg.norm(diff, axis=2)

    # Avoid division by zero: set small norms to inf so 1/norm -> 0
    norms_safe = np.where(norms > eps, norms, np.inf)

    # Normalized differences: shape (n, n, d)
    normalized = diff / norms_safe[:, :, np.newaxis]

    # Sum over all u != v (diagonal is already zero due to diff[v,v]=0)
    G = np.sum(normalized, axis=1)

    return G


def objective_full(X: np.ndarray, M: np.ndarray, gamma: float) -> float:
    data = float(np.sum((X - M) ** 2))
    son = son_value_all_pairs(M)
    return data + gamma * son


def objective_coupled(
    X: np.ndarray,
    M: np.ndarray,
    gamma: float,
    W: np.ndarray,
    A_graph: np.ndarray | None = None,
    mu: float = 1.0,
    mu_graph: float = 0.0,
    eps: float = 1e-12,
    A_graph_norm_sq: float | None = None,
) -> float:
    """
    Coupled objective:
        F = mu_graph * ||L - A_graph||_F^2  +  mu * (||X - M||^2 + gamma * SON)

    When mu_graph=0 this reduces to mu * objective_full.
    When mu=0 this reduces to mu_graph * ||L - A_graph||^2.

    Efficient: avoids forming the n×n matrix L = D^T D.
    Uses ||L-A||^2 = ||DD^T||^2_F - 2 * sum(DA * D) + ||A||^2_F
    where DD^T is K×K (small).
    """
    # Skip SON computation when mu=0 (graph-only mode)
    if mu == 0.0:
        clust = 0.0
    else:
        clust = objective_full(X, M, gamma)

    graph_term = 0.0
    if mu_graph != 0.0 and A_graph is not None and W.shape[0] > 0:
        D = np.sqrt(np.maximum(W, eps))           # (K, n)
        DDT = D @ D.T                              # (K, K) — cheap
        # ||L||^2_F = ||DD^T||^2_F = tr((DD^T)^2)
        L_norm_sq = float(np.sum(DDT ** 2))

        # tr(L A) = sum_{i,v} (DA)_{iv} * D_{iv}
        if scipy.sparse.issparse(A_graph):
            DA = D @ A_graph                       # (K, n) sparse matmul
        else:
            DA = D @ A_graph                       # (K, n)
        trLA = float(np.sum(DA * D))

        # ||A||^2_F — precomputed if provided
        if A_graph_norm_sq is not None:
            A_norm_sq = A_graph_norm_sq
        elif scipy.sparse.issparse(A_graph):
            A_norm_sq = float(A_graph.multiply(A_graph).sum())
        else:
            A_norm_sq = float(np.sum(A_graph ** 2))

        graph_term = L_norm_sq - 2.0 * trLA + A_norm_sq

    return mu_graph * graph_term + mu * clust


def q_rows_full(
    X: np.ndarray, M: np.ndarray, gamma: float, mu: float = 1.0, eps: float = 1e-12
) -> np.ndarray:
    # Short-circuit when mu=0 (graph-only mode): skip O(n²d) SON subgradient
    if mu == 0.0:
        return np.zeros_like(M)
    G_son = son_subgrad_all_pairs(M, eps=eps)
    return mu * (2.0 * (M - X) + gamma * G_son)


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


def forward_oracle_coupled(
    D: np.ndarray,
    A_graph,
    Q_M: np.ndarray,
    mu: float,
    mu_graph: float = 0.0,
    ell: float = 1.0,
    n_alt_iters: int = 10,
    eps: float = 1e-12,
):
    """
    Unified forward oracle via alternating minimization over (e, a).

    Solves:  min_{||e||=1, ||a||<=ell}  e^T G e  +  mu * a^T Q_M^T (e o e)

    where G = 2 * mu_graph * (L - A_graph), L = D^T D.

    e-update: eigenvector with largest |eigenvalue| of  G + Diag(mu * Q_M @ a)
    a-update: a = -ell * Q_M^T(e^2) / ||Q_M^T(e^2)||

    When mu_graph=0, G=0 and the oracle is driven by clustering only.
    When mu=0, the a-update is irrelevant and e = extremal eigvec of G.

    Uses an iterative eigensolver (scipy.sparse.linalg.eigsh) with implicit
    matvec to avoid forming the n×n matrix G:
        Gx = 2*mu_graph*(D^T(Dx) - A_graph @ x)

    Parameters
    ----------
    D : (K, n) array — sqrt of W
    A_graph : (n, n) array or sparse matrix — target adjacency/affinity
    Q_M : (n, d) array — clustering gradient rows
    mu, mu_graph : floats — weights
    ell : float — l2 ball radius
    n_alt_iters : int — alternating iterations
    eps : float — numerical tolerance

    Returns (e_star, a_star).
    """
    n = D.shape[1] if D.shape[0] > 0 else Q_M.shape[0]
    d = Q_M.shape[1]

    a = np.zeros(d, dtype=float)
    e = np.zeros(n, dtype=float)

    A_is_sparse = scipy.sparse.issparse(A_graph)

    for _ in range(n_alt_iters):
        # e-update: find eigenvector with largest |eigenvalue| of H
        # H = 2*mu_graph*(L - A_graph) + Diag(mu * Q_M @ a)
        # Implicit matvec: Hx = 2*mu_graph*(D^T(Dx) - A_graph @ x) + diag_vec * x
        diag_vec = mu * (Q_M @ a)  # (n,)

        def _matvec(x, D=D, A_graph=A_graph, mu_graph=mu_graph,
                    diag_vec=diag_vec, A_is_sparse=A_is_sparse):
            x = np.asarray(x, dtype=float).ravel()
            # Graph part: 2*mu_graph * (D^T(Dx) - Ax)
            if mu_graph != 0.0 and D.shape[0] > 0:
                Dx = D @ x                         # (K,)
                DTDx = D.T @ Dx                     # (n,)
                if A_is_sparse:
                    Ax = A_graph @ x                # (n,)
                else:
                    Ax = A_graph @ x                # (n,)
                graph_part = 2.0 * mu_graph * (DTDx - Ax)
            else:
                graph_part = np.zeros(len(x), dtype=float)
            return graph_part + diag_vec * x

        H_op = scipy.sparse.linalg.LinearOperator(
            shape=(n, n), matvec=_matvec, dtype=float
        )

        # Find extremal eigenvalues (largest and smallest) to pick the one
        # with largest absolute value
        try:
            # Get largest eigenvalue
            eigvals_lg, eigvecs_lg = scipy.sparse.linalg.eigsh(
                H_op, k=1, which='LA', tol=1e-6, maxiter=300
            )
            # Get smallest eigenvalue
            eigvals_sm, eigvecs_sm = scipy.sparse.linalg.eigsh(
                H_op, k=1, which='SA', tol=1e-6, maxiter=300
            )

            if abs(eigvals_lg[0]) >= abs(eigvals_sm[0]):
                l = abs(eigvals_lg[0])
                e = eigvecs_lg[:, 0]
            else:
                l = abs(eigvals_sm[0])
                e = eigvecs_sm[:, 0]
        except Exception:
            # Fallback: if iterative solver fails, build H explicitly
            if D.shape[0] > 0:
                L = D.T @ D
                G = 2.0 * mu_graph * (L - A_graph)
                if A_is_sparse:
                    G = np.asarray(G)
            else:
                G = np.zeros((n, n), dtype=float)
            H = G + np.diag(diag_vec)
            H = 0.5 * (H + H.T)
            eigvals_all, eigvecs_all = np.linalg.eigh(H)
            i = np.argmax(np.abs(eigvals_all))
            e = eigvecs_all[:, i]
            l = abs(eigvals_all[i])

        # a-update: a = -ell * Q_M^T(e^2) / ||Q_M^T(e^2)||
        e_sq = e**2
        g = Q_M.T @ e_sq
        g_norm = float(np.linalg.norm(g))
        if g_norm > eps:
            a = -ell * g / g_norm
        else:
            a = np.zeros(d, dtype=float)

    return np.sqrt(l) * e, a


def augment_W_with_rho(W: np.ndarray, rho: float, n: int) -> np.ndarray:
    rho = float(np.clip(rho, 0.0, 1.0))
    if W.size == 0:
        return np.ones((1, n), dtype=float)
    return np.vstack([(1.0 - rho) * W, rho * np.ones((1, n), dtype=float)])


def augment_W_with_e_star(
    W: np.ndarray,
    e_star: np.ndarray,
    rho: float,
    n: int,
) -> np.ndarray:
    """
    Add a new atom row to W using the e_star direction from the forward oracle.

    New row:  w_new = e_star^2  (sums to 1 because ||e_star||=1)
    Mixing:   W_new = [(1-rho) * W_old;  rho * w_new]

    This gives the new atom a non-uniform weight pattern, which is essential
    for breaking symmetry in the D-reparameterization.
    """
    rho = float(np.clip(rho, 1e-3, 1.0))  # must be > 0 for D-reparam
    w_new = (e_star**2).reshape(1, n)  # ||e||=1 => sums to 1
    if W.size == 0:
        return w_new
    D = np.vstack([(1.0 - rho) * W, rho * w_new])
    D = np.array([project_to_simplex(d) for d in D.T]).T
    return D


# ============================================================
# Main algorithm in W-form
# ============================================================


def cogent_structured_atoms_with_W(
    X: np.ndarray,
    gamma_obj: float,
    ell: float,
    max_iters: int = 20,
    init_with_one_atom: bool = True,
    inner_alt_cycles: int = 5,
    pg_max_iter: int = 300,
    w_step_size: float = 1e-2,
    A_step_size: float = 1e-3,
    pg_tol: float = 1e-6,
    eps: float = 1e-12,
    eta: float = 0.5,
    verbose: bool = True,
    truncation: bool = True,
    seed: int | None = None,
    # --- new: graph parameters (defaults recover feature-only behaviour) ---
    A_graph: np.ndarray | None = None,
    mu: float = 1.0,
    mu_graph: float = 0.0,
    w_method: str = "fmd",
    fwd_alt_iters: int = 10,
) -> dict:
    """
    CoGEnT-style solver for the coupled graph + feature objective:

        F = mu_graph * ||L - A_graph||^2  +  mu * (||X - M||^2 + gamma * SON)

    where L = D^T D, D = sqrt(W), M = W^T A_atoms.

    When mu_graph=0 (default) this reduces to the feature-only problem.

    New parameters
    --------------
    A_graph : (n, n) ndarray or None
        Target adjacency matrix.  Required when mu_graph > 0.
    mu : float
        Weight on the clustering term (default 1.0).
    mu_graph : float
        Weight on the graph term (default 0.0 = feature-only).
    w_method : str
        W-update method.  Use "d" for coupled objective (D-reparameterization).
    fwd_alt_iters : int
        Alternating iterations in the forward oracle (only used when mu_graph > 0).
    """
    if seed is not None:
        np.random.seed(seed)

    X = np.asarray(X, dtype=float)
    n, d = X.shape

    # If graph term is active, we need A_graph
    if mu_graph != 0.0 and A_graph is None:
        raise ValueError("A_graph must be provided when mu_graph != 0.")

    # Default A_graph to zeros so downstream code doesn't need None checks
    if A_graph is None:
        A_graph = np.zeros((n, n), dtype=float)

    # Precompute ||A_graph||^2_F once (used in efficient objective)
    if scipy.sparse.issparse(A_graph):
        _A_graph_norm_sq = float(A_graph.multiply(A_graph).sum())
    else:
        _A_graph_norm_sq = float(np.sum(A_graph ** 2))

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
        obj_prev = objective_coupled(
            X, M, gamma_obj, W, A_graph=A_graph, mu=mu, mu_graph=mu_graph, eps=eps,
            A_graph_norm_sq=_A_graph_norm_sq,
        )

        # ---- forward oracle ----
        if mu_graph == 0.0:
            # Feature-only: use original forward atom (no eigenvector needed)
            Q = q_rows_full(X, M, gamma_obj, mu=mu, eps=eps)
            a_new, v_star = forward_atom_from_Q(Q, ell=ell, eps=eps)
        else:
            # Coupled: unified forward oracle with alternating (e, a)
            D = (
                np.sqrt(np.maximum(W, eps))
                if W.shape[0] > 0
                else np.zeros((0, n), dtype=float)
            )
            Q_M = q_rows_full(X, M, gamma_obj, mu=mu, eps=eps)
            e_star, a_new = forward_oracle_coupled(
                D=D,
                A_graph=A_graph,
                Q_M=Q_M,
                mu=mu,
                mu_graph=mu_graph,
                ell=ell,
                n_alt_iters=fwd_alt_iters,
                eps=eps,
            )
            v_star = int(np.argmax(e_star**2))

        rho_opt = 0

        if mu_graph != 0.0 and (w_method == "d" or w_method == "d-spg"):
            # D-reparameterization: use e_star^2 as the new atom's weight
            # pattern (non-uniform) with rho >= 1e-3 to break symmetry.
            # e_star = np.ones(n, dtype=float)
            W = augment_W_with_e_star(W, e_star, rho=0.1, n=n)
        else:
            W = augment_W_with_rho(W, rho_opt, n)
        A_atoms = (
            np.vstack([A_atoms, a_new.reshape(1, d)])
            if A_atoms.size
            else a_new.reshape(1, d)
        )

        for _ in range(int(inner_alt_cycles)):
            W, A_atoms = alternating_WA_refinement(
                W=W,
                A=A_atoms,
                X=X,
                gamma_obj=gamma_obj,
                ell=ell,
                num_cycles=1,
                w_step_size=w_step_size,
                A_step_size=A_step_size,
                max_iter=pg_max_iter,
                tol=pg_tol,
                eps=eps,
                verbose=False,
                A_graph=A_graph,
                mu=mu,
                mu_graph=mu_graph,
                w_method=w_method,
            )

        M = build_M_from_W(W, A_atoms, n, d)
        obj_post = objective_coupled(
            X, M, gamma_obj, W, A_graph=A_graph, mu=mu, mu_graph=mu_graph, eps=eps,
            A_graph_norm_sq=_A_graph_norm_sq,
        )

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
                eta=eta,  # same role as in the paper threshold
                ell=ell,
                trunc_refine_cycles=inner_alt_cycles,  # or 0 if you want faster truncation
                pg_max_iter=pg_max_iter,
                w_step_size=w_step_size,
                A_step_size=A_step_size,
                pg_tol=pg_tol,
                verbose=False,
                A_graph=A_graph,
                mu=mu,
                mu_graph=mu_graph,
                w_method=w_method,
                A_graph_norm_sq=_A_graph_norm_sq,
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
                "q_norm_max": float(
                    np.max(
                        np.linalg.norm(
                            q_rows_full(X, M, gamma_obj, mu=mu, eps=eps), axis=1
                        )
                    )
                ),
            }
        )

        if verbose:
            print(
                f"iter={t:02d} "
                f"K={W.shape[0]} "
                # f"obj_prev={obj_prev:.4f} "
                f"obj_post={obj_post:.4f} "
                # f"rho={rho_opt:.4f} "
                # f"v_star={v_star}"
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
# Minimal test
# ============================================================

if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)

    d = 2
    ell = 1.0
    K_true = 3
    n_per_cluster = 80
    sigma = 0.15
    n = K_true * n_per_cluster
    gamma_val = 0.75 / n_per_cluster
    mu_feature = 0.0
    mu_graph = 1.0 / n

    radii = [2.0, 1.6, 1.8]
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

    # ---- 2. Generate SBM graph ----
    p_intra = 0.6
    q_inter = 0.15
    A, A_bar = generate_sbm_graph(labels, p=p_intra, q=q_inter, seed=seed)
    n = X.shape[0]

    result = solve_A_hard_assignments_cvxpy(
        X=X,
        labels=labels,
        ell=ell,
        gamma=gamma_val,
        solver=None,  # you can set "ECOS", "SCS", or "CLARABEL"
        verbose=False,
    )

    A_opt = result["A_opt"]
    W_true = build_W_from_labels(labels, K=K_true)
    # obj_check = objective_from_WA(X, W_true, A_opt, gamma_val)
    M = build_M_from_W(W_true, A_opt, n, d)
    obj_check = objective_coupled(
        X, M, gamma_val, W_true, A_graph=A, mu=mu_feature, mu_graph=mu_graph
    )

    print("Status:", result["status"])
    print("Cluster sizes:", result["cluster_sizes"])
    print("Optimal A:")
    print(np.array2string(A_opt, precision=4, suppress_small=True))
    print("Empirical means (gamma=0 solution before ball projection):")
    print(np.array2string(result["empirical_means"], precision=4, suppress_small=True))
    print("Fit value:", result["fit_value"])
    print("Fusion value:", result["fusion_value"])
    print("Optimal objective:", result["objective_value"])
    print("Objective recomputed from W,A:", obj_check)

    res = cogent_structured_atoms_with_W(
        X=X,
        gamma_obj=gamma_val,
        ell=ell,
        max_iters=40,
        init_with_one_atom=False,
        inner_alt_cycles=1,
        pg_max_iter=1000,
        w_step_size=1e-3,
        A_step_size=1e-3,
        pg_tol=1e-7,
        verbose=True,
        truncation=True,  # set to True to enable backward truncation
        seed=seed,
        A_graph=A,
        mu=mu_feature,
        mu_graph=mu_graph,
        w_method="d",
        eta=0.5,  # truncation threshold parameter (1.0 = only accept if obj does not increase)
        fwd_alt_iters=1,
    )

    print("\nFinal number of atoms:", res["A_atoms"].shape[0])
    print("Final objective:", res["obj_hist"][-1] if res["obj_hist"].size else None)
    print("Final lambdas:")
    print(np.array2string(res["lam"], precision=4, suppress_small=True))
    print("Final atoms A:")
    print(np.array2string(res["A_atoms"], precision=4, suppress_small=True))

    # plot_solution(X, labels, A_opt, centroids, ell, res, gamma_val)
    # plt.show()

    # # print the final W matrix
    # print("\nFinal W matrix (shape {}):".format(res["W"].shape))
    # print(np.array2string(res["W"], precision=4, suppress_small=True))

    # plot the final L vs A_graph
    D = np.sqrt(np.maximum(res["W"], 1e-12))
    L = D.T @ D
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title("Learned L = D^T D")
    plt.imshow(L, cmap="viridis")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.title("Target A_graph")
    plt.imshow(A, cmap="viridis")
    plt.colorbar()
    plt.tight_layout()
    plt.show()
