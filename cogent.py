import numpy as np
import matplotlib.pyplot as plt

from optimized_solution import (
    build_W_from_labels,
    objective_from_WA,
    solve_A_hard_assignments_cvxpy,
)

# from update_A import update_A_projected_gradient_original
from update_A import update_A, canonicalize_atoms

# from update_W import update_W_projected_gradient_original
from update_W import update_W, graph_term_value

from update_Pi import (
    update_Pi,
    label_term_value,
    predict_node_labels,
)

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
    num_cycles: int = 1,
    step_size: float = 1e-2,
    max_iter: int = 100,
    tol: float = 1e-7,
    eps: float = 1e-12,
    w_method: str = "pg",
    a_method: str = "fista",
    verbose: bool = False,
    A_graph: np.ndarray | None = None,
    gamma_graph: float = 0.0,
    Pi: np.ndarray | None = None,
    tilde_y: np.ndarray | None = None,
    train_mask: np.ndarray | None = None,
    beta: float = 0.0,
    K_classes: int = 0,
    pi_method: str = "cf",
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    num_cycles : int
        Number of alternating W/A update cycles.
    step_size : float
        Step size for the W update (projected gradient).
    max_iter : int
        Max iterations per update.
    tol : float
        Convergence tolerance.
    eps : float
        Numerical stability constant.
    w_method : str
        Method for W update ("fw", "bcd", "pg", "md").
    a_method : str
        Method for A update ("fista", "cf", "bcd", "pg", "qn").
    verbose : bool
        Print progress.
    A_graph : (n, n) ndarray or None
        Graph adjacency matrix. If provided, adds gamma_graph * ||A_graph - L||_1
        to the W subproblem objective where L = sqrt(W)^T sqrt(W).
    gamma_graph : float
        Weight of the graph term.

    Returns
    -------
    W : ndarray
        Refined W.
    A : ndarray
        Refined A.
    """
    use_label = (
        Pi is not None
        and tilde_y is not None
        and train_mask is not None
        and beta != 0.0
        and K_classes > 0
    )

    for _ in range(int(num_cycles)):
        W = update_W(
            W_init=W,
            A=A,
            X=X,
            gamma_obj=gamma_obj,
            method=w_method,
            step_size=step_size,
            max_iter=max_iter,
            randomized=True,
            seed=0,
            verbose=verbose,
            A_graph=A_graph,
            gamma_graph=gamma_graph,
        )

        # Use a conservative step for A: 1 / lambda_max gives cheap FISTA
        # iterations (accepted on first backtracking try) while remaining safe.
        lam_max = float(W.sum(axis=1).max()) if W.size else 1.0
        a_step = 1.0 / max(lam_max, 1.0)

        A = update_A(
            A_init=A,
            W=W,
            X=X,
            gamma_obj=gamma_obj,
            method=a_method,
            step_size=a_step,
            max_iter=max_iter,
            tol=tol,
            eps=eps,
            verbose=verbose,
        )

        if use_label and Pi.size > 0:
            Pi = update_Pi(
                Pi_init=Pi,
                W=W,
                tilde_y=tilde_y,
                train_mask=train_mask,
                beta=beta,
                K_classes=K_classes,
                method=pi_method,
            )

    if use_label:
        return W, A, Pi
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
    trunc_refine_cycles: int = 2,
    pg_max_iter: int = 100,
    pg_step_size: float = 1e-2,
    pg_tol: float = 1e-6,
    eps: float = 1e-12,
    verbose: bool = False,
    A_graph: np.ndarray | None = None,
    gamma_graph: float = 0.0,
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
            F = eta f(x_t) + (1-eta) f(x~_{t+1}).
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

    # X = np.asarray(X, dtype=float)
    W_cur = W
    # W_cur = np.asarray(W, dtype=float).copy()
    # A_cur = np.asarray(A_atoms, dtype=float).copy()
    A_cur = A_atoms

    n, d = X.shape
    threshold = float(eta * obj_prev + (1.0 - eta) * obj_post)

    obj_cur = objective_full(X, W_cur, A_cur, gamma_obj, A_graph, gamma_graph)

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
                W_try, A_try = alternating_WA_refinement(
                    W=W_try,
                    A=A_try,
                    X=X,
                    gamma_obj=gamma_obj,
                    num_cycles=trunc_refine_cycles,
                    step_size=pg_step_size,
                    max_iter=pg_max_iter,
                    tol=pg_tol,
                    eps=eps,
                    verbose=False,
                    A_graph=A_graph,
                    gamma_graph=gamma_graph,
                )

            # 3) Evaluate true objective after removal
            obj_try = objective_full(X, W_try, A_try, gamma_obj, A_graph, gamma_graph)

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


def _update_A_hard_clusters(
    X: np.ndarray,
    hard_labels: np.ndarray,
    gamma: float,
    max_iter: int = 500,
    tol: float = 1e-10,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Solve for optimal A given hard (0/1) cluster assignments.

    With hard W the objective collapses to a k-atom problem with only
    k*(k-1)/2 pairs (instead of n*(n-1)/2), making this very fast.

        min_A  (1/2) sum_k n_k ||a_k - xbar_k||^2
               + gamma * sum_{k<l} n_k*n_l * ||a_k - a_l||

    Uses FISTA with backtracking.
    """
    K = int(hard_labels.max()) + 1
    n, d = X.shape

    cluster_sizes = np.array([np.sum(hard_labels == k) for k in range(K)], dtype=float)
    cluster_means = np.vstack([X[hard_labels == k].mean(axis=0) for k in range(K)])

    # Cluster-level pair indices  (only K*(K-1)/2 pairs)
    ku_idx, kv_idx = np.triu_indices(K, k=1)  # (P,) each, P = K*(K-1)/2
    weights = cluster_sizes[ku_idx] * cluster_sizes[kv_idx]  # (P,)

    def _grad_and_obj(A):
        # Fit gradient: diag(n_k) A - diag(n_k) xbar  =>  n_k*(a_k - xbar_k)
        G = cluster_sizes[:, None] * (A - cluster_means)  # (K, d)
        fit = 0.5 * float(np.sum(cluster_sizes[:, None] * (A - cluster_means) ** 2))

        # SON between cluster representatives, weighted by n_k * n_l
        Z = A[ku_idx] - A[kv_idx]  # (P, d)
        norms = np.linalg.norm(Z, axis=1)  # (P,)
        fused = float(np.dot(weights, norms))

        U = np.zeros_like(Z)
        mask = norms > eps
        U[mask] = Z[mask] / norms[mask, None]

        # Gradient contribution from each pair to row k
        for p in range(len(ku_idx)):
            G[ku_idx[p]] += gamma * weights[p] * U[p]
            G[kv_idx[p]] -= gamma * weights[p] * U[p]

        return G, fit + gamma * fused

    # FISTA
    A = cluster_means.copy()
    Y = A.copy()
    t = 1.0
    _, obj = _grad_and_obj(A)

    step = 1.0 / float(np.max(cluster_sizes))  # safe Lipschitz init

    for _ in range(max_iter):
        G, _ = _grad_and_obj(Y)

        # backtracking line search
        while True:
            A_cand = Y - step * G
            _, obj_cand = _grad_and_obj(A_cand)
            if obj_cand <= obj:
                break
            step *= 0.5
            if step < 1e-20:
                break

        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        Y_new = A_cand + ((t - 1.0) / t_new) * (A_cand - A)
        # objective restart
        _, obj_y = _grad_and_obj(Y_new)
        if obj_y > obj_cand:
            t_new, Y_new = 1.0, A_cand.copy()

        rel = np.linalg.norm(A_cand - A) / max(np.linalg.norm(A), 1e-12)
        A, Y, t, obj = A_cand, Y_new, t_new, obj_cand
        if rel < tol:
            break

    return A


def objective_full(
    X: np.ndarray,
    W: np.ndarray,
    A: np.ndarray,
    gamma: float,
    A_graph: np.ndarray | None = None,
    gamma_graph: float = 0.0,
    Pi: np.ndarray | None = None,
    tilde_y: np.ndarray | None = None,
    train_mask: np.ndarray | None = None,
    beta: float = 0.0,
) -> float:
    """
    Eq.(15) full objective (weighted k-means + SON + optional graph + optional label):

        (1/2) sum_{i,v} W_iv ||x_v - a_i||^2
        + gamma * sum_{u<v} ||(W^T A)_u - (W^T A)_v||_2
        - gamma_graph * <Abar, L>           (if A_graph is not None)
        - beta  * sum_{v in train} sum_i W_iv * Pi_{i, tilde_y_v}
                                              (if Pi is not None and beta != 0)
          where Abar = 2*A_graph - 1,  L = sqrt(W)^T sqrt(W)
    """
    D = (
        np.sum(A**2, axis=1, keepdims=True)  # (k, 1)
        - 2.0 * (A @ X.T)  # (k, n)
        + np.sum(X**2, axis=1)[np.newaxis, :]  # (1, n)
    )
    fit = 0.5 * float(np.sum(W * D))
    M = W.T @ A
    son = son_value_all_pairs(M)
    graph = (
        gamma_graph * graph_term_value(W, A_graph)
        if A_graph is not None and gamma_graph != 0.0
        else 0.0
    )
    label = (
        label_term_value(W, Pi, tilde_y, train_mask, beta)
        if Pi is not None
        and beta != 0.0
        and tilde_y is not None
        and train_mask is not None
        else 0.0
    )
    return fit + gamma * son + graph + label


def q_rows_full(
    X: np.ndarray, M: np.ndarray, gamma: float, eps: float = 1e-12
) -> np.ndarray:
    """
    Per-node gradient used for forward-atom selection.

    grad_theta_v = (theta_v - x_v) + gamma * G_SON_v

    Corresponds to grad of (1/2)||x_v - theta_v||^2 + gamma*SON(M).
    """
    G_son = son_subgrad_all_pairs(M, eps=eps)
    return (M - X) + gamma * G_son


# ============================================================
# Forward atom: choose only a_new
# The new assignment row is initialized later through rho:
#   W_new = [ (1-rho) W ;
#             rho * 1_n^T ]
# so every data point gets rho mass on the new atom.
# ============================================================


def forward_atom_from_Q(Q: np.ndarray, X: np.ndarray, eps: float = 1e-12):
    norms = np.linalg.norm(Q, axis=1)
    v_star = int(np.argmax(norms))
    a_new = X[v_star].copy()
    return a_new, v_star


def augment_W_with_rho(W: np.ndarray, rho: float, n: int) -> np.ndarray:
    rho = float(np.clip(rho, 0.0, 1.0))
    if W.size == 0:
        return np.ones((1, n), dtype=float)
    return np.vstack([(1.0 - rho) * W, rho * np.ones((1, n), dtype=float)])


# ============================================================
# Main algorithm in W-form
# ============================================================


def cogent_structured_atoms_with_W(
    X: np.ndarray,
    gamma: float,
    mu: float = float("inf"),
    max_iters: int = 15,
    init_with_one_atom: bool = True,
    rho_grid_size: int = 21,
    inner_alt_cycles: int = 5,
    pg_max_iter: int = 30,
    pg_step_size: float = 1e-2,
    pg_tol: float = 1e-6,
    eps: float = 1e-12,
    verbose: bool = True,
    truncation: bool = True,
    seed: int | None = None,
    A_graph: np.ndarray | None = None,
    tilde_y: np.ndarray | None = None,
    train_mask: np.ndarray | None = None,
    beta: float = 0.0,
    K_classes: int = 0,
    pi_method: str = "cf",
    true_labels: np.ndarray | None = None,
) -> dict:
    """
    CoGEnT with quadratic (unconstrained) node loss.

    Parameters follow the paper (eq. 15):

        min  ||L-A||_1  +  mu*n * ( fit  +  gamma/n * SON )

    Equivalent (after dividing by mu*n and using ||L-A||_1 = const - <Abar,L>):

        min  fit  +  (gamma/n) * SON  +  (1/(mu*n)) * (-<Abar,L>)

    Parameters
    ----------
    gamma : float
        Fusion (SON) penalty — paper's γ.  The SON coefficient used internally
        is ``gamma_obj = gamma / n``.
    mu : float
        Relative weight of the node-information block vs. the graph term —
        paper's μ.  The graph-term weight used internally is
        ``gamma_graph = 1 / (mu * n)``.
        ``mu = inf`` (default) disables the graph term (node-only regime).
        ``mu → 0`` recovers the graph-only regime.
    A_graph : (n, n) ndarray, optional
        Binary adjacency matrix.  Required when mu < inf.

    Algorithm steps
    ---------------
    1) Find a_new from the forward step (initialized at the data point
       with the largest gradient norm).
    2) Find rho_opt and initialize
           W <- [ (1-rho_opt) W ;
                  rho_opt * 1_n^T ].
    3) Append a_new to A.
    4) Update W using projected gradient.
    5) Update A using unconstrained gradient descent.
    """
    if seed is not None:
        np.random.seed(seed)

    X = np.asarray(X, dtype=float)
    n, d = X.shape

    # Convert paper parameters (gamma, mu) to internal coefficients.
    #   gamma_obj  = gamma / n          (SON coefficient)
    #   gamma_graph = 1 / (mu * n)      (graph-term coefficient)
    # mu = inf  <=>  graph term disabled (node-only regime).
    gamma_obj: float = gamma / n
    use_graph = A_graph is not None and not np.isinf(mu) and mu > 0.0
    gamma_graph: float = (1.0 / (mu * n)) if use_graph else 0.0
    _A_graph = A_graph if use_graph else None

    use_label = (
        beta != 0.0 and tilde_y is not None and train_mask is not None and K_classes > 0
    )
    _tilde_y = np.asarray(tilde_y).astype(int) if use_label else None
    _train_mask = np.asarray(train_mask).astype(bool) if use_label else None

    W = np.zeros((0, n), dtype=float)
    A_atoms = np.zeros((0, d), dtype=float)
    Pi_atoms = np.zeros((0, K_classes if use_label else 0), dtype=float)

    if init_with_one_atom:
        a0 = np.mean(X, axis=0)
        W = np.ones((1, n), dtype=float)
        A_atoms = a0.reshape(1, d)
        if use_label:
            Pi_atoms = np.full((1, K_classes), 1.0 / K_classes, dtype=float)

    obj_hist = []
    info_hist = []

    for t in range(int(max_iters)):
        M = build_M_from_W(W, A_atoms, n, d)
        obj_prev = objective_full(
            X,
            W,
            A_atoms,
            gamma_obj,
            _A_graph,
            gamma_graph,
            Pi=Pi_atoms if use_label else None,
            tilde_y=_tilde_y,
            train_mask=_train_mask,
            beta=beta,
        )

        Q = q_rows_full(X, M, gamma_obj, eps=eps)
        a_new, v_star = forward_atom_from_Q(Q, X=X, eps=eps)

        # Give the new atom an equal initial share so weight can grow.
        # With the weighted-kmeans gradient >= 0, rho=0 traps the new atom
        # at zero weight forever.  rho = 1/(k+1) is the uniform distribution.
        k_cur = W.shape[0]
        rho_opt = 1.0 / (k_cur + 1)

        W = augment_W_with_rho(W, rho_opt, n)
        A_atoms = (
            np.vstack([A_atoms, a_new.reshape(1, d)])
            if A_atoms.size
            else a_new.reshape(1, d)
        )
        if use_label:
            new_pi = np.full((1, K_classes), 1.0 / K_classes, dtype=float)
            Pi_atoms = np.vstack([Pi_atoms, new_pi]) if Pi_atoms.size else new_pi

        for _ in range(int(inner_alt_cycles)):
            ret = alternating_WA_refinement(
                W=W,
                A=A_atoms,
                X=X,
                gamma_obj=gamma_obj,
                num_cycles=1,
                step_size=pg_step_size,
                max_iter=pg_max_iter,
                tol=pg_tol,
                eps=eps,
                verbose=False,
                A_graph=_A_graph,
                gamma_graph=gamma_graph,
                Pi=Pi_atoms if use_label else None,
                tilde_y=_tilde_y,
                train_mask=_train_mask,
                beta=beta,
                K_classes=K_classes if use_label else 0,
                pi_method=pi_method,
            )
            if use_label:
                W, A_atoms, Pi_atoms = ret
            else:
                W, A_atoms = ret

        M = build_M_from_W(W, A_atoms, n, d)
        obj_post = objective_full(
            X,
            W,
            A_atoms,
            gamma_obj,
            _A_graph,
            gamma_graph,
            Pi=Pi_atoms if use_label else None,
            tilde_y=_tilde_y,
            train_mask=_train_mask,
            beta=beta,
        )
        if truncation:
            W, A_atoms, obj_final, removed_log = truncation_backward_W(
                X=X,
                W=W,
                A_atoms=A_atoms,
                gamma_obj=gamma_obj,
                obj_prev=obj_prev,
                obj_post=obj_post,
                eta=0.9,
                trunc_refine_cycles=inner_alt_cycles,
                pg_max_iter=pg_max_iter,
                pg_step_size=1e-2,
                pg_tol=1e-6,
                verbose=False,
                A_graph=_A_graph,
                gamma_graph=gamma_graph,
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
            acc_str = ""
            if (
                use_label
                and Pi_atoms.size > 0
                and _tilde_y is not None
                and _train_mask is not None
            ):
                y_pred_iter = (W.T @ Pi_atoms).argmax(axis=1)
                train_acc = (y_pred_iter[_train_mask] == _tilde_y[_train_mask]).mean()
                acc_str = f"train_acc={train_acc:.4f} "
            cluster_acc_str = ""
            if true_labels is not None and W.size > 0:
                from scipy.optimize import linear_sum_assignment

                hard = W.argmax(axis=0)
                Kp = max(int(hard.max()), int(true_labels.max())) + 1
                Cm = np.zeros((Kp, Kp), dtype=int)
                for pi_, ti_ in zip(hard, true_labels):
                    Cm[pi_, ti_] += 1
                ri_, ci_ = linear_sum_assignment(-Cm)
                cluster_acc_str = (
                    f"cluster_acc={Cm[ri_, ci_].sum() / len(true_labels):.4f} "
                )
            print(
                f"iter={t:02d} "
                f"K={W.shape[0]} "
                f"obj_post={obj_post:.4f} "
                f"{acc_str}"
                f"{cluster_acc_str}"
            )

    M_final = build_M_from_W(W, A_atoms, n, d)
    lam, E_sq = recover_lambda_and_Esq(W, eps=eps)

    # Post-processing: derive interpretable atom positions.
    #
    # The objective only depends on M = W^T A (the predictions).  Many
    # different (W, A) pairs achieve the same M, so atoms found by the
    # soft alternating updates are not necessarily at meaningful positions.
    #
    # Strategy:
    #   1. Assign each node to its highest-weight atom ("hardening" W).
    #   2. Solve for the optimal A under those hard assignments using the
    #      closed-form Newton iteration.  This gives atoms at the biased
    #      class-centroid positions pulled by the SON fusion term — the
    #      same interpretable positions produced by the CVX reference.
    # if A_atoms.size > 0 and W.size > 0:
    #     hard_labels = W.argmax(axis=0)  # (n,)
    #     A_atoms = _update_A_hard_clusters(X, hard_labels, gamma_obj)

    return {
        "M": M_final,
        "W": W,  # shape (k, n)
        "S": W.T,  # compatibility view, shape (n, k)
        "A_atoms": A_atoms,  # shape (k, d)
        "Pi_atoms": Pi_atoms,  # shape (k, K_classes), simplex rows (empty if no label term)
        "pi_v": (W.T @ Pi_atoms) if Pi_atoms.size else np.zeros((n, 0)),
        "y_pred": (
            (W.T @ Pi_atoms).argmax(axis=1) if Pi_atoms.size else np.zeros(n, dtype=int)
        ),
        "lam": lam,  # lambda_i = sum_v w_iv
        "E_sq": E_sq,  # shape (k, n), e_{iv}^2
        "obj_hist": np.array(obj_hist, dtype=float),
        "info_hist": info_hist,
    }


# ============================================================
# Minimal test
# ============================================================

if __name__ == "__main__":
    import os
    import yaml  # pip install pyyaml

    # ── Load config ───────────────────────────────────────────────────────
    _cfg_path = os.path.join(os.path.dirname(__file__), "config", "cogent_config.yml")
    with open(_cfg_path) as _f:
        _cfg = yaml.safe_load(_f)

    _d_cfg = _cfg["data"]
    _g_cfg = _cfg["graph"]
    _m_cfg = _cfg["model"]
    _s_cfg = _cfg["solver"]

    seed = int(_d_cfg["seed"])
    d = int(_d_cfg["d"])
    ell = float(_d_cfg["ell"])
    K_true = int(_d_cfg["K_true"])
    n_per_cluster = int(_d_cfg["n_per_cluster"])
    sigma = float(_d_cfg["sigma"])
    radii = list(_d_cfg["radii"])
    num_active = int(_d_cfg["num_active_directions"])

    p = float(_g_cfg["p_out"])  # inter-cluster Bernoulli param

    gamma = float(_m_cfg["gamma"])  # paper's γ: SON coeff = γ/n
    mu = float(_m_cfg["mu"])  # paper's μ: graph weight = 1/(μ·n)

    max_iters = int(_s_cfg["max_iters"])
    init_one_atom = bool(_s_cfg["init_with_one_atom"])
    alt_cycles = int(_s_cfg["inner_alt_cycles"])
    pg_max_iter = int(_s_cfg["pg_max_iter"])
    pg_step_size = float(_s_cfg["pg_step_size"])
    pg_tol = float(_s_cfg["pg_tol"])
    eps = float(_s_cfg["eps"])
    truncation = bool(_s_cfg["truncation"])
    verbose = bool(_s_cfg["verbose"])

    np.random.seed(seed)

    directions = random_sparse_directions(K_true, d, num_active=num_active, seed=seed)

    # ── Data: uniform inside ball of radius rho := sigma around each centroid.
    # Guarantees max_v ||x_v - centroid_k|| == rho exactly (one boundary point
    # per cluster), and all other points strictly inside.
    rho = float(sigma)
    rng_data = np.random.default_rng(seed)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    dirs = directions / norms
    radii_arr = np.asarray(radii, dtype=float).reshape(-1)
    centroids = dirs * radii_arr[:, None]  # (K_true, d)

    X_list = []
    labels_list = []
    for k in range(K_true):
        # Uniform direction on unit sphere * uniform radius in [0, rho]
        u = rng_data.standard_normal((n_per_cluster, d))
        u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-12
        r = rng_data.uniform(0.0, rho, size=(n_per_cluster, 1))
        # Force the first point of each cluster to lie exactly on the sphere
        # so that max distance equals rho exactly.
        r[0, 0] = rho
        X_list.append(centroids[k] + u * r)
        labels_list.extend([k] * n_per_cluster)
    X = np.vstack(X_list)
    labels = np.array(labels_list, dtype=int)
    n = len(labels)

    # Sanity: max in-cluster distance should be exactly rho.
    max_dist = max(
        float(np.linalg.norm(X[labels == k] - centroids[k], axis=1).max())
        for k in range(K_true)
    )
    print(f"Data: rho={rho},  measured max ||x_v - centroid_k|| = {max_dist:.6f}")

    # Derived internal coefficients (shown for reference)
    gamma_obj_val = gamma / n  # SON coefficient
    gamma_graph_val = 1.0 / (mu * n)  # graph-term coefficient
    print(f"Config: gamma={gamma}, mu={mu}, n={n}")
    print(f"  -> gamma_obj (SON coeff)   = gamma/n      = {gamma_obj_val:.4e}")
    print(f"  -> gamma_graph (graph coeff) = 1/(mu*n) = {gamma_graph_val:.4e}")

    # ── SBM adjacency matrix (EXACT counts, not Bernoulli) ────────────────
    # p_max := p_out is interpreted as the EXACT fraction of cross-cluster
    # pairs that are connected (rounded to int).  Within-cluster fraction is
    # exactly (1 - p_out).  Edges within each pool are sampled uniformly
    # without replacement.
    rng = np.random.default_rng(seed)
    same_block = (labels[:, None] == labels[None, :]).astype(float)
    iu, ju = np.triu_indices(n, k=1)
    pair_same = labels[iu] == labels[ju]
    within_idx = np.flatnonzero(pair_same)
    between_idx = np.flatnonzero(~pair_same)
    n_within_target = int(round((1.0 - p) * within_idx.size))
    n_between_target = int(round(p * between_idx.size))
    chosen_within = rng.choice(within_idx, size=n_within_target, replace=False)
    chosen_between = rng.choice(between_idx, size=n_between_target, replace=False)
    A_graph = np.zeros((n, n), dtype=float)
    A_graph[iu[chosen_within], ju[chosen_within]] = 1.0
    A_graph[iu[chosen_between], ju[chosen_between]] = 1.0
    A_graph = A_graph + A_graph.T
    print(
        f"SBM (exact): within edges={n_within_target}/{within_idx.size}"
        f"  ({(1.0 - p):.3f}),  between edges={n_between_target}/{between_idx.size}"
        f"  ({p:.3f})"
    )

    # ── CVX reference (no graph term) ───────────────────────────────────
    result = solve_A_hard_assignments_cvxpy(
        X=X,
        labels=labels,
        gamma=gamma_obj_val,  # CVX solver uses internal gamma_obj
        solver=None,
        verbose=False,
    )
    A_opt = result["A_opt"]
    W_true = build_W_from_labels(labels, K=K_true)
    obj_check = objective_from_WA(X, W_true, A_opt, gamma_obj_val)

    print("CVX status:", result["status"])
    print("CVX optimal objective:", round(result["objective_value"], 5))
    print("CVX atoms:")
    print(np.array2string(A_opt, precision=4, suppress_small=True))

    # ── CoGEnT baseline (no graph term, mu=inf) ──────────────────────────
    print("\n── CoGEnT (no graph term) ──")
    res_base = cogent_structured_atoms_with_W(
        X=X,
        gamma=gamma,
        mu=float("inf"),  # node-only regime
        max_iters=max_iters,
        init_with_one_atom=init_one_atom,
        inner_alt_cycles=alt_cycles,
        pg_max_iter=pg_max_iter,
        pg_step_size=pg_step_size,
        pg_tol=pg_tol,
        verbose=verbose,
        truncation=truncation,
        seed=seed,
        true_labels=labels,
    )
    obj_base = res_base["obj_hist"][-1] if res_base["obj_hist"].size else float("nan")
    hard_base = res_base["W"].argmax(axis=0)

    # ── CoGEnT + graph term  -<Abar, L>,  Abar = 2A - 1 ─────────────────
    # paper's μ controls relative weight: graph coeff = 1/(μ·n)
    print(f"\n── CoGEnT + graph term -<Abar,L> (gamma={gamma}, mu={mu}, p={p}) ──")
    res_graph = cogent_structured_atoms_with_W(
        X=X,
        gamma=gamma,
        mu=mu,
        max_iters=max_iters,
        init_with_one_atom=init_one_atom,
        inner_alt_cycles=alt_cycles,
        pg_max_iter=pg_max_iter,
        pg_step_size=pg_step_size,
        pg_tol=pg_tol,
        verbose=verbose,
        truncation=truncation,
        seed=seed,
        A_graph=A_graph,
        true_labels=labels,
    )
    obj_graph = (
        res_graph["obj_hist"][-1] if res_graph["obj_hist"].size else float("nan")
    )
    hard_graph = res_graph["W"].argmax(axis=0)

    # ── Performance summary ──────────────────────────────────────────────
    def cluster_accuracy(pred_labels, true_labels):
        """Best-permutation accuracy via Hungarian matching (handles K_pred != K_true)."""
        from scipy.optimize import linear_sum_assignment

        K_p = int(pred_labels.max()) + 1
        K_t = int(true_labels.max()) + 1
        K = max(K_p, K_t)
        cost = np.zeros((K, K), dtype=int)
        for p_i, t_i in zip(pred_labels, true_labels):
            cost[p_i, t_i] += 1
        row_ind, col_ind = linear_sum_assignment(-cost)
        return float(cost[row_ind, col_ind].sum()) / len(true_labels)

    def graph_term_inner(W, A):
        """Evaluate -<Abar, L> where Abar = 2A-1, L = sqW^T sqW."""
        sqW = np.sqrt(np.maximum(W, 0.0))
        L = sqW.T @ sqW
        return float(-np.sum((2.0 * A - 1.0) * L))

    acc_base = cluster_accuracy(hard_base, labels)
    acc_graph = cluster_accuracy(hard_graph, labels)
    gt_base = graph_term_inner(res_base["W"], A_graph)
    gt_graph = graph_term_inner(res_graph["W"], A_graph)

    # SBM graph statistics (using true labels)
    n_within = int(A_graph[same_block == 1].sum()) // 2
    n_between = int(A_graph[same_block == 0].sum()) // 2
    dens_within = n_within / (n_per_cluster * (n_per_cluster - 1) // 2 * K_true)
    dens_between = n_between / (
        n_per_cluster * n_per_cluster * K_true * (K_true - 1) // 2
    )

    print("\n══════════ Graph statistics ══════════")
    print(f"  Total edges:              {n_within + n_between}")
    print(
        f"  Within-community edges:   {n_within}  (density={dens_within:.3f}, expected={1-p:.1f})"
    )
    print(
        f"  Between-community edges:  {n_between}  (density={dens_between:.3f}, expected={p:.1f})"
    )

    print("\n══════════ Performance summary ══════════")
    print(f"{'':30s} {'Baseline':>12s} {'+ Graph':>12s}")
    print(
        f"{'K (atoms found)':30s} {res_base['A_atoms'].shape[0]:>12d} {res_graph['A_atoms'].shape[0]:>12d}"
    )
    print(f"{'Obj (full)':30s} {obj_base:>12.4f} {obj_graph:>12.4f}")
    print(f"{'- <Abar,L> (lower=better)':30s} {gt_base:>12.4f} {gt_graph:>12.4f}")
    print(f"{'Cluster accuracy':30s} {acc_base:>12.4f} {acc_graph:>12.4f}")
    print(f"{'CVX obj (reference)':30s} {result['objective_value']:>12.4f}")
    print("══════════════════════════════════════════")

    # ── Plots ────────────────────────────────────────────────────────────
    cmap = plt.colormaps.get_cmap("tab10").resampled(K_true)
    node_colors = [cmap(labels[v]) for v in range(n)]

    # 1) Graph plot (node positions = X, edges from A_graph)
    fig_g, ax_g = plt.subplots(figsize=(7, 6))
    edges = np.argwhere(np.triu(A_graph, k=1))
    for u, v in edges:
        ax_g.plot(
            [X[u, 0], X[v, 0]],
            [X[u, 1], X[v, 1]],
            color="gray",
            linewidth=0.3,
            alpha=0.3,
            zorder=1,
        )
    ax_g.scatter(X[:, 0], X[:, 1], c=node_colors, s=20, zorder=2)
    ax_g.set_title(
        f"SBM graph  (p_in={1-p}, p_out={p}, n={n},  {edges.shape[0]} edges)"
    )
    ax_g.set_xlabel("x1")
    ax_g.set_ylabel("x2")
    ax_g.axis("equal")

    # 2) Side-by-side atom comparison
    fig_c, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, res, acc, title in zip(
        axes,
        [res_base, res_graph],
        [acc_base, acc_graph],
        [
            "Baseline (no graph)",
            "+ Graph term",
        ],
    ):
        hard = res["W"].argmax(axis=0)
        for k in range(K_true):
            idx = hard == k
            ax.scatter(X[idx, 0], X[idx, 1], alpha=0.4, label=f"pred {k}")
        C = np.asarray(centroids, dtype=float)
        ax.scatter(
            C[:, 0], C[:, 1], marker="x", s=200, color="black", label="true centroids"
        )
        A = res["A_atoms"]
        if A.size:
            ax.scatter(
                A[:, 0],
                A[:, 1],
                marker="o",
                s=180,
                edgecolors="red",
                facecolors="none",
                linewidths=2,
                label="CoGEnT atoms",
            )
        # CVX reference atoms
        if A_opt is not None:
            ax.scatter(
                A_opt[:, 0],
                A_opt[:, 1],
                marker="D",
                s=120,
                edgecolors="blue",
                facecolors="none",
                linewidths=2,
                label="CVX atoms",
            )
        ax.set_title(f"{title}\nacc={acc:.4f}", fontsize=11)
        ax.axis("equal")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.legend(fontsize=8)
    fig_c.suptitle(f"CoGEnT atoms  γ={gamma:.4f}  μ={mu:.1f}  p={p}", fontsize=13)
    plt.tight_layout()
    plt.show()
