import numpy as np
import matplotlib.pyplot as plt

from optimized_solution import (
    build_W_from_labels,
    objective_from_WA,
    solve_A_hard_assignments_cvxpy,
)

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
    step_size: float = 1e-2,
    max_iter: int = 100,
    tol: float = 1e-7,
    eps: float = 1e-12,
    w_method: str = "pg",
    a_method: str = "fista",
    verbose: bool = False,
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
    step_size : float
        Step size for optimization.
    max_iter : int
        Max iterations per update.
    tol : float
        Convergence tolerance.
    eps : float
        Numerical stability constant.
    w_method : str
        Method for W update ("fw", "bcd", "pg", "md").
    a_method : str
        Method for A update ("fista", "bcd", "pg", "qn").
    verbose : bool
        Print progress.

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
            step_size=step_size,
            max_iter=max_iter,
            randomized=True,
            seed=0,
            verbose=verbose,
        )

        A = update_A(
            A_init=A,
            W=W,
            X=X,
            gamma_obj=gamma_obj,
            ell=ell,
            method=a_method,
            step_size=step_size,
            max_iter=max_iter,
            tol=tol,
            eps=eps,
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

    # X = np.asarray(X, dtype=float)
    W_cur = W
    # W_cur = np.asarray(W, dtype=float).copy()
    # A_cur = np.asarray(A_atoms, dtype=float).copy()
    A_cur = A_atoms

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
                    step_size=pg_step_size,
                    max_iter=pg_max_iter,
                    tol=pg_tol,
                    eps=eps,
                    verbose=False,
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

        rho_opt = 0

        W = augment_W_with_rho(W, rho_opt, n)
        A_atoms = (
            np.vstack([A_atoms, a_new.reshape(1, d)])
            if A_atoms.size
            else a_new.reshape(1, d)
        )

        for _ in range(int(inner_alt_cycles)):
            # Lc = build_Lc_from_WA(W, A_atoms, eps=eps)

            W, A_atoms = alternating_WA_refinement(
                W=W,
                A=A_atoms,
                X=X,
                gamma_obj=gamma_obj,
                ell=ell,
                num_cycles=1,
                step_size=pg_step_size,
                max_iter=pg_max_iter,
                tol=pg_tol,
                eps=eps,
                verbose=False,
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
                trunc_refine_cycles=inner_alt_cycles,  # or 0 if you want faster truncation
                pg_max_iter=pg_max_iter,
                pg_step_size=1e-2,
                pg_tol=1e-6,
                verbose=False,
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
    n_per_cluster = 100
    sigma = 0.10
    gamma_val = 0.75 / n_per_cluster

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
    obj_check = objective_from_WA(X, W_true, A_opt, gamma_val)

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

    plot_data(X, labels, centroids, ell)

    res = cogent_structured_atoms_with_W(
        X=X,
        gamma_obj=gamma_val,
        ell=ell,
        max_iters=10,
        init_with_one_atom=False,
        rho_grid_size=21,
        inner_alt_cycles=2,
        pg_max_iter=100,
        pg_step_size=1e-2,
        pg_tol=1e-9,
        verbose=True,
        truncation=True,  # set to True to enable backward truncation
        seed=seed,
    )

    print("\nFinal number of atoms:", res["A_atoms"].shape[0])
    print("Final objective:", res["obj_hist"][-1] if res["obj_hist"].size else None)
    print("Final lambdas:")
    print(np.array2string(res["lam"], precision=4, suppress_small=True))
    print("Final atoms A:")
    print(np.array2string(res["A_atoms"], precision=4, suppress_small=True))

    plot_solution(X, labels, A_opt, centroids, ell, res, gamma_val)
    plt.show()
