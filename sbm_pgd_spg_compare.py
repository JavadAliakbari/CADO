
#!/usr/bin/env python3
"""
PGD and SPG solvers for

    minimize_D  0.5 * ||D^T D - A||_F^2
    subject to  D >= 0
                ||D[i, :]||_2 <= 1   for each row i

where A is a sparse adjacency matrix (n x n) and D is (k x n).

The script includes:
- SBM graph generation
- efficient objective/gradient routines that avoid forming D^T D
- Projected Gradient Descent (PGD)
- Spectral Projected Gradient (SPG) with BB step sizes + nonmonotone line search
- timing / comparison utilities

Usage example:
    python sbm_pgd_spg_compare.py --n 4000 --k 8 --blocks 8 --p_in 0.03 --p_out 0.003

Requirements:
    numpy
    scipy
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp


Array = np.ndarray


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def project_rows_nonnegative_unit_ball(D: Array) -> Array:
    """
    Project each row independently onto:
        {x : x >= 0, ||x||_2 <= 1}
    The projection is:
        x -> [x]_+ / max(1, ||[x]_+||_2)
    """
    X = np.maximum(D, 0.0)
    row_norms = np.linalg.norm(X, axis=1, keepdims=True)
    scale = np.maximum(1.0, row_norms)
    return X / scale


def make_sbm_adjacency(
    n: int,
    num_blocks: int,
    p_in: float,
    p_out: float,
    seed: int = 0,
    block_sizes: Optional[List[int]] = None,
) -> Tuple[sp.csr_matrix, Array]:
    """
    Generate an undirected SBM adjacency matrix without self-loops.

    Returns
    -------
    A : scipy.sparse.csr_matrix, shape (n, n)
        Symmetric sparse adjacency matrix
    labels : ndarray, shape (n,)
        Ground-truth block labels in {0, ..., num_blocks-1}
    """
    rng = np.random.default_rng(seed)

    if block_sizes is None:
        base = n // num_blocks
        rem = n % num_blocks
        block_sizes = [base + (1 if i < rem else 0) for i in range(num_blocks)]
    else:
        if sum(block_sizes) != n:
            raise ValueError("sum(block_sizes) must equal n")
        num_blocks = len(block_sizes)

    labels = np.empty(n, dtype=np.int64)
    start = 0
    for b, sz in enumerate(block_sizes):
        labels[start:start + sz] = b
        start += sz

    rows = []
    cols = []

    starts = np.cumsum([0] + block_sizes[:-1]).tolist()

    for bi in range(num_blocks):
        si = starts[bi]
        नि = block_sizes[bi]
        idx_i = np.arange(si, si + नि)

        # Within-block edges: only upper triangular, then symmetrize later
        m = नि
        if m >= 2 and p_in > 0.0:
            tri_r, tri_c = np.triu_indices(m, k=1)
            keep = rng.random(size=tri_r.size) < p_in
            rr = idx_i[tri_r[keep]]
            cc = idx_i[tri_c[keep]]
            rows.append(rr)
            cols.append(cc)

        # Between-block edges
        for bj in range(bi + 1, num_blocks):
            sj = starts[bj]
            nj = block_sizes[bj]
            idx_j = np.arange(sj, sj + nj)

            if p_out <= 0.0:
                continue

            # Bernoulli matrix for block pair (bi, bj)
            keep = rng.random(size=(नि, nj)) < p_out
            rr, cc = np.nonzero(keep)
            if rr.size > 0:
                rows.append(idx_i[rr])
                cols.append(idx_j[cc])

    if rows:
        row = np.concatenate(rows)
        col = np.concatenate(cols)
        data = np.ones_like(row, dtype=np.float64)
        A_upper = sp.coo_matrix((data, (row, col)), shape=(n, n))
        A = (A_upper + A_upper.T).tocsr()
    else:
        A = sp.csr_matrix((n, n), dtype=np.float64)

    A.setdiag(0.0)
    A.eliminate_zeros()
    return A, labels


def initialize_D(
    k: int,
    n: int,
    labels: Optional[Array] = None,
    seed: int = 0,
    mode: str = "random",
    noise: float = 0.05,
) -> Array:
    """
    Initialize D in shape (k, n), then project onto the feasible set.

    mode:
        - "random": random nonnegative initialization
        - "sbm": if labels are provided and k >= number of blocks, create
                 a block-informed warm start with small noise
    """
    rng = np.random.default_rng(seed)

    if mode == "sbm" and labels is not None:
        num_blocks = int(labels.max()) + 1
        D = np.zeros((k, n), dtype=np.float64)
        for j in range(n):
            b = int(labels[j])
            if b < k:
                D[b, j] = 1.0
        if noise > 0.0:
            D += noise * rng.random(size=(k, n))
    else:
        D = rng.random(size=(k, n))

    return project_rows_nonnegative_unit_ball(D)


# ---------------------------------------------------------------------
# Objective and gradient
# ---------------------------------------------------------------------

def objective_value(D: Array, A: sp.csr_matrix) -> float:
    """
    Compute
        f(D) = 0.5 * ||D^T D - A||_F^2
    efficiently using:
        ||D^T D - A||_F^2 = ||D D^T||_F^2 - 2 tr(D A D^T) + ||A||_F^2
    """
    G = D @ D.T  # (k, k)
    term1 = np.sum(G * G)

    DA = D @ A   # (k, n), works efficiently for sparse A
    term2 = np.sum(DA * D)

    # For a 0/1 adjacency matrix, ||A||_F^2 = number of nonzeros
    term3 = float(A.multiply(A).sum())

    return 0.5 * (term1 - 2.0 * term2 + term3)


def gradient(D: Array, A: sp.csr_matrix) -> Array:
    """
    Gradient for symmetric A:
        grad f(D) = 2 * ( (D D^T) D - D A )
    """
    G = D @ D.T
    DA = D @ A
    return 2.0 * (G @ D - DA)


# ---------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------

@dataclass
class SolverResult:
    D: Array
    objective_history: List[float]
    gradnorm_history: List[float]
    time_history: List[float]
    num_iters: int
    total_time: float
    status: str


def pgd_solver(
    A: sp.csr_matrix,
    D0: Array,
    max_iter: int = 500,
    step_size: float = 1e-2,
    tol: float = 1e-6,
    verbose: bool = True,
    record_every: int = 1,
) -> SolverResult:
    """
    Vanilla Projected Gradient Descent with fixed step size.
    """
    D = D0.copy()
    obj_hist: List[float] = []
    grad_hist: List[float] = []
    time_hist: List[float] = []

    t0 = time.perf_counter()

    for it in range(max_iter):
        G = gradient(D, A)
        gnorm = float(np.linalg.norm(G))

        if it % record_every == 0 or it == max_iter - 1:
            obj = objective_value(D, A)
            elapsed = time.perf_counter() - t0
            obj_hist.append(obj)
            grad_hist.append(gnorm)
            time_hist.append(elapsed)

            if verbose:
                print(
                    f"[PGD] iter={it:4d}  obj={obj:.6e}  "
                    f"||grad||={gnorm:.6e}  time={elapsed:.3f}s"
                )

        if gnorm <= tol:
            total_time = time.perf_counter() - t0
            return SolverResult(
                D=D,
                objective_history=obj_hist,
                gradnorm_history=grad_hist,
                time_history=time_hist,
                num_iters=it,
                total_time=total_time,
                status="converged_gradnorm",
            )

        D = project_rows_nonnegative_unit_ball(D - step_size * G)

    total_time = time.perf_counter() - t0
    return SolverResult(
        D=D,
        objective_history=obj_hist,
        gradnorm_history=grad_hist,
        time_history=time_hist,
        num_iters=max_iter,
        total_time=total_time,
        status="max_iter_reached",
    )


def spg_solver(
    A: sp.csr_matrix,
    D0: Array,
    max_iter: int = 500,
    alpha_init: float = 1e-2,
    alpha_min: float = 1e-10,
    alpha_max: float = 1e6,
    tol: float = 1e-6,
    memory: int = 10,
    sigma: float = 1e-4,
    beta: float = 0.5,
    max_linesearch: int = 30,
    verbose: bool = True,
    record_every: int = 1,
) -> SolverResult:
    """
    Spectral Projected Gradient (SPG) with BB step sizes and
    nonmonotone Armijo line search.

    This is often much faster than fixed-step PGD in practice.
    """
    D = D0.copy()
    obj_hist: List[float] = []
    grad_hist: List[float] = []
    time_hist: List[float] = []

    f = objective_value(D, A)
    G = gradient(D, A)
    alpha = alpha_init

    recent_f: List[float] = [f]

    t0 = time.perf_counter()

    for it in range(max_iter):
        gnorm = float(np.linalg.norm(G))

        if it % record_every == 0 or it == max_iter - 1:
            elapsed = time.perf_counter() - t0
            obj_hist.append(f)
            grad_hist.append(gnorm)
            time_hist.append(elapsed)

            if verbose:
                print(
                    f"[SPG] iter={it:4d}  obj={f:.6e}  "
                    f"||grad||={gnorm:.6e}  alpha={alpha:.3e}  time={elapsed:.3f}s"
                )

        if gnorm <= tol:
            total_time = time.perf_counter() - t0
            return SolverResult(
                D=D,
                objective_history=obj_hist,
                gradnorm_history=grad_hist,
                time_history=time_hist,
                num_iters=it,
                total_time=total_time,
                status="converged_gradnorm",
            )

        # Spectral projected direction
        D_trial = project_rows_nonnegative_unit_ball(D - alpha * G)
        P = D_trial - D
        directional = float(np.sum(G * P))

        # If projection killed the step, stop
        if np.linalg.norm(P) <= tol:
            total_time = time.perf_counter() - t0
            return SolverResult(
                D=D,
                objective_history=obj_hist,
                gradnorm_history=grad_hist,
                time_history=time_hist,
                num_iters=it,
                total_time=total_time,
                status="small_projected_step",
            )

        # Nonmonotone line search
        f_ref = max(recent_f[-memory:])
        accepted = False
        step = 1.0

        for _ in range(max_linesearch):
            D_new = D + step * P
            f_new = objective_value(D_new, A)

            if f_new <= f_ref + sigma * step * directional:
                accepted = True
                break
            step *= beta

        if not accepted:
            # Fallback: accept the projected point from current alpha
            D_new = D_trial
            f_new = objective_value(D_new, A)

        G_new = gradient(D_new, A)

        # BB step size update
        S = D_new - D
        Y = G_new - G
        sy = float(np.sum(S * Y))

        if sy > 1e-16:
            alpha = float(np.sum(S * S) / sy)
            alpha = min(max(alpha, alpha_min), alpha_max)
        else:
            alpha = alpha_init

        D = D_new
        G = G_new
        f = f_new
        recent_f.append(f)

    total_time = time.perf_counter() - t0
    return SolverResult(
        D=D,
        objective_history=obj_hist,
        gradnorm_history=grad_hist,
        time_history=time_hist,
        num_iters=max_iter,
        total_time=total_time,
        status="max_iter_reached",
    )


# ---------------------------------------------------------------------
# Comparison / reporting
# ---------------------------------------------------------------------

def summarize_result(name: str, result: SolverResult) -> Dict[str, float]:
    return {
        "solver": name,
        "iters": result.num_iters,
        "time_sec": result.total_time,
        "final_obj": result.objective_history[-1] if result.objective_history else np.nan,
        "final_gradnorm": result.gradnorm_history[-1] if result.gradnorm_history else np.nan,
    }


def print_summary_table(rows: List[Dict[str, float]]) -> None:
    headers = ["solver", "iters", "time_sec", "final_obj", "final_gradnorm"]
    line = "-" * 84
    print("\n" + line)
    print(
        f"{headers[0]:<10s} {headers[1]:>8s} {headers[2]:>14s} "
        f"{headers[3]:>22s} {headers[4]:>22s}"
    )
    print(line)
    for r in rows:
        print(
            f"{r['solver']:<10s} {int(r['iters']):>8d} {r['time_sec']:>14.4f} "
            f"{r['final_obj']:>22.6e} {r['final_gradnorm']:>22.6e}"
        )
    print(line + "\n")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PGD and SPG for sparse SBM adjacency factorization."
    )

    # Graph
    parser.add_argument("--n", type=int, default=20000, help="number of nodes")
    parser.add_argument("--blocks", type=int, default=5, help="number of SBM blocks")
    parser.add_argument("--p_in", type=float, default=0.7, help="within-block edge probability")
    parser.add_argument("--p_out", type=float, default=0.2, help="between-block edge probability")
    parser.add_argument("--graph_seed", type=int, default=0, help="random seed for graph generation")

    # Model
    parser.add_argument("--k", type=int, default=5, help="number of rows of D")
    parser.add_argument("--init", type=str, default="random", choices=["random", "sbm"],
                        help="initialization mode")

    # Solver
    parser.add_argument("--max_iter", type=int, default=40, help="maximum iterations")
    parser.add_argument("--tol", type=float, default=1e-6, help="gradient norm tolerance")
    parser.add_argument("--pgd_step", type=float, default=1e-2, help="fixed PGD step size")
    parser.add_argument("--spg_alpha", type=float, default=1e-2, help="initial SPG alpha")
    parser.add_argument("--memory", type=int, default=10, help="SPG nonmonotone memory")
    parser.add_argument("--record_every", type=int, default=1, help="record metrics every N iterations")
    parser.add_argument("--quiet", action="store_true", help="suppress per-iteration printing")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Generating SBM graph...")
    A, labels = make_sbm_adjacency(
        n=args.n,
        num_blocks=args.blocks,
        p_in=args.p_in,
        p_out=args.p_out,
        seed=args.graph_seed,
    )
    density = A.nnz / (args.n * args.n)

    print(f"Graph generated: n={args.n}, nnz={A.nnz}, density={density:.6e}")
    print(f"k={args.k}, blocks={args.blocks}, p_in={args.p_in}, p_out={args.p_out}")

    D0 = initialize_D(
        k=args.k,
        n=args.n,
        labels=labels,
        seed=args.graph_seed + 1,
        mode=args.init,
    )

    # Use the same initialization for both solvers
    print("\nRunning PGD...")
    pgd_result = pgd_solver(
        A=A,
        D0=D0,
        max_iter=args.max_iter,
        step_size=args.pgd_step,
        tol=args.tol,
        verbose=not args.quiet,
        record_every=args.record_every,
    )

    print("\nRunning SPG...")
    spg_result = spg_solver(
        A=A,
        D0=D0,
        max_iter=args.max_iter,
        alpha_init=args.spg_alpha,
        tol=args.tol,
        memory=args.memory,
        verbose=not args.quiet,
        record_every=args.record_every,
    )

    rows = [
        summarize_result("PGD", pgd_result),
        summarize_result("SPG", spg_result),
    ]
    print_summary_table(rows)

    print("Statuses:")
    print(f"  PGD: {pgd_result.status}")
    print(f"  SPG: {spg_result.status}")


if __name__ == "__main__":
    main()
