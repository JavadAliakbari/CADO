import numpy as np
from tqdm import tqdm

from utils.utils import *
from utils.algorithm_utils import *
from scipy.optimize import root


def F_system(x, eps, c_new):
    """
    Returns the vector F(t) of dimension n, where
       F_v(t) = sum_{i=1}^k [ c_new[i] * t_v^2 * eps[i,v]^2 / G_i(t ) ] - 1
    and G_i(t) = sum_{u=1}^n t_u^2 * eps[i,u]^2.

    Parameters
    ----------
    t      : (n,) array, the unknowns t_v (assumed > 0 ideally).
    eps    : (k, n) array, original vectors eps_i in R^n (row-wise).
    c_new  : (k,) array, new coefficients that sum to n.

    Returns
    -------
    F_val  : (n,) array, the residuals F_v(t).
    """
    n, k = eps.shape
    t = np.exp(x)
    # G_i(t) for each i
    # shape: (k,) => each G_i is sum_{u=1..n} t_u^2 * eps[i,u]^2
    G = np.sum(t[:, None] * eps, axis=0)

    F_val = np.zeros(n, dtype=np.float64)
    F_val = np.sum(c_new[None, :] * eps / G[None, :], axis=1) - 1
    return F_val


def solve_t_v(eps, c_new, x_init=None, method="hybr", **root_kwargs):
    """
    Solve for x_v in R, with t_v = exp(x_v) > 0, ensuring positivity.

    Uses scipy.optimize.root to solve F_system_in_log_space(x)=0.

    Parameters
    ----------
    eps        : (k, n) array
                 Original vectors eps_i in R^n (row i is eps_i).
    c_new      : (k,) array
                 New coefficients that sum to n.
    x_init     : (n,) array or None
                 Initial guess in log-space. If None, uses  log(1.0).
    method     : str
                 The method passed to `scipy.optimize.root` (e.g. 'hybr', 'lm', 'broyden1', 'krylov').
    **root_kwargs:
                 Additional keyword arguments for `scipy.optimize.root`.

    Returns
    -------
    t_sol      : (n,) array
                 The positive solution for t_v = exp(x_v).
    success    : bool
                 True if the solver succeeded.
    res        : OptimizeResult
                 Full result object from `root`.
    """
    k, n = eps.shape

    # If no initial guess, start with all zero in log-space => t_v=exp(0)=1
    if x_init is None:
        x_init = np.zeros(n, dtype=np.float64)

    # Solve for x via a root-finding method
    res = root(F_system, x_init, args=(eps, c_new), method=method, **root_kwargs)
    success = res.success

    # Convert x -> t
    t_sol = np.exp(res.x)

    return t_sol, success, res


def example_test():
    """
    Demonstrate solving the system with Newton or quasi-Newton solver on a small example.
    """
    # -- Setup: n=3, k=3
    n = 900
    k = 3

    # Original eps: use standard basis e1=(1,0,0), e2=(0,1,0), e3=(0,0,1)
    piv = np.random.randn(n, k)  # shape = (3,3)
    piv = project_to_simplex(piv)  # project to simplex
    eps, c_original = Piv2EC(piv)  # E is (n,k), c is (k,)
    eps = eps**2

    # Confirm original constraints are satisfied
    for v in range(3):
        val = np.sum(c_original * eps[v, :])
        assert np.isclose(val, 1.0), "Original constraint not satisfied."
    print("Original constraints satisfied for (c, eps).")

    # New c that sums to n=3
    # -- Now define new c:
    c_new = project_to_simplex(
        c_original - 0.1 * np.random.rand(k), n
    )  # random perturbation
    assert np.isclose(np.sum(c_new), n)

    # Solve with log-space approach => t_v = exp(x_v) > 0
    # Provide an initial guess x_init that might be near zero:
    x_init = np.log(np.ones(n))  # i.e. t=1 initially
    t_sol, success, res = solve_t_v(eps, c_new, x_init=x_init, method="hybr")

    if not success:
        print("Warning: solver did not report success. Info:", res.message)
    else:
        print("Solver converged successfully.")

    print("Solution t:", t_sol)

    # Construct tilde_eps
    tilde_eps = construct_tilde_epsilon(eps, t_sol)

    # -- Verify the new constraints sum_i c_new[i] * tilde_eps[i,v]^2 = 1 for each v
    check_vals = np.sum(c_new[None, :] * tilde_eps, axis=1)
    print("Check sum_i c_new[i]* tilde_eps[i,v]^2 for each v:", np.round(check_vals, 2))

    # -- Print final result
    print("\nFinal tilde_eps =\n", eps)
    print("\nFinal tilde_c =\n", c_original)
    print("\nFinal tilde_eps =\n", tilde_eps)
    print("\nFinal tilde_c =\n", c_new)
    return t_sol, tilde_eps


if __name__ == "__main__":
    t_solution, tilde_eps_solution = example_test()
    print("\nFinal tilde_eps:\n", tilde_eps_solution)
