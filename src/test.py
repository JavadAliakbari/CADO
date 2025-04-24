import numpy as np
from tqdm import tqdm

from utils.utils import *
from utils.algorithm_utils import *


def test_example():
    """
    Small test example in n=3 dimensions with k=3 vectors, each c_i = 1 initially.
    eps_i are chosen to be the standard basis vectors in R^3, which trivially satisfy
       sum_i c_i * eps_{i,v}^2 = 1 for each coordinate v,
    because each column is 'activated' by exactly one eps_i.
    Then we set new c's (which still sum to 3) and find the tilde_epsilon via the iteration.
    Finally, we verify the constraints.
    """
    # -- Setup: n=3, k=3
    n = 8
    k = 3

    # Original eps: use standard basis e1=(1,0,0), e2=(0,1,0), e3=(0,0,1)
    piv = np.random.randn(n, k)  # shape = (3,3)
    piv = project_to_simplex(piv)  # project to simplex
    eps, c_original = Piv2EC(piv)  # E is (n,k), c is (k,)
    eps = eps**2
    # But let's reorder so it's shape (k,n) = (3,3) with rows as vectors
    # It's actually fine as is: row0=(1,0,0), row1=(0,1,0), row2=(0,0,1)

    # Original c_i all 1 => sums to 3 => matches n=3
    # c_original = np.array([1.0, 1.0, 1.0])

    # Check that sum_i c_i eps_{i,v}^2 = 1 for each v (just a quick sanity check)
    for v in range(n):
        val = np.sum(c_original * eps[v, :])
        assert np.isclose(val, 1.0), f"Original constraint not satisfied at v={v}"
    print("Original (c, eps) constraints are satisfied as expected.")

    # -- Now define new c:
    c_new = project_to_simplex(
        c_original - 0.1 * np.random.rand(k), n
    )  # random perturbation
    # c_new = np.array([2.0, 0.5, 0.5])
    assert np.isclose(np.sum(c_new), n), "c_new must sum to n=3."
    print(f"New c: {c_new}, sum={c_new.sum():.2f}")

    # -- Solve for t:
    t = find_diagonal_scaling(eps, c_new, max_iter=2000, tol=1e-14)
    print(f"Solved diagonal scaling t = {t}")

    # -- Construct tilde_eps:
    tilde_eps = construct_tilde_epsilon(eps, t)

    # -- Verify unit-length:
    norms = np.sum(tilde_eps, axis=0)
    print("Norms of each tilde_eps[i]:", norms)

    # -- Verify the new constraints sum_i c_new[i] * tilde_eps[i,v]^2 = 1 for each v
    check_vals = np.sum(c_new[None, :] * tilde_eps, axis=1)
    print("Check sum_i c_new[i]* tilde_eps[i,v]^2 for each v:", np.round(check_vals, 2))

    # -- Print final result
    print("\nFinal tilde_eps =\n", eps)
    print("\nFinal tilde_c =\n", c_original)
    print("\nFinal tilde_eps =\n", tilde_eps)
    print("\nFinal tilde_c =\n", c_new)
    return t, tilde_eps


if __name__ == "__main__":
    t_solution, tilde_eps_solution = test_example()
