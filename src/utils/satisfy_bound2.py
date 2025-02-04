from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm


def compute_L(E, lambdas):
    """
    Compute L = sum_i lambda_i * (epsilon_i)(epsilon_i)^T.

    Parameters:
        E       : numpy array of shape (n, k), with each column a unit vector.
        lambdas : numpy array of shape (k,) containing the weights.

    Returns:
        L       : numpy array of shape (n, n).
    """
    return E @ np.diag(lambdas) @ E.T


def clip_matrix(L):
    """
    Clip the entries of matrix L to be between 0 and 1.

    Parameters:
        L : numpy array.

    Returns:
        L_clip : numpy array with all entries in [0, 1].
    """
    return np.clip(L, 0, 1)


def satisfy_bound(atoms, coefs):
    E = np.array([atom.e for atom in atoms]).T
    lambdas = np.array(coefs)
    adjusted_E = adjust_vectors(E, lambdas)

    for i, atom in enumerate(atoms):
        atom.e = adjusted_E[:, i]

    return atoms


def adjust_vectors(E_orig, lambdas, alpha=0.001, max_iter=10000, tol=1e-8):
    """
    Adjust the set of unit vectors so that the matrix L' = sum_i lambda_i * epsilon_i' (epsilon_i')^T
    is as close as possible to a version of L (formed from the original vectors) whose entries have been clipped
    to [0, 1]. Also, the adjustment tries to keep the new vectors close to the originals.

    Parameters:
        E_orig   : numpy array of shape (n, k) with original unit vectors as columns.
        lambdas  : numpy array of shape (k,) with the given weights.
        alpha    : step size for gradient descent.
        max_iter : maximum number of iterations.
        tol      : tolerance for convergence (change in objective).

    Returns:
        E_adjusted : numpy array of shape (n, k) with the new unit vectors (after Procrustes alignment).
    """
    n, k = E_orig.shape

    # 1. Compute the original matrix L and then clip it.
    L_orig = compute_L(E_orig, lambdas)
    L_clip = clip_matrix(L_orig)

    # 2. Initialize new vectors with the original ones.
    E = E_orig.copy()

    prev_obj = None
    # bar = tqdm(
    #     total=max_iter, desc="Gradient descent iterations", position=0, leave=False
    # )

    f_values = []

    # 3. Iteratively update the vectors.
    for it in range(max_iter):
        # Build the current L from updated vectors.
        L_current = compute_L(E, lambdas)
        diff = L_clip - L_current
        # Compute the squared Frobenius norm of the difference.
        obj = np.linalg.norm(diff, "fro") ** 2

        # Check for convergence.
        if prev_obj is not None and abs(obj - prev_obj) < tol:
            # print(
            #     f"Gradient descent converged after {it} iterations. Objective: {obj:.6e}"
            # )
            break
        prev_obj = obj

        # Update each vector.
        for i in range(k):
            # Compute a gradient for epsilon_i.
            # The (approximate) gradient is: grad = -4 * lambda_i * (L_clip - L_current) @ epsilon_i.
            grad = -4 * lambdas[i] * (diff @ E[:, i])
            # Gradient descent update.
            E[:, i] = E[:, i] - alpha * grad
            # Project back onto the unit sphere.
            E[:, i] = E[:, i] / np.linalg.norm(E[:, i])

        # bar.set_postfix({"Objective": obj})
        # bar.update(1)
        f_values.append(obj)

    # 4. Use Orthogonal Procrustes to align E to E_orig.
    # We wish to find an orthogonal Q such that E @ Q is as close as possible to E_orig.
    M = E.T @ E_orig  # This is a k x k matrix.
    U, _, Vt = np.linalg.svd(M)
    Q = Vt.T @ U.T  # This is the optimal rotation matrix.
    E_aligned = E @ Q

    return E_aligned


# Example usage:
if __name__ == "__main__":
    np.random.seed(0)
    n = 5  # Dimension of each vector.
    k = 3  # Number of vectors.

    # Generate random unit vectors.
    E_orig = np.random.randn(n, k)
    for i in range(k):
        E_orig[:, i] /= np.linalg.norm(E_orig[:, i])

    # Define some example weights.
    lambdas = np.array([1.0, 0.8, 1.2])

    # Adjust the vectors.
    E_adjusted, f_values = adjust_vectors(
        E_orig, lambdas, alpha=0.001, max_iter=10000, tol=1e-8
    )

    print("Original vectors (columns):")
    print(E_orig)
    print("\nAdjusted vectors (columns):")
    print(E_adjusted)
    print("\nOriginal matrix L:", E_orig @ np.diag(lambdas) @ E_orig.T)
    print("\nAdjusted matrix L:", E_adjusted @ np.diag(lambdas) @ E_adjusted.T)

    plt.figure()
    plt.plot(f_values)
    plt.xlabel("Iteration")
    plt.ylabel("Objective")
    plt.title("Objective value vs. iteration")
    plt.show()
