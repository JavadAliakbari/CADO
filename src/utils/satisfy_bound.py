import numpy as np


def compute_L(E, lambdas):
    """
    Compute L = sum_i lambda_i * (epsilon_i)(epsilon_i)^T.

    Parameters:
        E       : numpy array of shape (n, k), each column is a unit vector.
        lambdas : numpy array of shape (k,) with the weights.

    Returns:
        L       : numpy array of shape (n, n).
    """
    return E @ np.diag(lambdas) @ E.T


def clip_matrix(L):
    """
    Clip all entries of matrix L to the interval [0, 1].

    Parameters:
        L : numpy array.

    Returns:
        L_clip : numpy array with every entry in [0, 1].
    """
    return np.clip(L, 0, 1)


def satisfy_bound(atoms, coefs):
    E = np.array([atom.e for atom in atoms]).T
    lambdas = np.array(coefs)
    adjusted_E = adjust_vectors_factorization(E, lambdas)

    for i, atom in enumerate(atoms):
        atom.e = adjusted_E[:, i]

    return atoms


def adjust_vectors_factorization(E_orig, lambdas):
    """
    Adjust the vectors using the second approach:
      1. Compute L from the original vectors and lambdas.
      2. Clip L to get L_clip.
      3. Factorize L_clip (via eigen-decomposition) to get new unit vectors.
      4. Optionally, align the new vectors with the original ones via orthogonal Procrustes.

    Parameters:
        E_orig   : numpy array of shape (n, k) with original unit vectors as columns.
        lambdas  : numpy array of shape (k,) with given weights.

    Returns:
        E_aligned : numpy array of shape (n, k) containing the new unit vectors.
    """
    n, k = E_orig.shape

    # Step 1: Compute L from original vectors.
    L_orig = compute_L(E_orig, lambdas)

    # Step 2: Clip L to the interval [0, 1].
    L_clip = clip_matrix(L_orig)

    # Step 3: Factorize L_clip.
    # Since L_clip is symmetric, we use an eigen-decomposition.
    # Note: np.linalg.eigh returns eigenvalues in ascending order.
    eigenvals, eigenvecs = np.linalg.eigh(L_clip)

    # Sort eigenvalues and corresponding eigenvectors in descending order.
    idx_desc = np.argsort(eigenvals)[::-1]
    eigenvals_sorted = eigenvals[idx_desc]
    eigenvecs_sorted = eigenvecs[:, idx_desc]

    # We take the top k eigenvectors.
    E_new = eigenvecs_sorted[:, :k]
    # These vectors are already unit norm since they come from an orthogonal decomposition.

    # Now, our factorization gives:
    #     L_clip ≈ sum_{i=1}^{k} d_i (v_i)(v_i)^T,
    # where d_i are the eigenvalues and v_i the eigenvectors (columns of E_new).
    # We would like a representation of the form
    #     L_clip ≈ sum_{i=1}^{k} lambda_i (epsilon'_i)(epsilon'_i)^T.
    #
    # In many cases the given lambda_i might differ from the eigenvalues d_i.
    # Here we simply use the eigenvectors as our new directions.
    # (If matching the weights is critical, you might consider further scaling or an iterative procedure.)

    # Step 4: Align the new vectors with the original ones to reduce rotation.
    # We solve the orthogonal Procrustes problem:
    #    min_Q || E_new Q - E_orig ||_F,  with Q orthogonal.
    # The solution is given by the SVD of E_new^T E_orig.
    # M = E_new.T @ E_orig  # shape (k, k)
    # U, _, Vt = np.linalg.svd(M)
    # Q = Vt.T @ U.T
    # E_aligned = E_new @ Q

    # return E_aligned
    return E_new


# Example usage:
if __name__ == "__main__":
    np.random.seed(0)
    n = 5  # dimension of each vector
    k = 3  # number of vectors

    # Generate k random unit vectors of dimension n.
    E_orig = np.random.randn(n, k)
    for i in range(k):
        E_orig[:, i] /= np.linalg.norm(E_orig[:, i])

    # Define example weights.
    lambdas = np.array([1.0, 0.8, 1.2])

    # Apply the second approach to adjust the vectors.
    E_adjusted = adjust_vectors_factorization(E_orig, lambdas)

    # Print results.
    print("Original vectors (columns):")
    print(E_orig)
    print("\nAdjusted vectors (columns):")
    print(E_adjusted)

    # For comparison, you can also compute the original and adjusted L matrices.
    L_orig = compute_L(E_orig, lambdas)
    L_adjusted = compute_L(E_adjusted, lambdas)
    print("\nOriginal L:")
    print(L_orig)
    print("\nAdjusted L (from clipped and factorized vectors):")
    print(L_adjusted)
