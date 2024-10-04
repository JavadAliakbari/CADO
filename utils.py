import os
from dotenv import load_dotenv
import numpy as np

from config_parser import Config

load_dotenv()
config_path = os.environ.get("CONFIGPATH")
config = Config(config_path)


def matrix_dot_product(A, B):
    return np.sum(A * B)


def create_Ri_symmetric_matrix_with_constrained_eigenvalues(
    dimension, rho_minus, rho_plus, noise_level=0.0, include_extremes=True
):
    """
    Create a symmetric matrix with eigenvalues between rho_minus and rho_plus.

    Parameters:
    dimension (int): Size of the matrix (n x n).
    rho_minus (float): Lower bound of the eigenvalues.
    rho_plus  (float): Upper bound of the eigenvalues.
    noise_level (float): The scale of the noise to be added (small value to slightly perturb).
    include_extremes (bool): If True, include rho_minus and rho_plus as exact eigenvalues.

    Returns:
    np.ndarray: The symmetric matrix with eigenvalues in the range [rho_minus, rho_plus].
    """
    if include_extremes:
        if dimension < 2:
            raise ValueError(
                "Dimension must be at least 2 to include both rho_minus and rho_plus as eigenvalues."
            )
        # Generate the remaining dimension - 2 eigenvalues uniformly between rho_minus and rho_plus
        additional_eigenvalues = np.random.uniform(rho_minus, rho_plus, dimension - 2)
        # Combine all eigenvalues
        all_eigenvalues = np.concatenate(
            ([rho_minus, rho_plus], additional_eigenvalues)
        )
    else:
        # Generate dimension eigenvalues uniformly distributed between rho_minus and rho_plus
        all_eigenvalues = np.random.uniform(rho_minus, rho_plus, dimension)

    # Create rho_minus random orthogonal matrix Q using QR decomposition
    Q, _ = np.linalg.qr(np.random.randn(dimension, dimension))

    # Create symmetric noise matrix
    noise_matrix = noise_level * np.random.randn(dimension, dimension)
    symmetric_noise = (noise_matrix + noise_matrix.T) / 2

    # Construct the symmetric matrix using A = Q * Lambda * Q.T
    Lambda = np.diag(all_eigenvalues)  # Diagonal matrix of eigenvalues
    symmetric_matrix = Q @ Lambda @ Q.T + symmetric_noise

    return symmetric_matrix


def create_multiple_noisy_probability_vectors(
    count, dimension, noise_level=0.0, manual_probabilities=None
):
    """
    Create multiple noisy probability vectors where each starts with a unique index set to 1.

    Parameters:
    count (int): Number of probability vectors to generate.(In our problem: the number of clusters)
    dimension (int): Length of each probability vector.(In our probelm: the number of classes)
    noise_level (float): The scale of the noise to add.
    manual_probabilities (list of lists or None): If provided, use this as the probability vectors.

    Returns:
    np.ndarray: An array containing the noisy probability vectors.
    """

    # If manual probabilities are provided, return them as a numpy array
    if manual_probabilities is not None:
        # Check that manual probabilities have the correct shape
        manual_probabilities = np.array(manual_probabilities)
        if manual_probabilities.shape != (count, dimension):
            raise ValueError(
                f"manual_probabilities must have shape ({count}, {dimension})"
            )

        # Ensure that each row sums to 1 to confirm they are valid probability vectors
        if not np.allclose(np.sum(manual_probabilities, axis=1), 1):
            raise ValueError("Each probability vector must sum to 1")

        return manual_probabilities

    # Check that count does not exceed dimension
    if count > dimension:
        raise ValueError(
            "The number of vectors 'count' cannot exceed the dimension size."
        )

    probability_vectors = []

    # Step 1: Create 'count' one-hot vectors at unique indices
    for i in range(count):
        # Initialize a one-hot vector with 1 at index i
        probability_vector = np.zeros(dimension)
        probability_vector[i] = 1

        # Step 2: Add random noise to the vector
        noise = np.random.uniform(0, noise_level, dimension)
        noisy_vector = probability_vector + noise

        # Step 3: Normalize to make it a probability vector
        noisy_probability_vector = noisy_vector / np.sum(noisy_vector)

        probability_vectors.append(noisy_probability_vector)

    return np.array(probability_vectors)


def project_to_set_S(matrix, eigen_min, eigen_max):
    """
    Project a matrix onto the set of symmetric matrices with eigenvalues
    constrained between eigen_min and eigen_max.

    Parameters:
    matrix (np.ndarray): The input matrix to be projected.
    eigen_min (float): The minimum eigenvalue bound.
    eigen_max (float): The maximum eigenvalue bound.

    Returns:
    np.ndarray: The projected symmetric matrix with eigenvalues in the specified range.
    """
    # Step 1: Symmetrize the matrix
    sym_matrix = (matrix + matrix.T) / 2

    # Step 2: Perform eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(sym_matrix)

    # Step 3: Clip the eigenvalues to the specified range [eigen_min, eigen_max]
    clipped_eigenvalues = np.clip(eigenvalues, eigen_min, eigen_max)
    # mid_value = eigen_min + (eigen_max - eigen_min) / 2
    # cond1 = (eigenvalues < mid_value) & (eigenvalues > eigen_min)
    # cond2 = (eigenvalues > mid_value) & (eigenvalues < eigen_max)
    # eigenvalues[cond1] = eigen_min
    # eigenvalues[cond2] = eigen_max
    # Step 4: Reconstruct the matrix using the clipped eigenvalues
    projected_matrix = eigenvectors @ np.diag(clipped_eigenvalues) @ eigenvectors.T

    return projected_matrix


def project_to_simplex(y, a):
    u = np.flip(np.sort(y))
    k = 0
    while k <= len(y) - 1:
        tau = (np.sum(u[: k + 1]) - a) / (k + 1)
        if tau <= u[k]:
            k += 1
        else:
            break
    tau = (np.sum(u[:k]) - a) / (k)
    out = y - tau
    out[out < 0] = 0
    return out


def one_hot_max(vector):
    # Step 1: Find the index of the maximum value
    max_index = np.argmax(vector)

    # Step 2: Create a zero vector of the same length as the input vector
    one_hot_vector = np.zeros_like(vector)

    # Step 3: Set the element at max_index to 1
    one_hot_vector[max_index] = 1

    return one_hot_vector
