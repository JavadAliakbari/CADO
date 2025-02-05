from datetime import datetime
import json
import os
import pickle as pkl
import random

import cvxpy as cp
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from utils.config_parser import Config
from utils.logger import getLOGGER

seed = 4
random.seed(seed)
np.random.seed(seed)

load_dotenv()
config_path = os.environ.get("CONFIGPATH")
config = Config(config_path)

plot = os.environ.get("PLOT") == "True"

plt.rcParams["figure.figsize"] = [9, 6]
plt.rcParams["figure.dpi"] = 250  # 200 e.g. is really fine, but slower
plt.rcParams.update({"figure.max_open_warning": 0})
plt.rcParams["font.size"] = 10


now = datetime.now().strftime("%Y%m%d_%H%M%S")
result_path = os.environ.get("RESULTPATH")
save_path = f"{result_path}/{now}/"

# Define custom colors for 0 and 1 (e.g., 0 -> blue, 1 -> red)
colors = [
    (0, "white"),
    (1, "blue"),
]  # Replace 'blue' and 'red' with your desired colors

# Create a custom colormap using the colors
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)


def matrix_dot_product(A, B):
    """
    Compute the dot product of two tensors A and B.

    Parameters:
    A (numpy.ndarray): First input tensor.
    B (numpy.ndarray): Second input tensor.

    Returns:
    float: The dot product of tensors A and B.
    """
    return np.sum(A * B)


# Function to invert a matrix
def invert_matrix(matrix):
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return np.eye(matrix.shape[0])


def calc_accuracy(pred_y, y):
    """Calculate accuracy."""
    return sum(pred_y == y) / len(y)


def create_Ri_symmetric_matrix_with_constrained_eigenvalues(
    dimension, rho_minus, rho_plus, noise_level=0.0
):
    """
    Create a symmetric matrix with eigenvalues between rho_minus and rho_plus.

    Parameters:
    dimension (int): Size of the matrix (n x n).
    rho_minus (float): Lower bound of the eigenvalues.
    rho_plus  (float): Upper bound of the eigenvalues.
    noise_level (float): The scale of the noise to be added (small value to slightly perturb).

    Returns:
    np.ndarray: The symmetric matrix with eigenvalues in the range [rho_minus, rho_plus].
    """
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


def create_Ri_symmetric_matrix_with_constrained_eigenvalues2(
    dimension, rho_minus, rho_plus, num_signal_features
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
    # signal_eigenvalues = np.random.uniform(rho_plus, 2 * rho_plus, 2)
    signal_eigenvalues = np.array(num_signal_features * [rho_plus])
    noise_eigenvalues = np.array([rho_minus] * (dimension - num_signal_features))
    # noise_eigenvalues = np.random.uniform(0, rho_minus, dimension - 2)
    all_eigenvalues = np.concatenate((signal_eigenvalues, noise_eigenvalues))

    # Create rho_minus random orthogonal matrix Q using QR decomposition
    # Q, _ = np.linalg.qr(np.random.randn(dimension, dimension))
    idx = np.random.permutation(dimension)
    Q = np.zeros((dimension, dimension), dtype=float)
    Q[np.arange(dimension), idx] = 1

    # Construct the symmetric matrix using A = Q * Lambda * Q.T
    Lambda = np.diag(all_eigenvalues)  # Diagonal matrix of eigenvalues
    symmetric_matrix = Q @ Lambda @ Q.T

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


def project_to_unit_vector(vector):
    """
    Project a vector onto the unit vector.

    Parameters:
    vector (np.ndarray): The input vector to be projected.

    Returns:
    np.ndarray: The projected unit vector.
    """
    return vector / np.linalg.norm(vector)


def project_to_psd_cone(matrix, rho_minus, rho_plus):
    """
    Project a matrix onto the positive semidefinite cone.

    Parameters:
    matrix (np.ndarray): The input matrix to be projected.

    Returns:
    np.ndarray: The projected positive semidefinite matrix.
    """
    # Perform eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    # Clip the eigenvalues to ensure they are non-negative
    clipped_eigenvalues = np.clip(eigenvalues, rho_minus, rho_plus)

    # Reconstruct the matrix using the clipped eigenvalues
    projected_matrix = eigenvectors @ np.diag(clipped_eigenvalues) @ eigenvectors.T

    return projected_matrix


def project_to_simplex(y, a=1):
    return project_to_simplex2(y, a)
    # return y
    """
    Efficiently project a vector y onto the simplex defined by sum(x) = a and x >= 0.

    Parameters:
    y (np.ndarray): The input vector to be projected.
    a (float): The sum constraint for the simplex.

    Returns:
    np.ndarray: The projected vector.
    """
    if a < 0:
        raise ValueError("The sum constraint 'a' must be non-negative.")

    y[y < 0] = 0
    y *= a * np.sum(y) ** -1
    return y


def project_inside_simplex2(λ0, E, a):
    n, c = E.shape
    ones = a * np.ones(n)  # RHS of inequality constraints

    λ_var = cp.Variable(c)
    # Define the objective function
    objective = cp.Minimize(0.5 * cp.sum_squares(λ_var - λ0))

    # Define constraints
    constraints = [E @ λ_var <= ones]

    # Solve the optimization problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    return λ_var.value


def project_inside_simplex(y, a):
    if y.ndim == 1:
        return _project_vector_inside_simplex(y, a)
    elif y.ndim == 2:
        return np.apply_along_axis(_project_vector_inside_simplex, 1, y, a)
    else:
        raise ValueError("Input array must be 1D or 2D")


def project_to_simplex2(y, a):
    if y.ndim == 1:
        return _project_vector_to_simplex(y, a)
    elif y.ndim == 2:
        return np.apply_along_axis(_project_vector_to_simplex, 1, y, a)
    else:
        raise ValueError("Input array must be 1D or 2D")


def _project_vector_to_simplex(y, a):
    # Step 1: Sort y in descending order (largest to smallest)
    u = np.sort(y)[::-1]

    # Step 2: Find the rho value
    # rho is the largest index where the condition u * (index + 1) > cssv holds
    cssv = np.cumsum(u) - a
    rho = np.nonzero(u * np.arange(1, len(y) + 1) > cssv)[0][-1]
    # Step 3: Compute the theta value, which is used to shift the elements of y such that their sum equals 'a'
    theta = cssv[rho] / (rho + 1)

    # Step 4: Compute the projection to ensure the vector lies within the simplex defined by the sum constraint 'a'
    return np.maximum(y - theta, 0)


def _project_vector_inside_simplex(y, a):
    if sum(y) <= a and np.all(y >= 0):
        return y
    # Step 1: Sort y in descending order (largest to smallest)
    u = np.sort(y)[::-1]

    # Step 2: Find the rho value
    # rho is the largest index where the condition u * (index + 1) > cssv holds
    cssv = np.cumsum(u) - a
    rho = np.nonzero(u * np.arange(1, len(y) + 1) > cssv)[0][-1]
    # Step 3: Compute the theta value, which is used to shift the elements of y such that their sum equals 'a'
    theta = cssv[rho] / (rho + 1)

    # Step 4: Compute the projection to ensure the vector lies within the simplex defined by the sum constraint 'a'
    return np.maximum(y - theta, 0)


def project_to_bound(a, a_min=0, a_max=1):
    # a_mean = (a_max + a_min) / 2
    # a[a > a_mean] = 1
    # a[a < a_mean] = 0
    # return a
    return np.clip(a, a_min, a_max)


def one_hot_max(vector):
    # Step 1: Find the index of the maximum value
    max_index = np.argmax(vector)

    # Step 2: Create a zero vector of the same length as the input vector
    one_hot_vector = np.zeros_like(vector)

    # Step 3: Set the element at max_index to 1
    one_hot_vector[max_index] = 1

    return one_hot_vector


def plot_adj(A, name="L"):
    os.makedirs(save_path, exist_ok=True)
    # Display the array as an image
    plt.figure()
    plt.imshow(A, cmap=custom_cmap)

    # Remove axis for a clean output
    plt.axis("off")

    # Save the image with a specific DPI (dots per inch) to control the output size
    plt.savefig(f"{save_path}{name}.png", dpi=300, bbox_inches="tight", pad_inches=0)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")

    # x, y = np.meshgrid(range(A.shape[0]), range(A.shape[1]))
    # # z = np.zeros_like(A)

    # ax.plot_surface(
    #     x,
    #     y,
    #     A,
    #     # facecolors=plt.cm.viridis(A),
    #     # rstride=1,
    #     # cstride=1,
    #     # antialiased=False,
    #     # shade=False,
    # )

    # # ax.set_zlim(0, 1)
    # # ax.set_xlabel("X axis")
    # # ax.set_ylabel("Y axis")
    # # ax.set_zlabel("Z axis")

    # plt.savefig(f"{save_path}{name}.png", dpi=300, bbox_inches="tight", pad_inches=0)
    # # plt.show()


def log_results(atoms, coefs, graph, results, save_path, terminal=True):
    os.makedirs(save_path, exist_ok=True)
    LOGGER = getLOGGER(
        name=f"{now}",
        log_on_file=True,
        save_path=save_path,
        terminal=terminal,
    )
    sig = atoms.calc_sig(coefs)
    results = {
        f"atom {id}": {
            "e": str(np.round(atom.e, 2)),
            "b": str(np.round(atom.b, 2)),
            "R": str(np.round(np.linalg.eigvalsh(atom.R), 2)),
            "λ": str(np.round(c, 2)),
        }
        for id, (c, atom) in enumerate(zip(coefs, atoms))
    }

    mask = np.argmax(sig.pi, axis=1) == graph.y
    # Replace True with ✔ and False with ✘
    marks = np.array(["✔" if value else "✘" for value in mask])[:, None]

    LOGGER.info(json.dumps(results, indent=4))
    LOGGER.info(np.hstack((np.round(sig.pi, 2), marks)))

    plot_adj(sig.L)

    y_pred = np.argmax(sig.pi, axis=1)
    y_pred_test = y_pred[graph.test_mask]
    if y_pred_test.shape[0] > 0:
        acc = calc_accuracy(y_pred_test, graph.y[graph.test_mask])
    else:
        acc = 0
    LOGGER.info(acc)


def plot_results(
    f_values,
    accuracies,
    Lambda_norms,
    atoms,
    epsilons,
    components,
    admm,
    save_path,
    f_opt=None,
):
    os.makedirs(save_path, exist_ok=True)
    plt.figure()
    plt.plot(f_values)
    plt.title("Loss")
    plt.ylim(min(f_values[10:]) - 1, max(f_values[10:]) + 1)
    plt.hlines(f_opt, 0, len(f_values), colors="r", linestyles="--")
    # plt.yscale("log")
    plt.savefig(f"{save_path}loss.png")
    plt.figure()
    L = [component[0] for component in components]
    R = [component[1] for component in components]
    pi = [component[2] for component in components]
    ADMM = [component[3] for component in components]
    plt.plot(L, label="L")
    plt.plot(R, label="R")
    plt.plot(pi, label="pi")
    plt.plot(ADMM, label="ADMM")
    plt.plot(f_values, label="f")
    plt.title("Loss")
    plt.yscale("log")
    plt.legend()
    # plt.ylim(min(f_values[100:]) - 1, max(f_values[100:]) + 1)
    plt.hlines(f_opt, 0, len(f_values), colors="r", linestyles="--")
    # plt.yscale("log")
    plt.savefig(f"{save_path}loss2.png")
    plt.figure()
    plt.plot(accuracies)
    plt.title("Accuracy")
    plt.savefig(f"{save_path}accuracies.png")
    plt.figure()
    plt.plot(Lambda_norms)
    plt.title("Lambda norms")
    plt.savefig(f"{save_path}Lambda_norms.png")

    plt.figure()
    for id, atom in enumerate(atoms):
        plt.plot(atom.e, "*", label=f"atom {id}")
    plt.legend()
    plt.title("Eigenvectors")
    plt.savefig(f"{save_path}eigenvectors.png")

    fig, ax = plt.subplots()
    max_e = np.max(epsilons)
    min_e = np.min(epsilons)
    ax.set_ylim(min_e, max_e)
    ax.set_title("Eigenvectors")

    def update(frame):
        ax.clear()
        for id, e in enumerate(epsilons[frame]):
            ax.plot(e, "*", label=f"atom {id}")
        ax.legend()
        ax.set_title(f"Eigenvectors_{frame}")
        ax.set_ylim(min_e, max_e)

    # Reduce the number of frames by skipping some
    step = max(1, len(epsilons) // 500)
    frames = range(0, len(epsilons), step)  # Adjust the step as needed

    ani = FuncAnimation(fig, update, frames=frames, repeat=False)

    # Lower the resolution
    dpi = 80  # Adjust the DPI as needed

    ani.save(f"{save_path}eigenvectors.gif", writer=PillowWriter(fps=25), dpi=dpi)

    if admm is not None:
        plt.figure()
        plt.plot(Lambda_norms)
        plt.title("Lambda norms")
        plt.savefig(f"{save_path}Lambda_norms.png")
        plot_adj(admm.Lambda, name="Lambda")


def save_model_params(atoms, coefs, graph, admm, save_path):
    os.makedirs(save_path, exist_ok=True)
    filename = f"{save_path}atoms.pkl"
    with open(filename, "wb") as fid:
        pkl.dump(atoms, fid)

    filename = f"{save_path}coefs.pkl"
    with open(filename, "wb") as fid:
        pkl.dump(coefs, fid)
    filename = f"{save_path}graph.pkl"
    with open(filename, "wb") as fid:
        pkl.dump(graph, fid)

    if admm is not None:
        filename = f"{save_path}Lambda.pkl"
        with open(filename, "wb") as fid:
            pkl.dump(admm.Lambda, fid)

        filename = f"{save_path}Lp.pkl"
        with open(filename, "wb") as fid:
            pkl.dump(admm.Lp, fid)


def project_atoms_sig(atoms, coefs, graph):
    sig = atoms.calc_sig(coefs)
    pi = project_to_simplex2(sig.pi, 1)
    L = project_to_bound(sig.L, 0, 1)
    return sig
