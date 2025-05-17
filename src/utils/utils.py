from datetime import datetime
import json
import os
import pickle as pkl
import random
import shutil

import cvxpy as cp
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.linalg import logm, expm
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from utils.config_parser import Config
from utils.logger import getLOGGER
from utils.visualize_R import *

seed = 10
random.seed(seed)
np.random.seed(seed)

load_dotenv()
config_path = os.environ.get("CONFIGPATH")
config = Config(config_path)
data = config.data
algorithm = config.algorithm
optimizer = config.optimizer

plot = os.environ.get("PLOT") == "True"

plt.rcParams["figure.figsize"] = [16, 9]
plt.rcParams["figure.dpi"] = 100  # 200 e.g. is really fine, but slower
plt.rcParams.update({"figure.max_open_warning": 0})
plt.rcParams["font.size"] = 15


now = datetime.now().strftime("%Y%m%d_%H%M%S")
result_path = os.environ.get("RESULTPATH")
save_path = f"{result_path}/{now}/"

# Define custom colors for 0 and 1 (e.g., 0 -> blue, 1 -> red)
cmap_colors = [
    (0, "white"),
    (1, "blue"),
]  # Replace 'blue' and 'red' with your desired colors

# Create a custom colormap using the colors
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", cmap_colors)


def save_config(output_path, config_name="config.yaml"):
    """Save the current config to the output directory"""

    # Create destination filename with timestamp
    dest_file = os.path.join(output_path, config_name)

    # Copy the file
    shutil.copy2(config_path, dest_file)


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


def calc_accuracy(y_pred, y_true):
    """Calculate accuracy."""
    return sum(y_pred == y_true) / len(y_true)


def clustring_accuracy(y_pred, y_true):
    # --- map cluster IDs to class IDs with Hungarian matching ---
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_assignment(cm.max() - cm)  # maximize matches
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    y_aligned = np.vectorize(mapping.get)(y_pred)

    acc = accuracy_score(y_true, y_aligned)
    return acc


def water_filling(a, b, d):
    # --- water‑filling on the capped simplex ----------------------------
    r = np.full(d, a)  # give each slot the minimum mass a
    t = 1.0 - d * a  # mass still to distribute

    if t < -1e-12:
        raise ValueError("a too large: no feasible R with trace 1")

    for i in range(d):  # fill cheapest directions first
        if t <= 0:
            break
        add = min(b - a, t)
        r[i] += add
        t -= add

    if t > 1e-12:
        raise ValueError("b too small: cannot reach trace 1")

    return r

    # --------------------------------------------------------------------


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


def create_Ri_symmetric_matrix_with_constrained_trace(
    dimension, rho_minus=None, rho_plus=None
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
    all_eigenvalues = np.random.rand(dimension)
    if rho_minus is not None:
        all_eigenvalues = project_capped_simplex(all_eigenvalues, rho_minus, rho_plus)
    else:
        all_eigenvalues = project_to_simplex(all_eigenvalues)

    # Create rho_minus random orthogonal matrix Q using QR decomposition
    Q, _ = np.linalg.qr(np.random.randn(dimension, dimension))

    # Construct the symmetric matrix using A = Q * Lambda * Q.T
    Lambda = np.diag(all_eigenvalues)  # Diagonal matrix of eigenvalues
    symmetric_matrix = Q @ Lambda @ Q.T

    return symmetric_matrix


def create_Ri_symmetric_matrix_with_constrained_eigenvalues2(
    dimension,
    rho_minus,
    rho_plus,
    num_signal_features,
    Q=None,
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

    if Q is None:
        Q, _ = np.linalg.qr(np.random.randn(dimension, dimension))

    # Create rho_minus random orthogonal matrix Q using QR decomposition

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


def project_capped_simplex(r, a, b, tol=1e-12, max_iter=1000):
    """
    Project a vector r onto the set
        { x : sum(x) = 1,  a ≤ x_i ≤ b }   (capped simplex)

    Parameters
    ----------
    r : (d,) array_like
        Input vector (any real values).
    a, b : floats
        Lower and upper bounds with 0 < a < b < 1.
        Feasibility requires  d*a ≤ 1 ≤ d*b .
    tol : float, optional
        Bisection tolerance on the dual variable.
    max_iter : int, optional
        Maximum number of bisection steps.

    Returns
    -------
    x : (d,) ndarray
        The projected vector.
    """
    r = np.asarray(r, dtype=float)
    d = r.size

    # quick feasibility check
    if d * a - 1 > tol or d * b - 1 < -tol:
        raise ValueError("No feasible point: make sure d*a ≤ 1 ≤ d*b.")

    # f(tau) = sum_i clip(r_i - tau, a, b) - 1     (monotone ↓ in tau)
    lo = r.min() - b  # f(lo) = d*b - 1  ≥ 0
    hi = r.max() - a  # f(hi) = d*a - 1  ≤ 0

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        s = np.clip(r - mid, a, b).sum()
        if s > 1:  # we need a larger tau
            lo = mid
        else:  # we need a smaller tau
            hi = mid
        if hi - lo < tol:
            break

    tau = hi  # root of f(tau) ≈ 0
    return np.clip(r - tau, a, b)


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


def project_to_unit_trace(matrix, rho_min=None, rho_max=None):
    """
    Project a matrix onto the positive semidefinite cone.

    Parameters:
    matrix (np.ndarray): The input matrix to be projected.

    Returns:
    np.ndarray: The projected positive semidefinite matrix.
    """
    # Perform eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    if rho_min is not None:
        clipped_eigenvalues = project_capped_simplex(eigenvalues, rho_min, rho_max)
    else:
        clipped_eigenvalues = project_to_simplex(eigenvalues)

    # Clip the eigenvalues to ensure they are non-negative

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


def matrix_sqrt(A):
    """
    Returns A^(1/2) and A^(-1/2) for a symmetric PSD matrix A.
    We compute the eigen-decomposition A = Q * diag(eigvals) * Q^T,
    then take the square root of the eigenvalues.
    """
    # Eigen-decomposition
    w, V = np.linalg.eigh(A)

    # Square roots of eigenvalues (assuming all w >= 0)
    w_sqrt = np.sqrt(w)
    w_invsqrt = 1.0 / (w_sqrt + 1e-15)  # add epsilon to avoid division by zero

    # Reconstruct A^(1/2) and A^(-1/2)
    A_sqrt = (V * w_sqrt) @ V.T
    A_invsqrt = (V * w_invsqrt) @ V.T

    return A_sqrt, A_invsqrt


def log_euclidean_interp(R1, R2, alpha):
    return expm((1 - alpha) * logm(R1) + alpha * logm(R2))


def harmonic_interp(R1, R2, alpha):
    Rinv = (1 - alpha) * np.linalg.inv(R1) + alpha * np.linalg.inv(R2)
    return np.linalg.inv(Rinv)


def geodesic_interpolation(R1, R2, alpha):
    """
    Computes R(alpha) = R1^(1/2) * (R1^(-1/2) * R2 * R1^(-1/2))^alpha * R1^(1/2).
    The result is a PSD matrix that 'geodesically' interpolates between R1 and R2.
    """
    # Compute R1^(1/2) and R1^(-1/2)
    R1_sqrt, R1_invsqrt = matrix_sqrt(R1)

    # Compute M = R1^(-1/2) * R2 * R1^(-1/2)
    M = R1_invsqrt @ R2 @ R1_invsqrt

    # Diagonalize M
    w, V = np.linalg.eigh(M)

    # Raise eigenvalues to the alpha power
    w_alpha = w**alpha

    # Reconstruct M^alpha
    M_alpha = (V * w_alpha) @ V.T

    # Finally, compute the geodesic interpolation
    R_alpha = R1_sqrt @ M_alpha @ R1_sqrt

    return R_alpha


def plot_adj(A, name="L"):
    os.makedirs(save_path, exist_ok=True)
    # Display the array as an image
    plt.figure()
    plt.imshow(A, cmap=custom_cmap)

    # Remove axis for a clean output
    plt.axis("off")

    # Save the image with a specific DPI (dots per inch) to control the output size
    plt.savefig(f"{save_path}{name}.png", dpi=300, bbox_inches="tight", pad_inches=0)


def log_results(atoms, coefs, graph, results, save_path, terminal=True):
    os.makedirs(save_path, exist_ok=True)
    LOGGER = getLOGGER(
        name=f"{now}",
        log_on_file=True,
        save_path=save_path,
        terminal=terminal,
    )
    results = {
        f"atom {id}": {
            "e": str(np.round(atom.e, 2)),
            "b": str(np.round(atom.b, 2)),
            "R": str(np.round(np.linalg.eigvalsh(atom.R), 2)),
            "λ": str(np.round(c, 2)),
        }
        for id, (c, atom) in enumerate(zip(coefs, atoms))
    }

    sig = atoms.calc_sig(coefs)
    y_pred = np.argmax(sig.pi, axis=1)
    mask = y_pred == graph.y
    # Replace True with ✔ and False with ✘
    marks = np.array(["✔" if value else "✘" for value in mask])[:, None]

    # Temporarily remove terminal (stream) handlers to log only to file
    h = LOGGER.handlers[0]
    LOGGER.removeHandler(h)
    LOGGER.info(json.dumps(results, indent=4))
    LOGGER.info(np.hstack((np.round(sig.pi, 2), marks)))
    # Reattach terminal handlers
    LOGGER.addHandler(h)

    return LOGGER


def plot_results(
    f_values,
    accuracies,
    atoms,
    coefs,
    graph,
    epsilons,
    components,
    save_path,
    num_sig_features,
    y_preds,
    Qs,
    *opt_params,
):
    os.makedirs(save_path, exist_ok=True)
    sig = atoms.calc_sig(coefs)
    plot_adj(sig.L)

    f_opt, L_opt, R_opt, pi_opt = opt_params

    plt.figure()
    plt.plot(f_values)
    plt.title("Loss")
    # if len(f_values) > 10:
    #     plt.ylim(min(f_values[10:]) - 1, max(f_values[10:]) + 1)
    plt.hlines(f_opt, 0, len(f_values), colors="r", linestyles="--")
    # plt.yscale("log")
    plt.savefig(f"{save_path}loss.png")
    plt.figure()
    L = [component[0] for component in components]
    R = [component[1] for component in components]
    pi = [component[2] for component in components]
    plt.plot(L, label="L", color="b")
    plt.plot(R, label="R", color="r")
    plt.plot(pi, label="pi", color="orange")
    plt.plot(f_values, label="f", color="black")
    plt.title("Loss")
    plt.legend()
    # plt.ylim(min(f_values[100:]) - 1, max(f_values[100:]) + 1)
    plt.hlines(L_opt, 0, len(f_values), colors="b", linestyles="--")
    plt.hlines(R_opt, 0, len(f_values), colors="r", linestyles="--")
    plt.hlines(pi_opt, 0, len(f_values), colors="orange", linestyles="--")
    plt.hlines(f_opt, 0, len(f_values), colors="black", linestyles="--")
    # plt.yscale("log")
    plt.savefig(f"{save_path}loss2.png")
    plt.figure()
    min_L = np.min(L)
    L = np.array(L) - min_L
    min_R = np.min(R)
    R = np.array(R) - min_R
    min_pi = np.min(pi)
    pi = np.array(pi) - min_pi
    min_f = np.min(f_values)
    f = np.array(f_values) - min_f
    plt.plot(L, label="L", color="b")
    plt.plot(R, label="R", color="r")
    plt.plot(pi, label="pi", color="orange")
    plt.plot(f, label="f", color="black")
    plt.title("Loss")
    plt.yscale("log")
    plt.legend()
    # plt.ylim(min(f_values[100:]) - 1, max(f_values[100:]) + 1)
    plt.hlines(f_opt, 0, len(f_values), colors="r", linestyles="--")
    # plt.yscale("log")
    plt.savefig(f"{save_path}loss3.png")

    plt.figure()
    plt.plot(accuracies)
    plt.title("Accuracy")
    plt.savefig(f"{save_path}accuracies.png")

    plt.figure()
    for id, atom in enumerate(atoms):
        plt.plot(atom.e, "*", label=f"atom {id}", color=colors[id])
    plt.legend()
    plt.title("Eigenvectors")
    plt.savefig(f"{save_path}eigenvectors.png")

    fig, ax = plt.subplots()
    num_frames = len(epsilons)
    max_e = np.max(epsilons[num_frames // 2 :])
    min_e = np.min(epsilons[num_frames // 2 :])
    ax.set_ylim(min_e, max_e)
    ax.set_title("Eigenvectors")

    def update(frame):
        ax.clear()
        for id, e in enumerate(epsilons[frame]):
            # atom_id = np.argmax(graph.b[id])
            ax.plot(e, "*", label=f"atom {id}", color=colors[id])
        ax.legend(loc="lower right")
        ax.set_title(f"Eigenvectors_{frame} acc: {accuracies[frame]:.2f}")
        ax.set_ylim(min_e, max_e)

    # Reduce the number of frames by skipping some
    step = max(1, len(epsilons) // 500)
    frames = range(0, len(epsilons), step)  # Adjust the step as needed

    ani = FuncAnimation(fig, update, frames=frames, repeat=False)

    # Lower the resolution
    dpi = 50  # Adjust the DPI as needed

    ani.save(f"{save_path}eigenvectors.gif", writer=PillowWriter(fps=25), dpi=dpi)

    visualize_high_dim_data(
        graph.X,
        y_preds,
        graph.y,
        Qs,
        num_sig_features=num_sig_features,
        save_path=save_path,
    )
    # visualize_high_dim_data2(
    #     graph.X,
    #     y_preds,
    #     graph.y,
    #     Qs,
    #     num_sig_features=num_sig_features,
    #     save_path=save_path,
    # )
    # visualize_high_dim_data2(graph.X, y_pred, graph.y, Q)


def save_model_params(atoms, coefs, graph, save_path):
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
