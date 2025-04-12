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
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from utils.config_parser import Config
from utils.logger import getLOGGER

seed = 4
random.seed(seed)
np.random.seed(seed)

load_dotenv()
config_path = os.environ.get("CONFIGPATH")
config = Config(config_path)

plot = os.environ.get("PLOT") == "True"

plt.rcParams["figure.figsize"] = [16, 9]
plt.rcParams["figure.dpi"] = 250  # 200 e.g. is really fine, but slower
plt.rcParams.update({"figure.max_open_warning": 0})
plt.rcParams["font.size"] = 15


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


def visualize_high_dim_data(
    X,
    predicted_clusters,
    true_clusters,
    subspaces,
    centers=None,
    filename="3d_plot.html",
):
    if centers is None:
        centers = [np.zeros(X.shape[1]) for _ in range(len(subspaces))]

    # Project data into 3 dimensions using PCA.
    pca = PCA(n_components=3)
    X_3d = pca.fit_transform(X)

    # Project subspace lines
    k = len(centers)
    t_vals = np.linspace(-3, 3, 1000)
    subspace_lines_3d = []
    for j in range(k):
        center = centers[j]
        basis = subspaces[j][:, 0]  # assuming subspace dim = 1
        line_points = np.array([center + t * basis for t in t_vals])
        line_points_3d = pca.transform(line_points)
        subspace_lines_3d.append(line_points_3d)

    # Calculate opacity based on prediction correctness
    correct_mask = predicted_clusters == true_clusters
    opacities = np.where(
        correct_mask, 0.9, 0.3
    )  # Higher opacity for correct, lower for incorrect

    # Colors
    colors = ["red", "green", "blue", "cyan", "magenta", "yellow", "black"]

    # Start Plotly figure
    fig = go.Figure()

    # Add cluster scatter plots
    for j in range(k):
        idx_correct = (predicted_clusters == j) & (predicted_clusters == true_clusters)
        idx_incorrect = (predicted_clusters == j) & (
            predicted_clusters != true_clusters
        )
        fig.add_trace(
            go.Scatter3d(
                x=X_3d[idx_correct, 0],
                y=X_3d[idx_correct, 1],
                z=X_3d[idx_correct, 2],
                mode="markers",
                marker=dict(
                    size=3,
                    color=colors[j % len(colors)],
                    opacity=0.3,  # Set opacity based on correctness
                ),
                name=f"Cluster {j}",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=X_3d[idx_incorrect, 0],
                y=X_3d[idx_incorrect, 1],
                z=X_3d[idx_incorrect, 2],
                mode="markers",
                marker=dict(
                    size=3,
                    color=colors[j % len(colors)],
                    opacity=0.9,  # Set opacity based on correctness
                ),
                name=f"Cluster {j}",
            )
        )

    # Add subspace lines
    for j in range(k):
        line_points = subspace_lines_3d[j]
        fig.add_trace(
            go.Scatter3d(
                x=line_points[:, 0],
                y=line_points[:, 1],
                z=line_points[:, 2],
                mode="lines",
                line=dict(color=colors[j % len(colors)], width=4),
                name=f"Subspace {j}",
            )
        )

    fig.update_layout(
        title="Clusters and their projected subspaces (via PCA)",
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
    )

    # Save as interactive HTML
    fig.write_html(f"{save_path}{filename}")
    # print(f"Plot saved to {filename}")


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
    stream_handlers = [
        h for h in LOGGER.handlers if getattr(h, "stream", None) is not None
    ]
    for h in stream_handlers:
        LOGGER.removeHandler(h)
    LOGGER.info(json.dumps(results, indent=4))
    LOGGER.info(np.hstack((np.round(sig.pi, 2), marks)))
    # Reattach terminal handlers
    for h in stream_handlers:
        LOGGER.addHandler(h)

    y_pred_test = y_pred[graph.test_mask]
    if y_pred_test.shape[0] > 0:
        acc = calc_accuracy(y_pred_test, graph.y[graph.test_mask])
    else:
        acc = 0
    LOGGER.info(acc)
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
    plt.plot(L, label="L")
    plt.plot(R, label="R")
    plt.plot(pi, label="pi")
    plt.plot(f, label="f")
    plt.title("Loss")
    plt.yscale("log")
    plt.legend()
    # plt.ylim(min(f_values[100:]) - 1, max(f_values[100:]) + 1)
    plt.hlines(f_opt, 0, len(f_values), colors="r", linestyles="--")
    # plt.yscale("log")
    plt.savefig(f"{save_path}loss3.png")

    y_pred = np.argmax(sig.pi, axis=1)
    Q = np.array([np.linalg.eigh(atom.R)[1] for atom in atoms])
    visualize_high_dim_data(graph.X, y_pred, graph.y, Q)

    plt.figure()
    plt.plot(accuracies)
    plt.title("Accuracy")
    plt.savefig(f"{save_path}accuracies.png")

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
            # atom_id = np.argmax(graph.b[id])
            ax.plot(e, "*", label=f"atom {id}")
        ax.legend(loc="lower right")
        ax.set_title(f"Eigenvectors_{frame} acc: {accuracies[frame]:.2f}")
        ax.set_ylim(min_e, max_e)

    # Reduce the number of frames by skipping some
    step = max(1, len(epsilons) // 500)
    frames = range(0, len(epsilons), step)  # Adjust the step as needed

    ani = FuncAnimation(fig, update, frames=frames, repeat=False)

    # Lower the resolution
    dpi = 80  # Adjust the DPI as needed

    ani.save(f"{save_path}eigenvectors.gif", writer=PillowWriter(fps=25), dpi=dpi)


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


def project_atoms_sig(atoms, coefs, graph):
    sig = atoms.calc_sig(coefs)
    pi = project_to_simplex2(sig.pi, 1)
    L = project_to_bound(sig.L, 0, 1)
    return sig
