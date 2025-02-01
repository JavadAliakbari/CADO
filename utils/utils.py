from datetime import datetime
import json
import os
import pickle as pkl
import random

import cvxpy as cp
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from utils.config_parser import Config
from utils.logger import getLOGGER

seed = 42
random.seed(seed)
np.random.seed(seed)

###############################################################
###############################################################
###############################################################

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


###############################################################
###############################################################
###############################################################


def create_random_normalized_vector(n):
    vector = np.random.randn(n)
    normalized_vector = vector / np.linalg.norm(vector)
    return normalized_vector


def create_list_of_vectors_to_separate_space_of_signals(
    num_clusters, dimension, rho_min, rho_plus, signal_subspace_dim
):
    vectors = []
    for i in range(num_clusters):
        vector = np.ones(dimension) * rho_min
        vector[i * signal_subspace_dim : (i + 1) * signal_subspace_dim] = rho_plus
        vectors.append(vector)
    return vectors


def get_random_coef_between_1_and_(n):
    return np.random.uniform(1, n)


def matrix_dot_product(A, B):
    return np.sum(A * B)


def invert_matrix(matrix):
    # try:
    return np.linalg.inv(matrix)
    # except np.linalg.LinAlgError:
    # return np.eye(matrix.shape[0])
    # return np.linalg.inv(matrix + 1e-6 * np.eye(matrix.shape[0]))


def compute_inner_products(tensor):
    n, d, _ = tensor.shape
    result = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            result[i, j] = np.inner(tensor[i].flatten(), tensor[j].flatten())
    return result


def calc_accuracy(pred_y, y):
    """Calculate accuracy."""
    return sum(pred_y == y) / len(y)


###############################################################
###############################################################
###############################################################


def initialize_atoms(
    graph,
    init="rand",
    init_atoms_e=None,
    init_atoms_R=None,
    init_atoms_b=None,
    init_coefs=None,
):

    atoms_e_opt = []
    atoms_R_opt = []
    atoms_b_opt = []
    coeffss_opt = []

    for i in range(config.num_clusters):
        atoms_e_opt.append(graph.E[:, i])
        atoms_R_opt.append(graph.Ri_matrices[i])
        atoms_b_opt.append(graph.b[i])
        coeffss_opt.append(graph.λ[i])

    if init == "rand":
        atoms_e = []
        atoms_R = []
        atoms_b = []
        coeffss = []
        for i in range(config.num_clusters):
            atoms_e.append(create_random_normalized_vector(config.num_nodes))
            atoms_R.append(
                create_Ri_symmetric_matrix_with_constrained_eigenvalues(
                    config.dimension, config.rho_min, config.rho_max, noise_level=0.0
                )
            )
            atoms_b.append(
                create_multiple_noisy_probability_vectors(
                    1, config.num_clusters, noise_level=1
                )[0]
            )
            coeffss.append(get_random_coef_between_1_and_(config.num_nodes))

    elif init == "opt":
        atoms_e = atoms_e_opt
        atoms_R = atoms_R_opt
        atoms_b = atoms_b_opt
        coeffss = coeffss_opt

    elif init == "zeros":
        atoms_e = []
        atoms_R = []
        atoms_b = []
        coeffss = []
        for i in range(config.num_clusters):
            atoms_e.append(np.zeros(config.num_nodes))
            atoms_R.append(np.zeros((config.dimension, config.dimension)))
            atoms_b.append(np.zeros(config.num_clusters))
            coeffss.append(0)

    elif init == "load":
        atoms_e = init_atoms_e
        atoms_R = init_atoms_R
        atoms_b = init_atoms_b
        coeffss = init_coefs

    sig_L, _, _ = calculate_sig_from_atoms(atoms_e, atoms_R, atoms_b, coeffss)

    return (
        atoms_e,
        atoms_R,
        atoms_b,
        coeffss,
        atoms_e_opt,
        atoms_R_opt,
        atoms_b_opt,
        coeffss_opt,
        sig_L,
    )


###############################################################
###############################################################
###############################################################


def calculate_sig_from_atoms(atoms_e, atoms_R, atoms_b, coefs):
    sig_L = np.zeros((config.num_nodes, config.num_nodes))
    sig_Rv_s = np.zeros((config.num_nodes, config.dimension, config.dimension))
    sig_Piv_s = np.zeros((config.num_nodes, config.num_clusters))

    for e, R, b, coef in zip(atoms_e, atoms_R, atoms_b, coefs):
        sig_L += coef * np.outer(e, e)
        sig_Rv_s += coef * np.einsum("i,jk->ijk", (e**2), R)
        sig_Piv_s += coef * np.einsum("i,j->ij", (e**2), b)

    return sig_L, sig_Rv_s, sig_Piv_s


def compute_term_0(graph, sig_L):
    if config.Abar:
        term_0 = -matrix_dot_product(graph.Abar, sig_L) / config.num_nodes
    else:
        r = graph.A - sig_L
        term_0 = 0.5 * matrix_dot_product(r, r) / config.num_nodes
    return term_0


def compute_term_1(graph, sig_Rv_s):
    if config.mu1 != 0:
        sig_Rv_inv = np.array([invert_matrix(matrix) for matrix in sig_Rv_s])
        term_1 = np.einsum("ijk,ijk->", graph.cov_ADOP, sig_Rv_inv) / config.dimension
        term_11 = np.trace(np.sum(sig_Rv_s, axis=0)) / config.dimension
    else:
        term_1 = 0
        term_11 = 0
    return term_1 + term_11


def compute_term_2(graph, sig_Piv_s):
    if config.mu2 != 0:
        train_labels = graph.y[graph.train_mask]
        train_nodes = np.arange(graph.num_nodes)[graph.train_mask]
        term_2 = np.sum(sig_Piv_s[train_nodes, train_labels]) / config.num_clusters
    else:
        term_2 = 0
    return term_2


def compute_term_3(coefs):
    return np.sum(coefs) / config.num_clusters


def compute_objective_value(
    graph, atoms_e, atoms_R, atoms_b, coefs, defined_sig_L=None
):
    sig_L, sig_Rv_s, sig_Piv_s = calculate_sig_from_atoms(
        atoms_e, atoms_R, atoms_b, coefs
    )

    if defined_sig_L is not None:
        sig_L = defined_sig_L

    term_0 = compute_term_0(graph, sig_L)
    term_1 = compute_term_1(graph, sig_Rv_s)
    term_2 = compute_term_2(graph, sig_Piv_s)
    term_3 = compute_term_3(coefs)

    objective = term_0 + config.mu1 * term_1 - config.mu2 * term_2 + config.mu0 * term_3

    return objective


###############################################################
###############################################################
###############################################################


def calc_grad_ei_ADOP(graph, atom_e, atom_R, atom_b, coef, Rv_s_inv):
    # scale_factor = (config.rho_max / config.rho_min) ** 2
    # tmp1 = 2 * coef * grad_sig_L @ atom_e
    tmp2 = np.zeros(config.num_nodes)
    tmp3 = np.zeros(config.num_nodes)
    tmp4 = np.zeros(config.num_nodes)

    # temp = (graph.X) @ invert_matrix(atom_R) @ np.transpose(graph.X)

    if config.mu1 != 0:
        grad_Rv = np.einsum("nij,njk, nkl->nil", Rv_s_inv, graph.cov_ADOP, Rv_s_inv)
        grad_Ri = np.einsum("n,ij->nij", 2 * coef * atom_e, atom_R)
        tmp2 = np.einsum("nij,nij->n", grad_Ri, grad_Rv)
        atom_trace_R = np.trace(atom_R)
        tmp3 = 2 * coef * atom_e * atom_trace_R
    tmp4 = 2 * coef * atom_e[graph.train_mask] * atom_b[graph.y[graph.train_mask]]

    grad_ei = (
        -config.mu1 * tmp2 / config.dimension
        + config.mu1 * tmp3 / config.dimension
        - config.mu2 * tmp4 / config.num_clusters
    )
    return grad_ei


def calc_grad_Ri_ADOP(graph, atom_e, atom_R, atom_b, coef, Rv_s_inv):
    # atom_e[np.where(atom_e == 0)] = 1e-4
    # tmp_1 = np.einsum("ijk,i->jk", graph.cov_ADOP, atom_e ** (-2))
    # R_inv = invert_matrix(atom_R)

    grad_Rv = np.einsum("nij,njk, nkl->nil", Rv_s_inv, graph.cov_ADOP, Rv_s_inv)
    tmp_2 = np.einsum("n,nil->il", (coef * atom_e**2), grad_Rv)
    # tmp_2 = np.einsum("jk,km,mn->jn", R_inv, tmp_1, R_inv)
    tmp_3 = coef * np.identity(config.dimension) * np.sum(atom_e**2)

    grad_Ri = tmp_3 - tmp_2
    return config.mu1 * grad_Ri / config.dimension


def calc_grad_bi_ADOP(graph, atom_e, atom_R, atom_b, coef):
    tmp = np.zeros((config.num_nodes, config.num_clusters), dtype=float)
    train_labels = graph.y[graph.train_mask]
    train_nodes = np.arange(config.num_nodes)[graph.train_mask]
    tmp[train_nodes, train_labels] = 1.0

    grad_bi = np.einsum("i, ij-> j", atom_e**2, tmp)
    return -config.mu2 * coef * grad_bi / config.num_clusters


def claulate_grad_coefs_ADOP(graph, atoms_e, atoms_R, atoms_b, coefs, Rv_s_inv):
    grad_coefs = []
    for atom_e, atom_R, atom_b, coef in zip(atoms_e, atoms_R, atoms_b, coefs):
        if config.mu1 != 0:
            grad_Rv = np.einsum("nij,njk, nkl->nil", Rv_s_inv, graph.cov_ADOP, Rv_s_inv)
            grad_c = np.einsum("n,il->nil", atom_e**2, atom_R)
            tmp1 = np.einsum("nij,nij->", grad_c, grad_Rv)
            tmp2 = np.sum(atom_e**2) * np.trace(atom_R)
            grad1 = config.mu1 * (-tmp1 + tmp2)
        else:
            grad1 = 0

        # grad2 = config.mu2 * np.sum(2 * coef * atom_e * atom_b[graph.y])
        tmp = np.zeros((config.num_nodes, config.num_clusters), dtype=float)
        train_labels = graph.y[graph.train_mask]
        train_nodes = np.arange(config.num_nodes)[graph.train_mask]
        tmp[train_nodes, train_labels] = atom_b[graph.y[graph.train_mask]]

        tmp3 = np.einsum("i, ij-> j", atom_e**2, tmp)
        grad2 = -config.mu2 * np.sum(tmp3)

        grad3 = config.mu0

        grad = (
            grad1 / config.dimension
            + grad2 / config.num_clusters
            + grad3 / config.num_clusters
        )
        grad_coefs.append(grad)
    return np.array(grad_coefs)


def calc_grad_atom_ADOP(graph, atom_e, atom_R, atom_b, coef, Rv_s_inv):
    grad_ei = calc_grad_ei_ADOP(graph, atom_e, atom_R, atom_b, coef, Rv_s_inv)
    if config.mu1 != 0:
        grad_Ri = calc_grad_Ri_ADOP(graph, atom_e, atom_R, atom_b, coef, Rv_s_inv)
    else:
        grad_Ri = 0
    grad_bi = calc_grad_bi_ADOP(graph, atom_e, atom_R, atom_b, coef)

    return grad_ei, grad_Ri, grad_bi


def project_to_bound_ADOP(atom_e, atom_R, atom_b):
    proj_atom_e = project_to_unit_vector(atom_e)
    proj_atom_R = project_to_psd_cone_with_constrained_eigenvals(
        atom_R, config.rho_min, config.rho_max
    )
    proj_atom_b = project_to_simplex(atom_b, 1)
    return proj_atom_e, proj_atom_R, proj_atom_b


def project_to_unit_vector(vector):
    return vector / np.linalg.norm(vector)


def project_to_psd_cone_with_constrained_eigenvals(matrix, rho_minus, rho_plus):
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    clipped_eigenvalues = np.clip(eigenvalues, rho_minus, rho_plus)
    projected_matrix = eigenvectors @ np.diag(clipped_eigenvalues) @ eigenvectors.T
    return projected_matrix


def project_to_simplex(y, a):
    if y.ndim == 1:
        return _project_vector_to_simplex(y, a)
    elif y.ndim == 2:
        return np.apply_along_axis(_project_vector_to_simplex, 1, y, a)
    else:
        raise ValueError("Input array must be 1D or 2D")


def _project_vector_to_simplex(y, a):
    u = np.sort(y)[::-1]
    cssv = np.cumsum(u) - a
    rho = np.nonzero(u * np.arange(1, len(y) + 1) > cssv)[0][-1]
    theta = cssv[rho] / (rho + 1)
    return np.maximum(y - theta, 0)


def update_atom_ADOP(grad_ei, grad_Ri, grad_bi, atom_e, atom_R, atom_b, lr):
    new_atom_e = atom_e - lr * grad_ei
    new_atom_R = atom_R - lr * grad_Ri
    new_atom_b = atom_b - lr * grad_bi
    proj_atom_e, proj_atom_R, proj_atom_b = project_to_bound_ADOP(
        new_atom_e, new_atom_R, new_atom_b
    )
    return proj_atom_e, proj_atom_R, proj_atom_b


def update_atoms_and_L_ADOP(graph, atoms_e, atoms_R, atoms_b, coefs, sig_L):
    if config.mu1 != 0:
        _, sig_Rv_s_tmp, _ = calculate_sig_from_atoms(atoms_e, atoms_R, atoms_b, coefs)
        Rv_s_inv = np.array([invert_matrix(matrix) for matrix in sig_Rv_s_tmp])
    else:
        Rv_s_inv = None

    grad_coefs = claulate_grad_coefs_ADOP(
        graph, atoms_e, atoms_R, atoms_b, coefs, Rv_s_inv
    )
    grads = []
    for idx, (atom_e, atom_R, atom_b, coef) in enumerate(
        zip(atoms_e, atoms_R, atoms_b, coefs)
    ):
        out = calc_grad_atom_ADOP(graph, atom_e, atom_R, atom_b, coef, Rv_s_inv)
        grads.append(out)

    for idx, (grad_ei, grad_Ri, grad_bi) in enumerate(grads):
        coefs[idx] = update_coef_ADOP(coefs[idx], grad_coefs[idx], config.lr)
        atoms_e[idx], atoms_R[idx], atoms_b[idx] = update_atom_ADOP(
            grad_ei, grad_Ri, grad_bi, atom_e, atom_R, atom_b, config.lr
        )

    sig_L -= config.lr * calculate_grad_sig_L(graph, sig_L)

    return atoms_e, atoms_R, atoms_b, coefs, sig_L


###############################################################
###############################################################
###############################################################


def update_coef_ADOP(coef, grad_coef, lr):
    new_coef = coef - lr * grad_coef
    return new_coef


def calculate_grad_sig_L(graph, sig_L):
    if config.Abar:
        grad = -graph.Abar
    else:
        grad = graph.A - sig_L
    return grad


###############################################################
###############################################################
###############################################################


def project_to_Atomic_Decomposition(atoms_e, sig_L, coefs):
    A = np.array(
        [np.outer(atom_e, atom_e) for atom_e in atoms_e]
    )  # shape = (num_atoms, n, n)
    B = compute_inner_products(A)  # shape = (num_atoms, num_atoms)
    y_tmp = np.einsum("kij,k->ij", A, coefs) - sig_L  # shape = (n, n)
    y_bar = np.array(
        [np.einsum("ij,ij->", A[i], y_tmp) for i in range(config.num_clusters)]
    )  # shape = (num_atoms)
    tmp = invert_matrix(B + np.eye(config.num_clusters))
    alpha = np.matmul(tmp, y_bar)  # shape = (num_atoms)
    lagrange_z = y_tmp - np.einsum("kij,k->ij", A, alpha)

    sig_L += lagrange_z
    coefs -= np.einsum("ijk,jk->i", A, lagrange_z)

    return sig_L, coefs


def project_to_positive_coefs(coefs):
    return np.maximum(coefs, 0)


def project_to_bound(a, a_min=0, a_max=1):
    return np.clip(a, a_min, a_max)


def project_sig_L_and_coefs_ADOP(atoms_e, sig_L, coefs):
    sig_L, coefs = project_to_Atomic_Decomposition(atoms_e, sig_L, coefs)
    coefs = project_to_positive_coefs(coefs)
    sig_L = project_to_bound(sig_L)
    return sig_L, coefs


###############################################################
###############################################################
###############################################################


def calc_accuracy_ADOP(graph, atoms_e, atoms_R, atoms_b, coefs):
    sig_L, sig_Rv_s, sig_piv_s = calculate_sig_from_atoms(
        atoms_e, atoms_R, atoms_b, coefs
    )
    y_pred = np.argmax(sig_piv_s, axis=1)
    y_pred_test = y_pred[graph.test_mask]

    if y_pred_test.shape[0] > 0:
        acc = calc_accuracy(y_pred_test, graph.y[graph.test_mask])
    else:
        acc = 1

    return acc


###############################################################
###############################################################
###############################################################


def create_Ri_symmetric_matrix_with_constrained_eigenvalues(
    dimension, rho_minus, rho_plus, noise_level=0.0
):
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


def create_Ri_symmetric_matrix_with_constrained_eigenvalues2_ADOP(
    dimension,
    rho_minus,
    rho_plus,
    num_signal_features,
    all_eigenvalues,
    dimension_permutation,
):

    Q = np.zeros((dimension, dimension), dtype=float)
    Q[np.arange(dimension), dimension_permutation] = 1

    Lambda = np.diag(all_eigenvalues)
    symmetric_matrix = Q @ Lambda @ Q.T

    return symmetric_matrix


def create_multiple_noisy_probability_vectors(
    count, dimension, noise_level=0.0, manual_probabilities=None
):
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


###############################################################
###############################################################
###############################################################


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

    # plt.show(block=False)
    # plt.close()


###############################################################
###############################################################
###############################################################


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
    f_values, accuracies, Lambda_norms, atoms, Lambda, save_path, f_opt=None
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
    plt.plot(accuracies)
    plt.title("Accuracy")
    plt.savefig(f"{save_path}accuracies.png")
    plt.figure()
    plt.plot(Lambda_norms)
    plt.title("Lambda norms")
    plt.savefig(f"{save_path}Lambda_norms.png")

    plt.figure()
    for id, atom in enumerate(atoms):
        plt.plot(atom.e, "*-", label=f"atom {id}")
    plt.legend()
    plt.title("Eigenvectors")
    plt.savefig(f"{save_path}eigenvectors.png")

    plot_adj(Lambda, name="Lambda")


def plot_results2(f_values, accuracies, atoms, save_path, f_opt=None):
    os.makedirs(save_path, exist_ok=True)
    plt.figure()
    plt.plot(f_values)
    plt.title("Loss")
    plt.ylim(min(f_values[10:]) - 1, max(f_values[10:]) + 1)
    plt.hlines(f_opt, 0, len(f_values), colors="r", linestyles="--")
    # plt.yscale("log")
    plt.savefig(f"{save_path}loss.png")
    plt.figure()
    plt.plot(accuracies)
    plt.title("Accuracy")
    plt.savefig(f"{save_path}accuracies.png")

    plt.figure()
    for id, atom in enumerate(atoms):
        plt.plot(atom.e, "*-", label=f"atom {id}")
    plt.legend()
    plt.title("Eigenvectors")
    plt.savefig(f"{save_path}eigenvectors.png")


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

    if config.ADMM:
        filename = f"{save_path}Lambda.pkl"
        with open(filename, "wb") as fid:
            pkl.dump(admm.Lambda, fid)

        filename = f"{save_path}Lp.pkl"
        with open(filename, "wb") as fid:
            pkl.dump(admm.Lp, fid)


def save_model_params2(atoms, coefs, graph, save_path):
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
