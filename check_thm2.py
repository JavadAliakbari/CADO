from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

from utils.SBM import SBM
from Atom.coefs import Coefs
from Atom.atoms import Atoms
from Graph_Atom.graph_func import GraphFunc

from utils.algorithm_utils import *
from utils.utils import *

# Set print options to show the entire array
np.set_printoptions(threshold=np.inf, suppress=False)
# np.set_printoptions(suppress=True)


def normalize_columns_l2(matrix):
    # Calculate the L2 norm of each column
    col_norms = np.linalg.norm(matrix, axis=1)
    # To avoid division by zero in case a column is all zeros
    col_norms[col_norms == 0] = 1
    # Divide each column by its norm
    return matrix / col_norms[:, np.newaxis]


def random_one_hot_vector(size, num):
    """
    Choose a random one-hot vector of the given size.

    Args:
        size (int): The size of the one-hot vector.

    Returns:
        np.ndarray: A random one-hot vector.
    """
    one_hot_vectors = np.eye(size)
    random_index = np.random.choice(size, num, replace=True)
    return one_hot_vectors[random_index]


def create_sig(E, R, b, λ):
    ## Initialization
    atoms = Atoms()
    coefs = Coefs()

    for i in range(c):
        atom0 = GraphAtom()
        # S, U = np.linalg.eigh(R[i])
        # mask = S >= config.sig_rho_max - 0.0000001
        # S[mask] = config.rho_max - config.rho_min
        # S[~mask] = 0

        # RR = U @ np.diag(S) @ U.T
        atom0.set_params(E[:, i], R[i], b[i])
        atoms.add_atom(atom0)
        coefs.add_coef(λ[i])
    opt_sig = atoms.calc_sig(coefs)
    return opt_sig


def process_iteration(iter, noisy_E, noisy_coef, f, R, b):
    """
    Process a single iteration of the loop.

    Args:
        iter (int): The current iteration number.
        graph (SBM): The graph object.
        noisy_E (np.ndarray): The noisy E matrix.
        noisy_coef (np.ndarray): The noisy coefficients.
        f (GraphFunc): The graph function.
        R (np.ndarray): The R matrices.
        b (np.ndarray): The b vector.

    Returns:
        float: The f_noisy value.
    """
    noisy_sig = create_sig(noisy_E[iter], R, b, noisy_coef[iter])
    noisy_sig.L = project_to_bound(noisy_sig.L)
    f_noisy = f(noisy_sig)
    return f_noisy


def run_experiment(
    noisy_E,
    noisy_coef,
    noisy_R,
    noisy_b,
    probs,
    samples_per_cluster,
    num_signal_features,
    total,
):
    """
    Run the experiment with the given parameters.

    Args:
        total (int): The total number of iterations.
        samples_per_cluster (int): The number of samples per cluster.
        probs (np.ndarray): The probability matrix.
        num_features (int): The number of features.
        num_signal_features (int): The number of signal features.
        noisy_E (np.ndarray): The noisy E matrix.
        noisy_coef (np.ndarray): The noisy coefficients.

    Returns:
        list: The f_vals list.
        list: The f_opts list.
        float: The accuracy.
    """
    num_features = noisy_R.shape[1]

    f_vals = []
    f_opts = []
    acc = 0
    for iter in tqdm(range(total), leave=False):
        if iter % 100 == 0:
            graph = SBM(
                samples_per_cluster,
                probs,
                num_features,
                num_signal_features,
                train_ratio=0.5,
            )
            E = graph.E
            R = graph.Ri_matrices
            b = graph.b
            opt_sig = create_sig(E, R, b, graph.λ)
            f = GraphFunc(graph=graph)
            f_opt = f(opt_sig)
            f_opts.append(f_opt)

        f_noisy = process_iteration(iter, noisy_E, noisy_coef, f, noisy_R, noisy_b)
        f_vals.append(f_noisy)
        if f_noisy > f_opt:
            acc += 1.0

    return f_vals, f_opts, acc


def create_random_atoms(n, num_features, c, q):
    # probs = np.eye(c)
    # E[np.abs(E) < 0.00001] = 0.00001
    noisy_E = 1 * np.random.uniform(-1, 1, (total, n, c))
    noisy_E = normalize_columns_l2(noisy_E)
    noisy_coef = np.random.uniform(0, n, (total, c))
    noisy_coef = project_to_simplex(noisy_coef, n)
    noisy_R = np.zeros((c, num_features, num_features))
    for i in range(c):
        noisy_R[i] = create_Ri_symmetric_matrix_with_constrained_eigenvalues2(
            num_features, config.rho_min, config.rho_max, num_signal_features
        )
    noisy_b = random_one_hot_vector(c, n)

    return noisy_E, noisy_coef, noisy_R, noisy_b


def run_outer_loop(
    q_vals, total, samples_per_cluster, num_features, num_signal_features
):
    f_rnd_avg = []
    f_rnd_std = []
    f_opt_avg = []
    f_opt_std = []
    accuracies = []
    # q_vals = np.arange(0.4, 1, 0.02)
    bar = tqdm(total=len(q_vals))
    # Add noise to the signal
    # total = 100000

    noisy_E, noisy_coef, noisy_R, noisy_b = create_random_atoms(n, num_features, c, q)
    for q in q_vals:
        probs = np.full((c, c), q)
        np.fill_diagonal(probs, 0.2)
        f_vals, f_opts, acc = run_experiment(
            noisy_E,
            noisy_coef,
            noisy_R,
            noisy_b,
            probs,
            samples_per_cluster,
            num_signal_features,
            total,
        )
        f_rnd_avg.append(np.mean(f_vals))
        f_rnd_std.append(np.std(f_vals))
        f_opt_avg.append(np.mean(f_opts))
        f_opt_std.append(np.std(f_opts))
        accuracies.append(acc / total)
        bar.set_postfix_str(
            f"q={q:0.2f}, f_avg={f_rnd_avg[-1]:.2f}, f_opt={f_opt_avg[-1]:.2f}, acc={accuracies[-1]:.2f}"
        )
        bar.update(1)

    return (
        np.array(f_rnd_avg),
        np.array(f_rnd_std),
        np.array(f_opt_avg),
        np.array(f_opt_std),
        np.array(accuracies),
    )


# Example usage
# samples_per_cluster = [3, 4, 5]  # Number of training samples in each cluster
num_features = 4  # Dimension of the features at each node
samples_per_cluster = [50] * 3  # Number of training samples in each cluster
num_signal_features = 2
n = sum(samples_per_cluster)
c = len(samples_per_cluster)
config.tau = n
total = 10000
q_vals = np.arange(0.0, 1, 0.02)


# Initialize lists to store results for both u1 cases
f_rnd_avg_all = []
f_rnd_std_all = []
f_opt_avg_all = []
f_opt_std_all = []
accuracies_all = []

u1_vals = [0, 0.5]
for u1 in u1_vals:
    config.mu1 = u1
    f_rnd_avg, f_rnd_std, f_opt_avg, f_opt_std, accuracies = run_outer_loop(
        q_vals, total, samples_per_cluster, num_features, num_signal_features
    )
    f_rnd_avg_all.append(f_rnd_avg)
    f_rnd_std_all.append(f_rnd_std)
    f_opt_avg_all.append(f_opt_avg)
    f_opt_std_all.append(f_opt_std)
    accuracies_all.append(accuracies)

# Plot the objective function for both u1 cases
plt.figure()
for i, u1 in enumerate(u1_vals):
    plt.plot(q_vals, f_rnd_avg_all[i], "-*", label=f"$f(\Psi)$ for u1={u1}")
    plt.fill_between(
        q_vals,
        f_rnd_avg_all[i] - f_rnd_std_all[i],
        f_rnd_avg_all[i] + f_rnd_std_all[i],
        alpha=0.2,
        label=f"Std Dev for u1={u1}",
    )
    plt.plot(q_vals, f_opt_avg_all[i], "-*", label=f"$f(\Psi^*)$ for u1={u1}")
    plt.fill_between(
        q_vals,
        f_opt_avg_all[i] - f_opt_std_all[i],
        f_opt_avg_all[i] + f_opt_std_all[i],
        alpha=0.2,
    )
plt.xlabel("q")
plt.ylabel("$f(\Phi)$")
plt.title("Objective Function Comparison for u1=0 and u1=1")
plt.legend()
plt.savefig("Objective_Comparison.png")
# plt.show()

# Plot the accuracy for both u1 cases
plt.figure()
for i, u1 in enumerate([0, 1]):
    plt.plot(q_vals, accuracies_all[i], "-*", label=f"Accuracy for u1={u1}")
plt.plot(q_vals, np.ones_like(q_vals))
plt.xlabel("q")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison for u1=0 and u1=1")
plt.legend()
plt.savefig("Accuracy_Comparison.png")
plt.show()
