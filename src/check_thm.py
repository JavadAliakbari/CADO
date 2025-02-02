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


if __name__ == "__main__":
    #### CoGEnT algorithm
    samples_per_cluster = [20, 20, 20]  # Number of training samples in each cluster
    # samples_per_cluster = [3, 4, 5]  # Number of training samples in each cluster
    num_features = 4  # Dimension of the features at each node
    num_signal_features = 2
    n = sum(samples_per_cluster)
    c = len(samples_per_cluster)
    config.tau = n
    # plot_adj(graph.A, name="A")

    f_rnd_avg = []
    f_rnd_std = []
    f_opt_avg = []
    f_opt_std = []
    accuracies = []
    q_vals = np.arange(0.4, 1, 0.02)
    bar = tqdm(total=len(q_vals))
    # Add noise to the signal
    total = 100000

    for q in q_vals:
        probs = np.full((c, c), q)
        np.fill_diagonal(probs, 0.2)
        # probs = np.eye(c)
        # E[np.abs(E) < 0.00001] = 0.00001
        noisy_E = 1 * np.random.uniform(-1, 1, (total, n, c))
        noisy_E = normalize_columns_l2(noisy_E)
        noisy_coef = np.random.uniform(0, n, (total, c))
        noisy_coef = project_to_simplex(noisy_coef, n)

        f_vals = []
        f_opts = []
        acc = 0
        for iter in range(total):
            if iter % 50 == 0:
                graph = SBM(
                    samples_per_cluster,
                    probs,
                    num_features,
                    num_signal_features,
                    # train_ratio=1,
                    train_ratio=0.5,
                )
                E = graph.E
                R = graph.Ri_matrices
                b = graph.b
                opt_sig = create_sig(E, R, b, graph.λ)
                f = GraphFunc(graph=graph)
                f_opt = f(opt_sig)
                f_opts.append(f_opt)

            # scale = 10
            # noisy_E = normalize_columns_l2(graph.E + scale*np.random.randn(*graph.E.shape))
            # noisy_coef = project_to_simplex(graph.λ + 20*np.random.rand(*graph.λ.shape), n)
            noisy_sig = create_sig(noisy_E[iter], R, b, noisy_coef[iter])
            noisy_sig.L = project_to_bound(noisy_sig.L)
            f_noisy = f(noisy_sig)
            f_vals.append(f_noisy)
            if f_noisy > f_opt:
                acc += 1.0

        f_rnd_avg.append(np.mean(f_vals))
        f_rnd_std.append(np.std(f_vals))
        f_opt_avg.append(np.mean(f_opts))
        f_opt_std.append(np.std(f_opts))
        accuracies.append(acc / total)
        bar.set_postfix_str(
            f"q={q:0.2f}, f_avg={f_rnd_avg[-1]:.2f}, f_opt={f_opt_avg[-1]:.2f}, acc={accuracies[-1]:.2f}"
        )
        bar.update(1)

    plt.plot(q_vals, f_rnd_avg, "-*", label="$f(\Psi)$")
    # Fill the area between f_avg - f_std and f_avg + f_std
    f_rnd_avg = np.array(f_rnd_avg)
    f_rnd_std = np.array(f_rnd_std)
    plt.fill_between(
        q_vals,
        f_rnd_avg - f_rnd_std,
        f_rnd_avg + f_rnd_std,
        color="b",
        alpha=0.2,
        # label="Standard Deviation",
    )
    f_opt_avg = np.array(f_opt_avg)
    f_opt_std = np.array(f_opt_std)
    plt.plot(q_vals, f_opt_avg, "-*", label="$f(\Psi^*)$")
    plt.fill_between(
        q_vals,
        f_opt_avg - f_opt_std,
        f_opt_avg + f_opt_std,
        color="r",
        alpha=0.2,
        # label="Standard Deviation",
    )
    plt.xlabel("q")
    plt.ylabel("$f(\Phi)$")
    plt.savefig("Objective.png")
    plt.legend()

    plt.figure()
    plt.plot(q_vals, accuracies, "-*")
    plt.plot(q_vals, np.ones(f_opt_avg.shape))
    plt.xlabel("q")
    plt.ylabel("Accuracy")
    plt.savefig("Accuracy.png")
    plt.show()