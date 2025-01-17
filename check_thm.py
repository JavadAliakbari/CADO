import json
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

from Atom.sig import Sig
from utils.SBM import SBM
from Atom.coefs import Coefs
from Atom.atoms import Atoms
from Graph_Atom.graph_func import GraphFunc

from Graph_Atom.graph_sig import EmptySig
from utils.algorithm_utils import *
from utils.utils import *
import pickle as pkl

# Set print options to show the entire array
np.set_printoptions(threshold=np.inf, suppress=False)
# np.set_printoptions(suppress=True)

def normalize_columns_l2(matrix):
    # Calculate the L2 norm of each column
    col_norms = np.linalg.norm(matrix, axis=0)
    # To avoid division by zero in case a column is all zeros
    col_norms[col_norms == 0] = 1
    # Divide each column by its norm
    return matrix / col_norms

def create_sig(E, R, b, λ):
    ## Initialization
    atoms = Atoms()
    coefs = Coefs()

    for i in range(c):
        atom0 = GraphAtom()
        S, U = np.linalg.eigh(R[i])
        mask = S >= config.sig_rho_max - 0.0000001
        S[mask] = config.rho_max - config.rho_min
        S[~mask] = 0

        RR = U @ np.diag(S) @ U.T
        atom0.set_params(E[:, i], RR, b[i])
        atoms.add_atom(atom0)
        coefs.add_coef(λ[i])
    opt_sig = atoms.calc_sig(coefs)
    return opt_sig



if __name__ == "__main__":
    #### CoGEnT algorithm
    samples_per_cluster = [100, 100, 100]  # Number of training samples in each cluster
    # samples_per_cluster = [3, 4, 5]  # Number of training samples in each cluster
    num_features = 4  # Dimension of the features at each node
    num_signal_features = 2
    n = sum(samples_per_cluster)
    c = len(samples_per_cluster)
    config.tau = n
    # plot_adj(graph.A, name="A")


    f_avg = []
    f_opts = []
    accuracies = []
    p_vals = np.arange(0,1, 0.03)
    for p in tqdm(p_vals):
        probs = np.full((c, c), p)
        np.fill_diagonal(probs, 1-p)
        # probs = np.eye(c)
        graph = SBM(
            samples_per_cluster,
            probs,
            num_features,
            num_signal_features,
            # train_ratio=1,
            train_ratio=1.,
        )
        E = graph.E
        # E[np.abs(E) < 0.00001] = 0.00001
        R = graph.Ri_matrices
        b = graph.b
        opt_sig = create_sig(E, R, b, graph.λ)
        f = GraphFunc(graph=graph)
        f_opt = f(opt_sig)
        f_opts.append(f_opt)

        f_vals = []
        acc = 0
        total = 20000
        for iter in range(total):
            noisy_E = 1*np.random.randn(*E.shape)
            noisy_E = normalize_columns_l2(E)
            noisy_coef = n * np.random.randn(c)
            noisy_coef = project_to_simplex(noisy_coef, n)

            # scale = 10
            # noisy_E = normalize_columns_l2(graph.E + scale*np.random.randn(*graph.E.shape))
            # noisy_coef = project_to_simplex(graph.λ + 20*np.random.rand(*graph.λ.shape), n)
            noisy_sig = create_sig(noisy_E, R, b, noisy_coef)
            noisy_sig.L = project_to_bound(noisy_sig.L)
            f_noisy = f(noisy_sig)
            f_vals.append(f_noisy)
            if f_noisy > f_opt:
                acc += 1

        f_avg.append(np.mean(f_vals))
        accuracies.append(acc / total)



    plt.plot(p_vals, f_avg)    
    plt.plot(p_vals, f_opts)    

    plt.figure()
    plt.plot(p_vals, accuracies)
    plt.show()








   