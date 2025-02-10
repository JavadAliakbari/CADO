import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
from tqdm import tqdm

from CANO import CANO
from Graph_Atom.ADMM import ADMM_cls
from load_atom import load_atom
from utils.algorithm_utils_orig import *
from utils.SBM import SBM
from utils.utils import *

# Set print options to show the entire array
np.set_printoptions(threshold=np.inf, suppress=False)


def soft_thresholdinhg(A, L, lr=0.01, mu=1):
    """
    Soft thresholding algorithm
    """
    if config.Abar:
        Lp = L + lr * A
    else:
        Lp = L + lr * (A - L)
    D, E = np.linalg.eigh(Lp)
    D = np.maximum(D - mu * lr, 0)
    L_new = E @ np.diag(D) @ E.T
    L_new = project_to_bound(L_new)

    return L_new


if __name__ == "__main__":
    samples_per_cluster = [100, 100, 100]  # Number of training samples in each cluster
    num_features = 6  # Dimension of the features at each node
    num_signal_features = 2
    n = sum(samples_per_cluster)
    c = len(samples_per_cluster)
    n_clusters = c

    probs = np.full((c, c), config.q)
    np.fill_diagonal(probs, config.p)
    # probs = np.eye(c)
    config.tau = n
    graph = SBM(
        samples_per_cluster,
        probs,
        num_features,
        num_signal_features,
        # train_ratio=1,
        train_ratio=config.train_ratio,
    )
    plot_adj(graph.A, name="A")

    atoms_opt, coefs_opt = create_optimal_atoms(graph)
    opt_sig = atoms_opt.calc_sig(coefs_opt)
    L_opt = opt_sig.L

    L = deepcopy(opt_sig.L)
    # L = np.zeros((n, n))
    diff_list = []  # List to store the difference between L and L_opt
    total_iter = 1000
    bar = tqdm(total=total_iter, leave=True)
    for i in range(1, total_iter):
        # print(f"Iteration {i}")
        L = soft_thresholdinhg(graph, L, lr=0.01, mu=10)
        diff = np.linalg.norm(L - L_opt)
        bar.set_postfix_str({"Diff": diff})
        bar.update(1)
        diff_list.append(diff)

    plot_adj(L, name="L")

    plt.figure()
    plt.plot(diff_list)
    plt.xlabel("Iteration")
    plt.ylabel("Difference")
    plt.title("Difference between L and L_opt")
    plt.savefig(f"{save_path}soft_thresholding_diff.png")
    # plt.show()
