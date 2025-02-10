import numpy as np
from scipy.linalg import qr

from utils.utils import plot_adj

import numpy as np
from tqdm import tqdm

from utils.satisfy_bound2 import satisfy_bound
from utils.SBM import SBM
from Graph_Atom.graph_func import GraphFunc
from Graph_Atom.ADMM import ADMM_cls
from utils.algorithm_utils_orig import *
from utils.utils import *

# Set print options to show the entire array
np.set_printoptions(threshold=np.inf, suppress=False)

if __name__ == "__main__":
    #### CoGEnT algorithm
    samples_per_cluster = [100, 100, 100]  # Number of training samples in each cluster
    num_features = 6  # Dimension of the features at each node
    num_signal_features = 2
    n = sum(samples_per_cluster)
    c = len(samples_per_cluster)
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

    rho = 0.1
    step = 0.1
    reg = 0.05 * n

    L = graph.Abar
    Z = np.zeros_like(L, dtype=float)
    # L = np.zeros_like(L, dtype=float)

    E = np.zeros((n, c), dtype=float)
    lam = np.zeros((c))

    e_temp = np.zeros((n))

    bar = tqdm(total=config.iters)
    for count in range(config.iters):
        Estim = E @ np.diag(lam) @ E.T
        Error = L - Estim

        Z += rho * Error

        val1 = np.linalg.norm(Error)
        val2 = reg * np.sum(lam) - np.sum(graph.Abar * L)

        # print(val1, val2)

        # L+=step*(Graph-Z)-Error
        L = (rho * Estim - Z + graph.Abar) / rho
        L[L > 1] = 1
        L[L < 0] = 0

        M = Error + Z / rho
        for count2 in range(c):
            e_temp = E[:, count2]
            M += lam[count2] * np.outer(e_temp, e_temp)
            lam_comp, e_comp = np.linalg.eigh(M)
            lam_temp = lam_comp[-1] - reg / rho
            if lam_temp < 0:
                lam_temp = 0
            e_temp = e_comp[:, -1]
            lam[count2] = lam_temp
            E[:, count2] = e_temp
            M -= lam_temp * np.outer(e_temp, e_temp)

        bar.set_postfix({"val1": val1, "val2": val2})
        bar.update(1)

    plot_adj(L, name="L")
