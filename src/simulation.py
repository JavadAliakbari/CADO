import os
import sys

pythonpath = os.getcwd()
if pythonpath not in sys.path:
    sys.path.append(pythonpath)

import numpy as np

from src.CANO import CANO
from src.Graph_Atom.ADMM import ADMM_cls
from src.load_atom import load_atom
from src.utils.SBM import SBM
from src.utils.algorithm_utils_orig import *
from src.utils.utils import *

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

    # Run spectral clustering
    n_clusters = c  # Number of clusters
    cluster_labels = run_spectral_clustering(graph, n_clusters)
    plt.figure()
    plt.plot(cluster_labels)
    plt.title("Cluster labels")
    plt.savefig(f"{save_path}/cluster_labels.png")
    # plt.show()
    a = 2

    if config.warm_start == "random":
        atoms, coefs = create_random_atoms(n, num_features, c, c)
    elif config.warm_start == "optimal":
        atoms, coefs = create_optimal_atoms(graph)
    elif config.warm_start == "cluster":
        atoms, coefs = create_atoms_from__cluster_labels(graph, cluster_labels)

    # path = "results/20250127_002519/"
    # atoms, coefs, graph, Lp, Lambda = load_atom(path)
    # admm = ADMM_cls(Lp, project_to_bound)
    # admm.Lambda = Lambda

    (
        atoms,
        coefs,
        graph,
        results,
        f_values,
        num_atoms_list,
        accuracies,
        f_opt,
        admm,
        Lambda_norms,
        epsilsons,
    ) = CANO(graph, atoms, coefs)
    # ) = CANO(graph)
    # ) = CANO(graph, atoms, coefs, admm)

    log_results(atoms, coefs, graph, results, save_path)
    plot_results(
        f_values,
        num_atoms_list,
        accuracies,
        Lambda_norms,
        atoms,
        epsilsons,
        admm,
        save_path,
        f_opt,
    )
    save_model_params(atoms, coefs, graph, admm, save_path)
