import numpy as np


from CANO import CANO
from utils.SBM import SBM
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

    # Run spectral clustering
    n_clusters = c  # Number of clusters
    cluster_labels = run_spectral_clustering(graph, n_clusters)
    plt.figure()
    plt.plot(cluster_labels)
    # plt.show()
    a = 2

    atoms, coefs, graph, results, f_values, num_atoms_list, accuracies, f_opt, admm = (
        CANO(graph)
    )

    log_results(atoms, coefs, graph, results, save_path)
    plot_results(f_values, num_atoms_list, accuracies, atoms, save_path, f_opt)
    save_model_params(atoms, coefs, graph, admm, save_path)
