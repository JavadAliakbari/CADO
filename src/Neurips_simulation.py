import numpy as np
from sklearn.metrics import adjusted_rand_score

from idea5 import CANO
from load_atom import load_atom
from utils.algorithm_utils_orig import *
from utils.SBM import SBM
from utils.utils import *

# Set print options to show the entire array
np.set_printoptions(threshold=np.inf, suppress=False)

if __name__ == "__main__":
    samples_per_cluster = [
        1000,
        1000,
        1000,
    ]  # Number of training samples in each cluster
    # samples_per_cluster = [600, 400, 200]  # Number of training samples in each cluster
    num_features = 6  # Dimension of the features at each node
    num_signal_features = 2
    n = sum(samples_per_cluster)
    c = len(samples_per_cluster)
    n_clusters = c

    probs = np.full((c, c), config.q)
    np.fill_diagonal(probs, config.p)
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
    ari = adjusted_rand_score(graph.y, cluster_labels)
    plt.figure()
    plt.plot(cluster_labels)
    plt.title("Cluster labels")
    plt.savefig(f"{save_path}/cluster_labels.png")

    if config.warm_start == "random":
        atoms, coefs = create_random_atoms(n, num_features, c, c)
    elif config.warm_start == "optimal":
        atoms, coefs = create_optimal_atoms(graph)
    elif config.warm_start == "cluster":
        atoms, coefs = create_atoms_from__cluster_labels(graph, cluster_labels)
    elif config.warm_start == "load":
        path = "results/20250410_011326/"
        atoms, coefs, graph = load_atom(path)

    (
        atoms,
        coefs,
        graph,
        results,
        f_values,
        accuracies,
        epsilsons,
        components,
        opt_params,
    ) = CANO(graph, atoms, coefs, log_results=False)

    LOGGER = log_results(atoms, coefs, graph, results, save_path)
    LOGGER.info(f"ARI: {ari:.4f}")
    LOGGER.info(f"loss: {f_values[-1]:.4f}")
    plot_results(
        f_values,
        accuracies,
        atoms,
        coefs,
        graph,
        epsilsons,
        components,
        save_path,
        *opt_params,
    )
    save_model_params(atoms, coefs, graph, save_path)
