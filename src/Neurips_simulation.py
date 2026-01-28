import numpy as np

from idea5 import CANO
from load_atom import load_atom
from utils.algorithm_utils import *
from utils.SBM import SBM
from utils.utils import *

# Set print options to show the entire array
np.set_printoptions(threshold=np.inf, suppress=False)

if __name__ == "__main__":
    # Use SBM configuration fields
    samples_per_cluster = data.samples_per_cluster
    num_features = data.num_features
    num_signal_features = data.num_signal_features
    n = sum(samples_per_cluster)
    c = len(samples_per_cluster)
    n_clusters = c

    probs = np.full((c, c), data.q)
    np.fill_diagonal(probs, data.p)
    graph = SBM(
        samples_per_cluster,
        probs,
        num_features,
        num_signal_features,
        train_ratio=data.train_ratio,
    )
    plot_adj(graph.A, name="A")

    # Run spectral clustering
    n_clusters = c  # Number of clusters
    cluster_labels = run_spectral_clustering(graph, n_clusters)
    ari = clustring_accuracy(graph.y, cluster_labels)
    plt.figure()
    plt.plot(cluster_labels)
    plt.title("Cluster labels")
    plt.savefig(f"{save_path}/cluster_labels.png")

    if algorithm.warm_start == "random":
        atoms, coefs = create_random_atoms(n, num_features, c, c)
    if algorithm.warm_start == "similar":
        atoms, coefs = create_similar_atoms(graph, c)
    elif algorithm.warm_start == "optimal":
        atoms, coefs = create_optimal_atoms(graph)
    elif algorithm.warm_start == "cluster":
        atoms, coefs = create_atoms_from__cluster_labels(graph, cluster_labels)
    elif algorithm.warm_start == "load":
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
        R_params,
    ) = CANO(
        graph,
        atoms,
        coefs,
        lr_search=False,
        log_results=True,
    )

    LOGGER = log_results(atoms, coefs, graph, results, save_path)
    LOGGER.info(f"acc: {accuracies[-1]:.4f}")
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
        num_signal_features,
        *R_params,
        *opt_params,
    )
    save_model_params(atoms, coefs, graph, save_path)
