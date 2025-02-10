import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
from tqdm import tqdm

from CANO import CANO
from Graph_Atom.ADMM import ADMM_cls
from load_atom import load_atom
from main import CANO2
from utils.algorithm_utils_orig import *
from utils.SBM import SBM
from utils.utils import *

# Set print options to show the entire array
np.set_printoptions(threshold=np.inf, suppress=False)

if __name__ == "__main__":
    samples_per_cluster = [100, 100, 100]  # Number of training samples in each cluster
    num_features = 6  # Dimension of the features at each node
    num_signal_features = 2
    n = sum(samples_per_cluster)
    c = len(samples_per_cluster)
    n_clusters = c

    ##### add for loop for p and q for q in range(0.,1.,0.1):
    simulation_number = 1

    acc_clustering_avg = []
    acc_clustering_std = []
    acc_our_method_avg = []
    acc_our_method_std = []

    f_clustering_avg = []
    f_clustering_std = []
    f_our_method_avg = []
    f_our_method_std = []

    p_vals = np.arange(0.4, 0.04, -0.02)
    total_runs = 50
    bar = tqdm(total=len(p_vals))

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

    admm = None
    if config.warm_start == "random":
        atoms, coefs = create_random_atoms(n, num_features, c, c)
    elif config.warm_start == "optimal":
        atoms, coefs = create_optimal_atoms(graph)
    elif config.warm_start == "cluster":
        atoms, coefs = create_atoms_from__cluster_labels(graph, cluster_labels)
    elif config.warm_start == "load":
        path = "results/20250203_004544/"
        atoms, coefs, graph, Lp, Lambda = load_atom(path)
        admm = ADMM_cls(Lp, project_to_bound)
        admm.Z = Lambda

    sig = atoms.calc_sig(coefs)
    f = GraphFunc(graph=graph)
    f_sc = f(sig)

    (
        atoms,
        coefs,
        graph,
        results,
        f_values,
        accuracies,
        f_opt,
        admm,
        Lambda_norms,
        epsilsons,
        components,
    ) = CANO2(graph, atoms, coefs, admm, log_results=False)
    # ) = CANO(graph)
    # ) = CANO(graph, atoms, coefs, admm)

    log_results(atoms, coefs, graph, results, save_path)
    plot_results(
        f_values,
        accuracies,
        Lambda_norms,
        atoms,
        epsilsons,
        components,
        admm,
        save_path,
        f_opt,
    )
    save_model_params(atoms, coefs, graph, admm, save_path)
