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

    for ind, p in enumerate(p_vals):
        acc_clustering = []
        acc_our_method = []

        f_clustering = []
        f_our_method = []

        for rand_exp in tqdm(range(total_runs), leave=False):

            probs = np.full((c, c), config.q)
            np.fill_diagonal(probs, p)
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
            cluster_labels = run_spectral_clustering(graph, n_clusters)
            # Calculate ARI score
            ari = adjusted_rand_score(graph.y, cluster_labels)
            acc_clustering.append(ari)

            atoms, coefs = create_atoms_from__cluster_labels(graph, cluster_labels)
            sig = atoms.calc_sig(coefs)
            f = GraphFunc(graph=graph)
            f_sc = f(sig)
            f_clustering.append(f_sc)

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
            ) = CANO(graph, atoms, coefs, log_results=False)
            # ) = CANO(graph)
            # ) = CANO(graph, atoms, coefs, admm)

            # log_results(atoms, coefs, graph, results, save_path)
            plot_results(
                f_values,
                accuracies,
                Lambda_norms,
                atoms,
                admm.Lambda,
                save_path,
                f_opt,
            )
            save_model_params(atoms, coefs, graph, admm, save_path)
            acc_our_method.append(accuracies[-1])
            f_our_method.append(f_values[-1])

        acc_clustering_avg.append(np.mean(acc_clustering))
        acc_clustering_std.append(np.std(acc_clustering))
        acc_our_method_avg.append(np.mean(acc_our_method))
        acc_our_method_std.append(np.std(acc_our_method))

        f_clustering_avg.append(np.mean(f_clustering))
        f_clustering_std.append(np.std(f_clustering))
        f_our_method_avg.append(np.mean(f_our_method))
        f_our_method_std.append(np.std(f_our_method))

        bar.set_postfix_str(
            f"p={p:0.2f}, acc_clustering_avg={acc_clustering_avg[-1]:.2f}, acc_our_method_avg={acc_our_method_avg[-1]:.2f}"
        )
        bar.update(1)

        plt.figure()
        plt.plot(p_vals[: ind + 1], acc_clustering_avg, "*", label="clustering")
        acc_clustering_avg_ = np.array(acc_clustering_avg)
        acc_clustering_std_ = np.array(acc_clustering_std)
        plt.fill_between(
            p_vals[: ind + 1],
            acc_clustering_avg_ - acc_clustering_std_,
            acc_clustering_avg_ + acc_clustering_std_,
            color="b",
            alpha=0.2,
            # label="Standard Deviation",
        )
        plt.plot(p_vals[: ind + 1], acc_our_method_avg, "*", label="CANO")
        acc_our_method_avg_ = np.array(acc_our_method_avg)
        acc_our_method_std_ = np.array(acc_our_method_std)
        plt.fill_between(
            p_vals[: ind + 1],
            acc_our_method_avg_ - acc_our_method_std_,
            acc_our_method_avg_ + acc_our_method_std_,
            color="r",
            alpha=0.2,
            # label="Standard Deviation",
        )
        plt.xlabel("p")
        plt.ylabel("accuracy")
        plt.legend()
        plt.savefig(f"{save_path}/accuracy.png")
        plt.close()

        plt.figure()
        plt.plot(p_vals[: ind + 1], f_clustering_avg, "*", label="clustering")
        f_clustering_avg_ = np.array(f_clustering_avg)
        f_clustering_std_ = np.array(f_clustering_std)
        plt.fill_between(
            p_vals[: ind + 1],
            f_clustering_avg_ - f_clustering_std_,
            f_clustering_avg_ + f_clustering_std_,
            color="b",
            alpha=0.2,
            # label="Standard Deviation",
        )
        plt.plot(p_vals[: ind + 1], f_our_method_avg, "*", label="CANO")
        f_our_method_avg_ = np.array(f_our_method_avg)
        f_our_method_std_ = np.array(f_our_method_std)
        plt.fill_between(
            p_vals[: ind + 1],
            f_our_method_avg_ - f_our_method_std_,
            f_our_method_avg_ + f_our_method_std_,
            color="r",
            alpha=0.2,
            # label="Standard Deviation",
        )
        plt.xlabel("p")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(f"{save_path}/lossssssss.png")
        plt.close()
