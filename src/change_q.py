from itertools import cycle
from multiprocessing import Pool, Manager
from matplotlib.colors import to_hex
from tqdm import tqdm
import numpy as np
import pandas as pd
from multiprocessing import Pool
import numpy as np
from tqdm import tqdm

from utils.SBM import SBM
from Graph_Atom.graph_func import GraphFunc

from utils.algorithm_utils_orig import *
from utils.utils import *

# Set print options to show the entire array
np.set_printoptions(threshold=np.inf, suppress=False)
# np.set_printoptions(suppress=True)


def process_task(args):
    # print("start")
    ind, total, noisy_sigs, c, q, samples_per_cluster, num_signal_features, config = (
        args
    )
    num_features = noisy_sigs[0].Rv.shape[1]
    probs = np.full((c, c), q)
    np.fill_diagonal(probs, 0.3)

    f_vals = []
    f_opts = []
    acc = 0
    for iter in range(total):
        if iter % 100 == 0:
            graph = SBM(
                samples_per_cluster,
                probs,
                num_features,
                num_signal_features,
                train_ratio=config.train_ratio,
            )
            opt_atoms, opt_coefs = create_optimal_atoms(graph)
            opt_sig = opt_atoms.calc_sig(opt_coefs)
            f = GraphFunc(graph=graph, mu1=config.mu1, mu2=config.mu2)
            f_opt = f(opt_sig)
            f_opts.append(f_opt)

        noisy_sig = noisy_sigs[iter]
        f_noisy = f(noisy_sig)
        f_vals.append(f_noisy)
        if f_noisy > f_opt:
            acc += 1.0

    # print("end")

    return {
        # "index": ind,
        # "mu1": config.mu1,
        # "mu2": config.mu2,
        "q_value": q,
        "mean_f_vals": np.mean(f_vals),
        "std_f_vals": np.std(f_vals),
        "mean_f_opts": np.mean(f_opts),
        "std_f_opts": np.std(f_opts),
        "accuracy": acc / total,
    }


def parallel_processing(
    total, noisy_sigs, c, q_vals, samples_per_cluster, num_signal_features, config
):
    num_workers = 4  # Adjust the number of workers as needed
    args = [
        (i, total, noisy_sigs, c, q, samples_per_cluster, num_signal_features, config)
        for i, q in enumerate(q_vals)
    ]

    results = []
    with Manager() as manager:
        with Pool(num_workers) as pool:
            with tqdm(total=len(args)) as pbar:
                for result in pool.imap(process_task, args):
                    results.append(result)
                    # Update tqdm with additional information
                    pbar.set_postfix(result)
                    pbar.update()

    return results


def change_mu(noisy_sig, u1, u2, ax_acc, main_results=False, color_map=None):
    config.mu1 = u1
    config.mu2 = u2
    results = parallel_processing(
        total, noisy_sig, c, q_vals, samples_per_cluster, num_signal_features, config
    )
    alpha = 1 if main_results else 0.5
    ax_acc.plot(
        q_vals,
        [result["accuracy"] for result in results],
        "-*",
        label=f"Accuracy for u1={u1} and u2={u2}",
        alpha=alpha,
        color=color_map,
    )
    ax_acc.set_xlabel("q")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.legend(prop={"size": 8}, markerscale=0.0)

    return results


def change_num_atoms(num_atoms, color_map):
    results_path = f"{save_path}num_atoms({num_atoms})/"
    os.makedirs(results_path, exist_ok=True)
    noisy_sigs = []
    for i in range(total):
        atoms, coefs = create_random_atoms(n, num_features, c, num_atoms)
        noisy_sig = atoms.calc_sig(coefs)
        noisy_sig.L = project_to_bound(noisy_sig.L)
        noisy_sigs.append(noisy_sig)

    u1_vals = [0.05, 0.0]
    for u1 in u1_vals:
        if u1 == 0:
            u2_vals = [0, 1]
        else:
            u2_vals = [1]
        for u2 in u2_vals:
            results = change_mu(
                noisy_sigs, u1, u2, ax_acc, num_atoms == 3, color_map[(u1, u2)]
            )
            pd.DataFrame(results).to_csv(f"{results_path}results_u0({u1})_u1({u2}).csv")


if __name__ == "__main__":
    # Example usage
    total = 2000
    c = 3
    # num_signal_features = 10
    num_features = 10  # Dimension of the features at each node
    num_signal_features = 2
    samples_per_cluster = [50] * 3  # Number of training samples in each cluster
    n = sum(samples_per_cluster)
    c = len(samples_per_cluster)
    config.train_ratio = 0.5
    q_vals = np.arange(0.0, 1, 0.02)

    color_cycle = cycle(plt.cm.tab10.colors)
    color_map = {}
    u1_vals = [0.05, 0.0]
    for u1 in u1_vals:
        if u1 == 0:
            u2_vals = [0, 1]
        else:
            u2_vals = [1]
        for u2 in u2_vals:
            color_map[(u1, u2)] = to_hex(next(color_cycle))

    fig_acc, ax_acc = plt.subplots()
    for num_atoms in range(1, 7):
        change_num_atoms(num_atoms, color_map)
    fig_acc.savefig(f"{save_path}accuracy.png")
