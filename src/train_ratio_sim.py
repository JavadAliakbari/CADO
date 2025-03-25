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
    (
        total,
        noisy_sigs,
        c,
        train_ratio,
        samples_per_cluster,
        num_signal_features,
        config,
    ) = args
    num_features = noisy_sigs[0].Rv.shape[1]
    probs = np.full((c, c), config.q)
    np.fill_diagonal(probs, config.p)

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
                train_ratio=train_ratio,
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
        "train_ratio": train_ratio,
        "mean_f_vals": np.mean(f_vals),
        "std_f_vals": np.std(f_vals),
        "mean_f_opts": np.mean(f_opts),
        "std_f_opts": np.std(f_opts),
        "accuracy": acc / total,
    }


def parallel_processing(
    total,
    noisy_sigs,
    c,
    train_ratio_range,
    samples_per_cluster,
    num_signal_features,
    config,
):
    num_workers = 4  # Adjust the number of workers as needed
    args = [
        (
            total,
            noisy_sigs,
            c,
            train_ratio,
            samples_per_cluster,
            num_signal_features,
            config,
        )
        for train_ratio in train_ratio_range
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


def change_mu(noisy_sig):
    results = parallel_processing(
        total,
        noisy_sig,
        c,
        train_ratio_range,
        samples_per_cluster,
        num_signal_features,
        config,
    )

    return results


def change_num_atoms(mu2):
    config.mu2 = mu2
    results_path = f"{save_path}mu2({mu2})/"
    os.makedirs(results_path, exist_ok=True)
    noisy_sigs = []
    for i in range(total):
        atoms, coefs = create_random_atoms(n, num_features, c, c)
        noisy_sig = atoms.calc_sig(coefs)
        noisy_sig.L = project_to_bound(noisy_sig.L)
        noisy_sigs.append(noisy_sig)

    results = change_mu(noisy_sigs)
    pd.DataFrame(results).to_csv(f"{results_path}results_mu2({mu2}).csv")

    return results


if __name__ == "__main__":
    # Example usage
    total = 10000
    c = 3
    # num_signal_features = 10
    num_features = 10  # Dimension of the features at each node
    num_signal_features = 2
    samples_per_cluster = [50] * 3  # Number of training samples in each cluster
    n = sum(samples_per_cluster)
    c = len(samples_per_cluster)
    train_ratio_range = np.arange(0.0, 1, 0.05)
    config.mu1 = 0.0
    config.p = 0.3
    config.q = 0.55

    results = []
    for mu2 in range(1, 10):
        result = change_num_atoms(mu2)
        results.append([res["accuracy"] for res in result])

    results = np.array(results)
    mean_acc = np.mean(results, axis=0)
    std_acc = np.std(results, axis=0)

    fig_acc, ax_acc = plt.subplots()
    ax_acc.plot(
        train_ratio_range,
        mean_acc,
        "-*",
        label=f"Accuracy $\mu_2$={config.mu2}",
    )
    ax_acc.fill_between(
        train_ratio_range,
        mean_acc - std_acc,
        mean_acc + std_acc,
        alpha=0.3,
    )
    ax_acc.set_xlabel("q")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.legend(prop={"size": 8}, markerscale=0.0)
    fig_acc.savefig(f"{save_path}accuracy.png")
