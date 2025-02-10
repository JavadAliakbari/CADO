from multiprocessing import Pool, Manager
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
        ind,
        total,
        noisy_sigs,
        c,
        rho_plus,
        samples_per_cluster,
        num_signal_features,
        config,
    ) = args
    num_features = noisy_sigs[0].Rv.shape[1]
    probs = np.full((c, c), config.q)
    np.fill_diagonal(probs, config.p)
    # config.sig_rho_max = rho_plus * config.sig_rho_min
    config.rho_max = rho_plus * config.rho_min

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
                train_ratio=0.5,
                rho_min=config.rho_min,
                rho_max=config.rho_max,
            )
            opt_atoms, opt_coefs = create_optimal_atoms(graph)
            opt_sig = opt_atoms.calc_sig(opt_coefs)
            f = GraphFunc(graph=graph, mu1=config.mu1, mu2=config.mu2)
            f_opt = f(opt_sig)
            f_opts.append(f_opt)

        atoms, coefs = create_random_atoms(opt_sig.n, num_features, c, c)
        noisy_sig = atoms.calc_sig(coefs)
        noisy_sig.L = project_to_bound(noisy_sig.L)
        # noisy_sig = noisy_sigs[iter]
        f_noisy = f(noisy_sig)
        f_vals.append(f_noisy)
        if f_noisy > f_opt:
            acc += 1.0

    # print("end")

    return {
        # "index": ind,
        # "mu1": config.mu1,
        # "mu2": config.mu2,
        # "q_value": q,
        "rho_plus": rho_plus,
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
    rho_plus_range,
    samples_per_cluster,
    num_signal_features,
    config,
):
    num_workers = 6  # Adjust the number of workers as needed
    args = [
        (
            i,
            total,
            noisy_sigs,
            c,
            rho_plus,
            samples_per_cluster,
            num_signal_features,
            config,
        )
        for i, rho_plus in enumerate(rho_plus_range)
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


def change_rho_plus(ax_f, ax_acc, rho_plus_range):
    results = parallel_processing(
        total,
        noisy_sigs,
        c,
        rho_plus_range,
        samples_per_cluster,
        num_signal_features,
        config,
    )
    pd.DataFrame(results).to_csv(f"{save_path}results_rhoplus.csv")

    # Save results
    # plt.figure()
    ax_f.plot(
        rho_plus_range,
        [result["mean_f_vals"] for result in results],
        "-*",
        label=f"$f(\Psi)$",
    )
    ax_f.plot(
        rho_plus_range,
        [result["mean_f_opts"] for result in results],
        "-*",
        label=f"$f(\Psi^*)$",
    )
    ax_f.fill_between(
        rho_plus_range,
        [result["mean_f_vals"] - result["std_f_vals"] for result in results],
        [result["mean_f_vals"] + result["std_f_vals"] for result in results],
        alpha=0.3,
    )
    ax_f.fill_between(
        rho_plus_range,
        [result["mean_f_opts"] - result["std_f_opts"] for result in results],
        [result["mean_f_opts"] + result["std_f_opts"] for result in results],
        alpha=0.3,
    )
    ax_f.set_xlabel("$\\frac{\\rho^+}{\\rho^-}$")
    ax_f.set_ylabel("Metrics")
    ax_f.set_title("f Values")
    ax_f.legend()
    # plt.close()

    ax_acc.plot(
        rho_plus_range,
        [result["accuracy"] for result in results],
        "-*",
        # label=f"Accuracy for $\\rho^+$={u1} and u2={u2}",
    )
    ax_acc.set_xlabel("$\\rho^+$")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.legend()


if __name__ == "__main__":
    # Example usage
    total = 4000
    c = 3
    # num_signal_features = 10
    num_features = 10  # Dimension of the features at each node
    num_signal_features = 2
    samples_per_cluster = [25] * 3  # Number of training samples in each cluster
    n = sum(samples_per_cluster)
    c = len(samples_per_cluster)
    num_atoms = c
    config.mu1 = 0.1
    config.mu2 = 1
    config.p = 0.2
    config.q = 0.55
    config.train_ratio = 0.2
    config.rho_min = 0.5
    config.sig_rho_min = 0.5
    config.sig_rho_max = 5 * config.sig_rho_min

    os.makedirs(save_path, exist_ok=True)
    noisy_sigs = []
    for i in range(total):
        atoms, coefs = create_random_atoms(n, num_features, c, num_atoms)
        noisy_sig = atoms.calc_sig(coefs)
        noisy_sig.L = project_to_bound(noisy_sig.L)
        noisy_sigs.append(noisy_sig)

    fig_f, ax_f = plt.subplots()
    fig_acc, ax_acc = plt.subplots()
    rho_plus_range = np.arange(1, 5, 0.1)
    change_rho_plus(ax_f, ax_acc, rho_plus_range)

    fig_f.savefig(f"{save_path}results.png")
    fig_acc.savefig(f"{save_path}accuracy.png")
