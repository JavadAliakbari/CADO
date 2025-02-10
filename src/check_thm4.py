from multiprocessing import Pool
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

from utils.SBM import SBM
from Atom.coefs import Coefs
from Atom.atoms import Atoms
from Graph_Atom.graph_func import GraphFunc

from utils.algorithm_utils_orig import *
from utils.utils import *

# Set print options to show the entire array
np.set_printoptions(threshold=np.inf, suppress=False)
# np.set_printoptions(suppress=True)


def run_experiment(
    noisy_sigs,
    # q,
    # samples_per_cluster,
    # num_signal_features,
    # total,
):
    q = 0.1
    """
    Run the experiment with the given parameters.

    Args:
        total (int): The total number of iterations.
        samples_per_cluster (int): The number of samples per cluster.
        probs (np.ndarray): The probability matrix.
        num_features (int): The number of features.
        num_signal_features (int): The number of signal features.
        noisy_E (np.ndarray): The noisy E matrix.
        noisy_coef (np.ndarray): The noisy coefficients.

    Returns:
        list: The f_vals list.
        list: The f_opts list.
        float: The accuracy.
    """
    num_features = noisy_sigs[0].Rv.shape[1]
    probs = np.full((c, c), q)
    np.fill_diagonal(probs, 0.3)
    print("fooooooooooyredxxcvbxzzaqa`````zcvbnmkl;;;;;;pooiuytrè12oo")

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
            )
            opt_atoms, opt_coefs = create_optimal_atoms(graph)
            opt_sig = opt_atoms.calc_sig(opt_coefs)
            f = GraphFunc(graph=graph)
            f_opt = f(opt_sig)
            f_opts.append(f_opt)
            print(f"bar{iter}")

        noisy_sig = noisy_sigs[iter]
        f_noisy = f(noisy_sig)
        f_vals.append(f_noisy)
        if f_noisy > f_opt:
            acc += 1.0

    return [np.mean(f_vals), np.std(f_vals), np.mean(f_opts), np.std(f_opts), acc]


result_list = []


def log_result(result):
    # This is called whenever foo_pool(i) returns a result.
    # result_list is modified only by the main process, not the pool workers.
    result_list.append(result)
    # f_rnd_avg.append(np.mean(f_vals))
    # f_rnd_std.append(np.std(f_vals))
    # f_opt_avg.append(np.mean(f_opts))
    # f_opt_std.append(np.std(f_opts))
    # accuracies.append(acc / total)
    # bar.set_postfix_str(
    #     f"q={q:0.2f}, f_avg={f_rnd_avg[-1]:.2f}, f_opt={f_opt_avg[-1]:.2f}, acc={accuracies[-1]:.2f}"
    # )
    # bar.update(1)


def run_outer_loop(
    q_vals, total, samples_per_cluster, num_features, num_signal_features
):
    f_rnd_avg = []
    f_rnd_std = []
    f_opt_avg = []
    f_opt_std = []
    accuracies = []
    # q_vals = np.arange(0.4, 1, 0.02)
    # bar = tqdm(total=len(q_vals))
    # Add noise to the signal
    # total = 100000

    noisy_sigs = []
    for i in range(total):
        atoms, coefs = create_random_atoms(n, num_features, c, c)
        noisy_sig = atoms.calc_sig(coefs)
        noisy_sig.L = project_to_bound(noisy_sig.L)
        noisy_sigs.append(noisy_sig)

    # pool = Pool()
    with Pool() as pool:
        rrr = []
        for q in q_vals:
            res = pool.apply_async(
                run_experiment,
                args=(
                    noisy_sigs,
                    # q,
                    # samples_per_cluster,
                    # num_signal_features,
                    # total,
                ),
                callback=log_result,
            )
            rrr.append(res)
        for result in rrr:
            result.wait()
    # pool.close()
    # pool.join()
    # noisy_means, noisy_std, opt_mean, opt_std, acc = pool.map(run_experiment, args)

    # f_vals, f_opts, acc = run_experiment(
    #     noisy_sigs,
    #     probs,
    #     samples_per_cluster,
    #     num_signal_features,
    #     total,
    # )
    # f_rnd_avg.append(np.mean(f_vals))
    # f_rnd_std.append(np.std(f_vals))
    # f_opt_avg.append(np.mean(f_opts))
    # f_opt_std.append(np.std(f_opts))
    # accuracies.append(acc / total)
    # bar.set_postfix_str(
    #     f"q={q:0.2f}, f_avg={f_rnd_avg[-1]:.2f}, f_opt={f_opt_avg[-1]:.2f}, acc={accuracies[-1]:.2f}"
    # )
    # bar.update(1)

    return (
        np.array(f_rnd_avg),
        np.array(f_rnd_std),
        np.array(f_opt_avg),
        np.array(f_opt_std),
        np.array(accuracies),
    )


if __name__ == "__main__":

    # Example usage
    # samples_per_cluster = [3, 4, 5]  # Number of training samples in each cluster
    num_features = 4  # Dimension of the features at each node
    samples_per_cluster = [20] * 3  # Number of training samples in each cluster
    num_signal_features = 2
    n = sum(samples_per_cluster)
    c = len(samples_per_cluster)
    config.tau = n
    total = 100
    q_vals = np.arange(0.0, 1, 0.02)

    # Initialize lists to store results for both u1 cases
    f_rnd_avg_all = []
    f_rnd_std_all = []
    f_opt_avg_all = []
    f_opt_std_all = []
    accuracies_all = []

    u1_vals = [0, 1]

    for u1 in u1_vals:
        config.mu1 = u1
        if u1 == 0:
            u2_vals = [0, 1]
        else:
            u2_vals = [1]

        for u2 in u2_vals:
            config.mu2 = u2

            if u1 != 0:
                total1 = 1 * total
            else:
                total1 = total

            f_rnd_avg, f_rnd_std, f_opt_avg, f_opt_std, accuracies = run_outer_loop(
                q_vals, total1, samples_per_cluster, num_features, num_signal_features
            )
            f_rnd_avg_all.append(f_rnd_avg)
            f_rnd_std_all.append(f_rnd_std)
            f_opt_avg_all.append(f_opt_avg)
            f_opt_std_all.append(f_opt_std)
            accuracies_all.append(accuracies)

    # Plot the objective function for both u1 cases
    plt.figure()
    i = 0
    for u1 in u1_vals:
        config.mu1 = u1
        if u1 == 0:
            u2_vals = [0, 1]
        else:
            u2_vals = [1]

        for u2 in u2_vals:
            plt.plot(
                q_vals,
                f_rnd_avg_all[i],
                "-*",
                label=f"$f(\Psi)$ for u1={u1} and u2= {u2}",
            )
            plt.fill_between(
                q_vals,
                f_rnd_avg_all[i] - f_rnd_std_all[i],
                f_rnd_avg_all[i] + f_rnd_std_all[i],
                alpha=0.2,
                label=f"Std Dev for u1={u1}",
            )
            plt.plot(
                q_vals,
                f_opt_avg_all[i],
                "-*",
                label=f"$f(\Psi^*)$ for u1={u1} and u2= {u2}",
            )
            plt.fill_between(
                q_vals,
                f_opt_avg_all[i] - f_opt_std_all[i],
                f_opt_avg_all[i] + f_opt_std_all[i],
                alpha=0.2,
            )
            i += 1
    plt.xlabel("q")
    plt.ylabel("$f(\Phi)$")
    plt.title("Objective Function Comparison for u1=0 and u1=1")
    plt.legend()
    plt.savefig("Objective_Comparison.png")
    # plt.show()

    # Plot the accuracy for both u1 cases
    plt.figure()
    i = 0
    for u1 in u1_vals:
        config.mu1 = u1
        if u1 == 0:
            u2_vals = [0, 1]
        else:
            u2_vals = [1]

        for u2 in u2_vals:
            plt.plot(
                q_vals,
                accuracies_all[i],
                "-*",
                label=f"Accuracy for u1={u1} and u2={u2}",
            )
            i += 1
    plt.plot(q_vals, np.ones_like(q_vals))
    plt.xlabel("q")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison for u1=0 and u1=1")
    plt.legend()
    plt.savefig("Accuracy_Comparison.png")
    plt.show()
