#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import concurrent.futures
from copy import deepcopy

from idea5 import CANO
from plot_files2 import plot
from utils.SBM import SBM
from utils.utils import clustring_accuracy, data
from utils.algorithm_utils import create_random_atoms, run_spectral_clustering
from utils.utils import data, save_path
from sklearn.metrics import adjusted_rand_score


# helper function to run a single run of the experiment
def run_single(run, p):
    # build SBM
    c = len(data.samples_per_cluster)
    probs = np.full((c, c), data.q)
    np.fill_diagonal(probs, p)
    graph = SBM(
        data.samples_per_cluster,
        probs,
        data.num_features,
        data.num_signal_features,
        train_ratio=data.train_ratio,
    )
    # random warm start
    orig_atoms, orig_coefs = create_random_atoms(
        sum(data.samples_per_cluster), data.num_features, c, c
    )
    atoms, coefs = deepcopy(orig_atoms), deepcopy(orig_coefs)
    # run CANO with parameters
    _, _, _, _, f_vals1, accs1, *_ = CANO(
        graph,
        atoms,
        coefs,
        mu1=0,
        lr_search=False,
        log_results=False,
        leave=False,
    )
    loss_final0 = f_vals1[-1]
    acc_final0 = accs1[-1]

    atoms, coefs = deepcopy(orig_atoms), deepcopy(orig_coefs)
    _, _, _, _, f_vals2, accs2, *_ = CANO(
        graph,
        atoms,
        coefs,
        mu1=1,
        lr_search=False,
        log_results=False,
        leave=False,
    )
    loss_final1 = f_vals2[-1]
    acc_final1 = accs2[-1]
    # _, _, _, _, f_vals3, accs3, *_ = CANO(
    #     graph,
    #     atoms,
    #     coefs,
    #     mu1=1,
    #     mu2=0,
    #     lr_search=False,
    #     log_results=False,
    #     leave=False,
    # )
    # loss_final2 = f_vals3[-1]
    # acc_final2 = accs3[-1]

    # Compute ARI for spectral clustering
    # Perform spectral clustering using the graph's adjacency matrix (assumed to be graph.A)
    cluster_labels = run_spectral_clustering(graph, c)
    sc_acc = clustring_accuracy(cluster_labels, graph.y)

    return {
        "p": np.round(p, 2),
        "run": run,
        "loss0": np.round(loss_final0, 4),
        "acc0": np.round(acc_final0, 4),
        "loss1": np.round(loss_final1, 4),
        "acc1": np.round(acc_final1, 4),
        "sc_acc": np.round(sc_acc, 4),
    }


def process_p(p, out_dir="./"):
    # override data.p for this p
    data.p = float(p)

    # execute each run in parallel
    with concurrent.futures.ProcessPoolExecutor(
        # max_workers=10
    ) as executor:  # modified to limit to 5 tasks
        total_runs = 200 if p <= 0.1 else 50
        futures = [executor.submit(run_single, run, p) for run in range(total_runs)]
        results = []
        # Using tqdm to track completion of run_single calls
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc=f"p={p:.3f} runs",
            leave=False,
        ):
            results.append(future.result())

    # save per-p CSV
    p_csv = os.path.join(out_dir, f"sweep_results_p{p:.3f}.csv")
    df = pd.DataFrame(results)
    df.to_csv(p_csv, index=False)
    mean_acc0 = df["acc0"].mean()
    mean_acc1 = df["acc1"].mean()
    mean_ari = df["sc_acc"].mean()
    return mean_acc0, mean_acc1, mean_ari


def main():
    p_values = np.arange(0.13, 0.0, -0.01)

    # dispatch all p in parallel
    path = "sweep p/"
    out_dir = os.path.join(save_path, path)
    os.makedirs(out_dir, exist_ok=True)

    bar = tqdm(p_values, unit="p", leave=False)
    for p in p_values:
        mean_acc0, mean_acc1, mean_ari = process_p(p, out_dir=out_dir)
        bar.set_postfix(
            {"p": p, "acc0": mean_acc0, "acc": mean_acc1, "sc_acc": mean_ari}
        )
        bar.update()
        # print(f"Processed p={p:.3f}, saved to {p_csv}")

    plot(out_dir)


if __name__ == "__main__":
    main()
