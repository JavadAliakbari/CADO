#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import concurrent.futures

from idea5 import CANO
from utils.SBM import SBM
from utils.utils import data
from utils.algorithm_utils import create_random_atoms, run_spectral_clustering
from utils.utils import data, save_path
from sklearn.metrics import adjusted_rand_score  # new import
from sklearn.cluster import SpectralClustering  # new import


# helper function to run a single run of the experiment
def run_single(
    run,
    p,
    mu0=None,
    mu1=None,
    mu2=None,
):
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
    atoms, coefs = create_random_atoms(
        sum(data.samples_per_cluster), data.num_features, c, c
    )
    # run CANO with parameters
    _, _, _, _, f_vals, accs, *_ = CANO(
        graph,
        atoms,
        coefs,
        mu0=mu0,
        mu1=mu1,
        mu2=mu2,
        lr_search=False,
        log_results=False,
        leave=False,
    )
    loss_final = f_vals[-1]
    acc_final = accs[-1]

    # Compute ARI for spectral clustering
    # Perform spectral clustering using the graph's adjacency matrix (assumed to be graph.A)
    cluster_labels = run_spectral_clustering(graph, c)
    ari = adjusted_rand_score(graph.y, cluster_labels)

    return {
        "p": np.round(p, 2),
        "run": run,
        "loss": np.round(loss_final, 4),
        "accuracy": np.round(acc_final, 4),
        "ari": np.round(ari, 4),
    }


def process_p(p, mu0=None, mu1=None, mu2=None, out_dir="./"):
    # override data.p for this p
    data.p = float(p)

    # execute each run in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(run_single, run, p, mu0, mu1, mu2) for run in range(50)
        ]
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
    mean_loss = df["loss"].mean()
    mean_accuracy = df["accuracy"].mean()
    mean_ari = df["ari"].mean()
    return mean_loss, mean_accuracy,   mean_ari


def main():
    p_values = np.arange(0.2, 0.0, -0.01)
    # define your mu0, mu1, mu2 here
    mu0_val, mu2_val = data.mu0, data.mu2

    # dispatch all p in parallel
    for path in ["with R/", "without R/"]:
        out_dir = os.path.join(save_path, path)
        os.makedirs(out_dir, exist_ok=True)

        if path == "without R/":
            mu1_val = 0.0
        else:
            mu1_val = 1.0

        bar = tqdm(p_values, unit="p", leave=False)
        for p in p_values:
            mean_loss, mean_acc, mean_ari = process_p(
                p, mu0=mu0_val, mu1=mu1_val, mu2=mu2_val, out_dir=out_dir
            )
            bar.set_postfix({"p": p, "loss": mean_loss, "acc": mean_acc, "ari": mean_ari})
            bar.update()
            # print(f"Processed p={p:.3f}, saved to {p_csv}")


if __name__ == "__main__":
    main()
