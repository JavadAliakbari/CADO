#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import traceback
from copy import deepcopy

from idea5 import CANO
from plot_files2 import plot
from utils.SBM import SBM
from utils.utils import clustring_accuracy, data
from utils.algorithm_utils import create_random_atoms, run_spectral_clustering
from utils.utils import data, save_path, save_config


# helper function to run a single run of the experiment
def run_single(run, mu0):
    # print(f"Running p={p:.3f} run={run}")
    # build SBM
    c = len(data.samples_per_cluster)
    probs = np.full((c, c), data.q)
    np.fill_diagonal(probs, data.p)
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
    _, _, _, _, f_vals0, accs0, *_ = CANO(
        graph,
        atoms,
        coefs,
        mu0=mu0,
        lr_search=False,
        log_results=False,
        leave=False,
    )
    loss_final0 = f_vals0[-1]
    acc_final0 = accs0[-1]

    atoms, coefs = deepcopy(orig_atoms), deepcopy(orig_coefs)
    _, _, _, _, f_vals1, accs1, *_ = CANO(
        graph,
        atoms,
        coefs,
        mu0=mu0,
        mu1=0,
        lr_search=False,
        log_results=False,
        leave=False,
    )
    loss_final1 = f_vals1[-1]
    acc_final1 = accs1[-1]

    atoms, coefs = deepcopy(orig_atoms), deepcopy(orig_coefs)
    _, _, _, _, f_vals2, accs2, *_ = CANO(
        graph,
        atoms,
        coefs,
        mu0=mu0,
        mu2=0,
        lr_search=False,
        log_results=False,
        leave=False,
    )
    loss_final2 = f_vals2[-1]
    acc_final2 = accs2[-1]

    atoms, coefs = deepcopy(orig_atoms), deepcopy(orig_coefs)
    if mu0 != 0:
        _, _, _, _, f_vals3, accs3, *_ = CANO(
            graph,
            atoms,
            coefs,
            mu0=mu0,
            mu1=0,
            mu2=0,
            lr_search=False,
            log_results=False,
            leave=False,
        )
        loss_final3 = f_vals3[-1]
        acc_final3 = accs3[-1]
    else:
        loss_final3 = 0
        acc_final3 = 1 / graph.num_clusters

    # Compute ARI for spectral clustering
    # Perform spectral clustering using the graph's adjacency matrix (assumed to be graph.A)
    # cluster_labels = run_spectral_clustering(graph, c)
    # sc_acc = clustring_accuracy(cluster_labels, graph.y)

    return {
        "mu0": np.round(mu0, 2),
        "run": run,
        "loss0": np.round(loss_final0, 4),
        "acc0": np.round(acc_final0, 4),
        "loss1": np.round(loss_final1, 4),
        "acc1": np.round(acc_final1, 4),
        "loss2": np.round(loss_final2, 4),
        "acc2": np.round(acc_final2, 4),
        "loss3": np.round(loss_final3, 4),
        "acc3": np.round(acc_final3, 4),
        # "sc_acc": np.round(sc_acc, 4),
    }


def process_p(mu0, out_dir="./", debug_mode=False):
    # override data.p for this p
    results = []

    # For debugging: run a single process directly without parallel execution
    if debug_mode:
        try:
            result = run_single(0, mu0)
            results.append(result)
            print("Debug run successful")
        except Exception as e:
            print(f"Error in debug run: {str(e)}")
            traceback.print_exc()
            return 0.0, 0.0, 0.0, 0.0
    else:
        # execute each run in parallel
        with concurrent.futures.ProcessPoolExecutor(
            # max_workers=10
        ) as executor:  # modified to limit to 5 tasks
            total_runs = 200 if mu0 <= 0.1 else 200
            futures = [
                executor.submit(run_single, run, mu0) for run in range(total_runs)
            ]
            results = []
            # Using tqdm to track completion of run_single calls
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc=f"mu0={mu0:.3f} runs",
                leave=False,
            ):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"\nError in worker process: {str(e)}")
                    print("\nTraceback:")
                    traceback.print_exc()
                    print("\nContinuing with other tasks...")

    mu_csv = os.path.join(out_dir, f"sweep_results_mu0{mu0:.3f}.csv")
    df = pd.DataFrame(results)
    df.to_csv(mu_csv, index=False)
    mean_acc0 = df["acc0"].mean()
    mean_acc1 = df["acc1"].mean()
    mean_acc2 = df["acc2"].mean()
    mean_acc3 = df["acc3"].mean()
    # mean_ari = df["sc_acc"].mean()
    return mean_acc0, mean_acc1, mean_acc2, mean_acc3


def main():
    mu0_values = [0.0, 0.5, 1, 5, 10, 100, 1000]

    # dispatch all mu0 in parallel
    path = "sweep mu0/"
    out_dir = os.path.join(save_path, path)
    os.makedirs(out_dir, exist_ok=True)

    save_config(out_dir, f"config.yml")

    bar = tqdm(mu0_values, unit="mu0", leave=False)
    for mu0 in mu0_values:
        try:
            mean_acc0, mean_acc1, mean_acc2, mean_acc3 = process_p(mu0, out_dir=out_dir)
            bar.set_postfix(
                {
                    "mu0": mu0,
                    "acc0": mean_acc0,
                    "acc1": mean_acc1,
                    "acc2": mean_acc2,
                    "acc3": mean_acc3,
                    # "sc_acc": mean_ari,
                }
            )
        except Exception as e:
            print(f"\nError processing mu0={mu0:.3f}: {str(e)}")
            print("\nTraceback:")
            traceback.print_exc()
            print("\nContinuing with next mu0 value...")
        finally:
            bar.update()

    try:
        plot(out_dir)
    except Exception as e:
        print(f"\nError in plotting: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
    # run_single(100, 0.0)
