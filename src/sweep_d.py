#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import traceback
import sys
import warnings
from copy import deepcopy

# Suppress specific sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from idea5 import CANO
from plot_files2 import plot
from utils.SBM import SBM
from utils.utils import clustering_accuracy, data
from utils.algorithm_utils import create_random_atoms, run_spectral_clustering
from utils.utils import data, save_path, save_config


# helper function to run a single run of the experiment
def run_single(run, num_sig, ratio):
    # print(f"Running p={p:.3f} run={run}")
    # build SBM
    c = len(data.samples_per_cluster)
    probs = np.full((c, c), data.q)
    np.fill_diagonal(probs, data.p)
    num_signal_features = num_sig
    num_features = int(num_signal_features * ratio)
    graph = SBM(
        data.samples_per_cluster,
        probs,
        num_features,
        num_signal_features,
        train_ratio=data.train_ratio,
    )
    # random warm start
    orig_atoms, orig_coefs = create_random_atoms(
        sum(data.samples_per_cluster), num_features, c, c
    )
    atoms, coefs = deepcopy(orig_atoms), deepcopy(orig_coefs)
    _, _, _, _, f_vals0, accs0, *_ = CANO(
        graph,
        atoms,
        coefs,
        lr_search=False,
        log_results=False,
        leave=False,
    )
    loss_final0 = f_vals0[-1]
    acc_final0 = accs0[-1]

    # atoms, coefs = deepcopy(orig_atoms), deepcopy(orig_coefs)
    # _, _, _, _, f_vals1, accs1, *_ = CANO(
    #     graph,
    #     atoms,
    #     coefs,
    #     mu1=0,
    #     lr_search=False,
    #     log_results=False,
    #     leave=False,
    # )
    # loss_final1 = f_vals1[-1]
    # acc_final1 = accs1[-1]

    atoms, coefs = deepcopy(orig_atoms), deepcopy(orig_coefs)
    _, _, _, _, f_vals1, accs1, *_ = CANO(
        graph,
        atoms,
        coefs,
        mu2=0,
        lr_search=False,
        log_results=False,
        leave=False,
    )
    loss_final1 = f_vals1[-1]
    acc_final1 = accs1[-1]

    # atoms, coefs = deepcopy(orig_atoms), deepcopy(orig_coefs)
    # _, _, _, _, f_vals3, accs3, *_ = CANO(
    #     graph,
    #     atoms,
    #     coefs,
    #     mu1=0,
    #     mu2=0,
    #     lr_search=False,
    #     log_results=False,
    #     leave=False,
    # )
    # loss_final3 = f_vals3[-1]
    # acc_final3 = accs3[-1]

    # Compute ARI for spectral clustering
    # Perform spectral clustering using the graph's adjacency matrix (assumed to be graph.A)
    # cluster_labels = run_spectral_clustering(graph, c)
    # sc_acc = clustring_accuracy(cluster_labels, graph.y)

    return {
        "num_sig": np.round(num_sig, 2),
        "run": run,
        "loss0": np.round(loss_final0, 4),
        "acc0": np.round(acc_final0, 4),
        "loss1": np.round(loss_final1, 4),
        "acc1": np.round(acc_final1, 4),
    }


def process_p(num_sig, ratio=3, out_dir="./", debug_mode=False):
    # override data.p for this p
    # data.p = float(p)

    results = []

    # For debugging: run a single process directly without parallel execution
    if debug_mode:
        try:
            result = run_single(0, num_sig, ratio)
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
            total_runs = 200
            futures = [
                executor.submit(run_single, run, num_sig, ratio)
                for run in range(total_runs)
            ]
            results = []
            # Using tqdm to track completion of run_single calls
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc=f"num_sig={num_sig:.3f} runs",
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

    n_csv = os.path.join(out_dir, f"sweep_results_num_sig{num_sig}.csv")
    df = pd.DataFrame(results)
    df.to_csv(n_csv, index=False)
    mean_acc0 = df["acc0"].mean()
    mean_acc1 = df["acc1"].mean()
    return mean_acc0, mean_acc1


def main():
    ratio = 3
    num_signals_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # dispatch all num_sig in parallel
    path = "sweep num_sig/"
    out_dir = os.path.join(save_path, path)
    os.makedirs(out_dir, exist_ok=True)

    save_config(out_dir, f"config.yml")

    bar = tqdm(num_signals_list, unit="p", leave=False)
    for num_sig in num_signals_list:
        try:
            mean_acc0, mean_acc1, mean_acc2, mean_acc3, mean_ari = process_p(
                num_sig, ratio, out_dir=out_dir
            )
            bar.set_postfix(
                {
                    "num_sig": num_sig,
                    "acc0": mean_acc0,
                    "acc1": mean_acc1,
                }
            )
        except Exception as e:
            print(f"\nError processing n={num_sig:.3f}: {str(e)}")
            print("\nTraceback:")
            traceback.print_exc()
            print("\nContinuing with next num_sig value...")
        finally:
            bar.update()

    try:
        plot(out_dir)
    except Exception as e:
        print(f"\nError in plotting: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
    # run_single(10, 1, 3)
