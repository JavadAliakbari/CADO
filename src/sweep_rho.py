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
from utils.utils import data
from utils.algorithm_utils import create_random_atoms
from utils.utils import data, save_path


# helper function to run a single run of the experiment
def run_single(run, rho_min):
    # build SBM
    # data.sig_rho_min = float(rho_min)
    c = len(data.samples_per_cluster)
    probs = np.full((c, c), data.q)
    np.fill_diagonal(probs, data.p)
    graph = SBM(
        data.samples_per_cluster,
        probs,
        data.num_features,
        data.num_signal_features,
        train_ratio=data.train_ratio,
        rho_min=rho_min,
        rho_max=data.sig_rho_max,
    )
    # random warm start
    orig_atoms, orig_coefs = create_random_atoms(
        sum(data.samples_per_cluster), data.num_features, c, c
    )

    # First run
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

    # Second run with mu1=0
    atoms, coefs = deepcopy(orig_atoms), deepcopy(orig_coefs)
    _, _, _, _, f_vals1, accs1, *_ = CANO(
        graph,
        atoms,
        coefs,
        lr_search=False,
        log_results=False,
        leave=False,
        mu0=0,
    )
    loss_final1 = f_vals1[-1]
    acc_final1 = accs1[-1]

    # Third run with mu2=0
    atoms, coefs = deepcopy(orig_atoms), deepcopy(orig_coefs)
    _, _, _, _, f_vals2, accs2, *_ = CANO(
        graph,
        atoms,
        coefs,
        lr_search=False,
        log_results=False,
        leave=False,
        mu2=0,
    )
    loss_final2 = f_vals2[-1]
    acc_final2 = accs2[-1]

    # Fourth run with mu1=0, mu2=0
    atoms, coefs = deepcopy(orig_atoms), deepcopy(orig_coefs)
    _, _, _, _, f_vals3, accs3, *_ = CANO(
        graph,
        atoms,
        coefs,
        lr_search=False,
        log_results=False,
        leave=False,
        mu0=0,
        mu2=0,
    )
    loss_final3 = f_vals3[-1]
    acc_final3 = accs3[-1]

    # Restore the original rho_min
    return {
        "rho_min": np.round(rho_min, 4),
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


def process_rho_min(rho_min, out_dir="./"):
    # execute each run in parallel, with fewer workers and error handling
    # try:
    total_runs = 60  # Reduced number of runs to avoid memory issues
    with concurrent.futures.ProcessPoolExecutor(
        # max_workers=5,
    ) as executor:
        futures = [
            executor.submit(run_single, run, rho_min) for run in range(total_runs)
        ]
        results = []
        # Using tqdm to track completion of run_single calls
        bar = tqdm(
            total=len(futures),
            desc=f"rho_min={rho_min:.4f} runs",
            leave=False,
        )
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
                bar.set_postfix(
                    {
                        "acc0": results[-1]["acc0"],
                        "acc1": results[-1]["acc1"],
                        "acc2": results[-1]["acc2"],
                        "acc3": results[-1]["acc3"],
                        # "sc_acc": results[-1]["sc_acc"],
                    }
                )
                bar.update()
            except Exception as e:
                print(f"Error in worker process: {e}")
                # Continue with other processes instead of crashing
    # except concurrent.futures.process.BrokenProcessPool:
    #     print(
    #         f"Process pool broke for rho_min={rho_min}. Switching to sequential execution."
    #     )
    #     # Fall back to sequential execution
    #     results = []
    #     for run in tqdm(
    #         range(total_runs),
    #         desc=f"rho_min={rho_min:.4f} runs (sequential)",
    #         leave=False,
    #     ):
    #         try:
    #             result = run_single(run, rho_min)
    #             results.append(result)
    #         except Exception as e:
    #             print(f"Error in run {run}: {e}")
    #             # Continue with other runs

    # save per-rho_min CSV
    rho_min_csv = os.path.join(out_dir, f"sweep_results_rho_min{rho_min:.4f}.csv")
    df = pd.DataFrame(results)
    df.to_csv(rho_min_csv, index=False)
    mean_acc0 = df["acc0"].mean()
    mean_acc1 = df["acc1"].mean()
    mean_acc2 = df["acc2"].mean()
    mean_acc3 = df["acc3"].mean()
    # mean_sc_acc = df["sc_acc"].mean()
    return mean_acc0, mean_acc1, mean_acc2, mean_acc3


def main():
    # Define the range of rho_min values to sweep
    rho_min_values = np.arange(0.0, 5, 0.25)  # From 0.01 to 0.2 with 10 points

    # dispatch all rho_min in parallel
    path = "sweep rho_min/"
    out_dir = os.path.join(save_path, path)
    os.makedirs(out_dir, exist_ok=True)

    bar = tqdm(rho_min_values, unit="rho_min", leave=False)
    for rho_min in rho_min_values:
        mean_acc0, mean_acc1, mean_acc2, mean_acc3 = process_rho_min(
            rho_min, out_dir=out_dir
        )
        bar.set_postfix(
            {
                "rho_min": rho_min,
                "acc0": mean_acc0,
                "acc1": mean_acc1,
                "acc0": mean_acc2,
                "acc1": mean_acc3,
                # "sc_acc": mean_sc_acc,
            }
        )
        bar.update()

    # Plot the results
    plot(out_dir)


if __name__ == "__main__":
    main()
