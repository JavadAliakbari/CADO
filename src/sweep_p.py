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
from utils.algorithm_utils import create_random_atoms
from utils.utils import data, save_path


def process_p(p, mu0=None, mu1=None, mu2=None, out_dir="./"):
    # override data.p for this p
    data.p = float(p)

    records_p = []
    for run in range(2):
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

        # run CANO
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
        # tqdm.write(f"[p={p:.3f} run={run}] loss={loss_final:.4f} acc={acc_final:.4f}")
        records_p.append(
            {"p": p, "run": run, "loss": loss_final, "accuracy": acc_final}
        )
    # save per-p CSV
    p_csv = os.path.join(out_dir, f"sweep_results_p{p:.3f}.csv")
    pd.DataFrame(records_p).to_csv(p_csv, index=False)
    return p_csv


# new helper to read per-p CSVs and plot aggregated curves
def plot_aggregate_results(out_dir):
    files = glob.glob(os.path.join(out_dir, "sweep_results_p*.csv"))
    records = []
    for fp in files:
        dfp = pd.read_csv(fp)
        p_val = dfp["p"].iloc[0]
        records.append(
            {"p": p_val, "accuracy": dfp["accuracy"].mean(), "loss": dfp["loss"].mean()}
        )
    summary = pd.DataFrame(records).sort_values("p")

    # plot accuracy vs p
    plt.figure()
    plt.plot(summary["p"], summary["accuracy"], marker="o")
    plt.gca().invert_xaxis()
    plt.xlabel("p")
    plt.ylabel("Accuracy")
    plt.title("Average CANO Accuracy vs p (per-p CSVs)")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "per_p_accuracy_vs_p.png"))
    plt.close()

    # plot loss vs p
    plt.figure()
    plt.plot(summary["p"], summary["loss"], marker="o", color="r")
    plt.gca().invert_xaxis()
    plt.xlabel("p")
    plt.ylabel("Loss")
    plt.title("Average CANO Loss vs p (per-p CSVs)")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "per_p_loss_vs_p.png"))
    plt.close()


def main():
    p_values = np.arange(0.3, 0.0, -0.01)
    # define your mu0, mu1, mu2 here
    mu0_val, mu2_val = data.mu0, data.mu2

    # dispatch all p in parallel
    for path in [ "with R/", "without R/"]:
        out_dir = os.path.join(save_path, path)
        os.makedirs(out_dir, exist_ok=True)

        if path == "without R/":
            mu1_val = 0.0
        else:
            mu1_val = 1.0

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(
                    process_p, float(p), mu0_val, mu1_val, mu2_val, out_dir
                ): p
                for p in p_values
            }
            bar = tqdm(total=len(p_values), desc=f"jobs {path}", leave=False)
            for fut in concurrent.futures.as_completed(futures):
                p = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    tqdm.write(f"Error in p={p:.3f}: {e}")
                finally:
                    bar.update()
            bar.close()


if __name__ == "__main__":
    main()
