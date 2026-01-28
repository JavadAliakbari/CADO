#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import glob

from utils.utils import *


def plot(out_dir):
    # read all per-mu2 CSVs
    files = glob.glob(os.path.join(out_dir, "sweep_results_mu2*.csv"))
    records = []
    for fp in files:
        dfp = pd.read_csv(fp)
        records.append(dfp)

    # save raw results
    df = pd.concat(records, ignore_index=True)
    csv_path = os.path.join(out_dir, "sweep_results.csv")
    df.to_csv(csv_path, index=False)

    # compute per-mu2 averages
    summary = df.groupby("mu2").mean().reset_index()
    summary.to_csv(os.path.join(out_dir, "sweep_results_mu2_summary.csv"), index=False)

    # plot accuracy vs mu2
    plt.figure()
    plt.plot(summary["mu2"], summary["acc1"], marker="o", label="with L", color="b")
    plt.plot(summary["mu2"], summary["acc0"], marker="o", label="without L", color="r")
    plt.xscale("log")  # Set x-axis to logarithmic scale
    plt.xlabel("$\\mu_{0}$ (log scale)")
    plt.ylabel("Accuracy")
    plt.title("Average CANO Accuracy vs $\\mu_{0}$")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(out_dir, "accuracy_vs_mu2.png"))
    plt.close()

    # plot loss vs mu2
    plt.figure()
    plt.plot(summary["mu2"], summary["loss1"], marker="o", color="b", label="with L")
    plt.plot(summary["mu2"], summary["loss0"], marker="o", color="r", label="without L")
    plt.xscale("log")  # Set x-axis to logarithmic scale
    plt.xlabel("$\\mu_{0}$ (log scale)")
    plt.ylabel("Loss")
    plt.title("Average CANO Loss vs $\\mu_{0}$")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(out_dir, "loss_vs_mu2.png"))
    plt.close()


if __name__ == "__main__":
    out_dir = "results/20250523_002301/sweep mu2/"
    plot(out_dir)
