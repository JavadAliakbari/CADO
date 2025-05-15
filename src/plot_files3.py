#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import glob

from utils.utils import *


def plot(out_dir):
    # read all per-p CSVs
    files = glob.glob(os.path.join(out_dir, "sweep_results_rho_min*.csv"))
    records = []
    for fp in files:
        dfp = pd.read_csv(fp)
        records.append(dfp)

    # save raw results
    df = pd.concat(records, ignore_index=True)
    csv_path = os.path.join(out_dir, "sweep_results.csv")
    df.to_csv(csv_path, index=False)

    # compute per-p averages
    summary = df.groupby("rho_min").mean().reset_index()

    # plot accuracy vs p
    plt.figure()
    plt.plot(summary["rho_min"], summary["acc1"], marker="o", label="with L", color="b")
    plt.plot(
        summary["rho_min"], summary["acc0"], marker="o", label="without L", color="r"
    )
    # plt.gca().invert_xaxis()  # optional: show largest p at left
    plt.xlabel("$\\rho_{min}$")
    plt.ylabel("Accuracy")
    plt.title("Average CANO Accuracy vs $\\rho_{min}$")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(out_dir, "accuracy_vs_rho_min.png"))
    plt.close()

    # plot loss vs p
    plt.figure()
    plt.plot(
        summary["rho_min"], summary["loss1"], marker="o", color="b", label="with L"
    )
    plt.plot(
        summary["rho_min"], summary["loss0"], marker="o", color="r", label="without L"
    )
    # plt.gca().invert_xaxis()
    plt.xlabel("$\\rho_{min}$")
    plt.ylabel("Loss")
    plt.title("Average CANO Loss vs $\\rho_{min}$")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(out_dir, "loss_vs_rho_min.png"))
    plt.close()


if __name__ == "__main__":
    out_dir = "results/20250428_121519/sweep rho_min/"
    plot(out_dir)
