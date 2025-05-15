#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import glob

from utils.utils import *


def plot(out_dir):
    # read all per-p CSVs
    files = glob.glob(os.path.join(out_dir, "sweep_results_p*.csv"))
    records = []
    for fp in files:
        dfp = pd.read_csv(fp)
        records.append(dfp)

    # save raw results
    df = pd.concat(records, ignore_index=True)
    csv_path = os.path.join(out_dir, "sweep_results.csv")
    df.to_csv(csv_path, index=False)

    # compute per-p averages
    summary = df.groupby("p").mean().reset_index()

    # plot accuracy vs p
    plt.figure()
    plt.plot(summary["p"], summary["acc1"], marker="o", label="with R", color="b")
    plt.plot(summary["p"], summary["acc0"], marker="o", label="without R", color="r")
    plt.plot(
        summary["p"],
        summary["sc_acc"],
        marker="o",
        label="Spectral clustring",
        color="g",
    )
    plt.gca().invert_xaxis()  # optional: show largest p at left
    plt.xlabel("p")
    plt.ylabel("Accuracy")
    plt.title("Average CANO Accuracy vs p")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(out_dir, "accuracy_vs_p.png"))
    plt.close()

    # plot loss vs p
    plt.figure()
    plt.plot(summary["p"], summary["loss1"], marker="o", color="b", label="with R")
    plt.plot(summary["p"], summary["loss0"], marker="o", color="r", label="without R")
    # plt.gca().invert_xaxis()
    plt.xlabel("p")
    plt.ylabel("Loss")
    plt.title("Average CANO Loss vs p")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(out_dir, "loss_vs_p.png"))
    plt.close()


if __name__ == "__main__":
    out_dir = "results/20250427_105847/sweep p/"
    plot(out_dir)
