#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import glob

from utils.utils import *


def plot(out_dir):
    # read all per-p CSVs
    files = glob.glob(os.path.join(out_dir, "sweep_results_c*.csv"))
    records = []
    for fp in files:
        dfp = pd.read_csv(fp)
        records.append(dfp)

    # save raw results
    df = pd.concat(records, ignore_index=True)
    csv_path = os.path.join(out_dir, "sweep_results.csv")
    df.to_csv(csv_path, index=False)

    # compute per-p averages
    summary = df.groupby("c").mean().reset_index()
    summary.to_csv(os.path.join(out_dir, "sweep_results_avg.csv"), index=False)

    # plot accuracy vs p
    plt.figure()
    plt.plot(summary["c"], summary["acc0"], marker="o", label="f+g+l", color="b")
    plt.plot(summary["c"], summary["acc1"], marker="o", label="f+l", color="r")
    plt.plot(summary["c"], summary["acc2"], marker="o", label="f+g", color="g")
    plt.plot(summary["c"], summary["acc3"], marker="o", label="f", color="y")
    # plt.gca().invert_xaxis()  # optional: show largest p at left
    plt.xlabel("c")
    plt.ylabel("Accuracy")
    plt.title("Average CANO Accuracy vs c")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(out_dir, "accuracy_vs_n.png"))
    plt.close()

    # plot loss vs p
    plt.figure()
    plt.plot(summary["c"], summary["loss1"], marker="o", color="b", label="with L")
    plt.plot(summary["c"], summary["loss0"], marker="o", color="r", label="without L")
    # plt.gca().invert_xaxis()
    plt.xlabel("c")
    plt.ylabel("Loss")
    plt.title("Average CANO Loss vs c")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(out_dir, "loss_vs_c.png"))
    plt.close()


if __name__ == "__main__":
    out_dir = "results/20250523_115203/sweep c/"
    plot(out_dir)
