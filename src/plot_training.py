#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import glob

from utils.utils import *


def plot(out_dir):
    # read all per-p CSVs
    files = glob.glob(os.path.join(out_dir, "sweep_results_training_ratio*.csv"))
    records = []
    for fp in files:
        dfp = pd.read_csv(fp)
        records.append(dfp)

    # save raw results
    df = pd.concat(records, ignore_index=True)
    csv_path = os.path.join(out_dir, "sweep_results.csv")
    df.to_csv(csv_path, index=False)

    # compute per-p averages
    summary = df.groupby("training_ratio").mean().reset_index()
    summary.to_csv(
        os.path.join(out_dir, "sweep_results_avg.csv"), index=False
    )  # save per-p averages

    # plot accuracy vs p
    plt.figure()
    plt.plot(
        summary["training_ratio"], summary["acc0"], marker="o", label="f+g+l", color="b"
    )
    plt.plot(
        summary["training_ratio"], summary["acc1"], marker="o", label="f+l", color="r"
    )
    plt.plot(
        summary["training_ratio"], summary["acc2"], marker="o", label="g+l", color="g"
    )
    # plt.plot(
    #     summary["training_ratio"], summary["acc3"], marker="o", label="l", color="y"
    # )
    # plt.gca().invert_xaxis()  # optional: show largest p at left
    plt.xlabel("training_ratio")
    plt.ylabel("Accuracy")
    plt.title("Average CANO Accuracy vs training_ratio")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(out_dir, "accuracy_vs_training_ratio.png"))
    plt.close()

    # plot loss vs p
    plt.figure()
    plt.plot(
        summary["training_ratio"],
        summary["loss1"],
        marker="o",
        color="b",
        label="with L",
    )
    plt.plot(
        summary["training_ratio"],
        summary["loss0"],
        marker="o",
        color="r",
        label="without L",
    )
    # plt.gca().invert_xaxis()
    plt.xlabel("training_ratio")
    plt.ylabel("Loss")
    plt.title("Average CANO Loss vs training_ratio")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(out_dir, "loss_vs_training_ratio.png"))
    plt.close()


if __name__ == "__main__":
    out_dir = "results/20250516_105742/sweep training_ratio/"
    plot(out_dir)
