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



def main():
    # output dir
    out_dir = "results/20250427_045052/with R/"
    os.makedirs(out_dir, exist_ok=True)

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

    out_dir = "results/20250427_045052/without R/"
    os.makedirs(out_dir, exist_ok=True)

    # read all per-p CSVs
    files = glob.glob(os.path.join(out_dir, "sweep_results_p*.csv"))
    records = []
    for fp in files:
        dfp = pd.read_csv(fp)
        records.append(dfp)

    # save raw results
    df2 = pd.concat(records, ignore_index=True)
    csv_path = os.path.join(out_dir, "sweep_results2.csv")
    df2.to_csv(csv_path, index=False)

    # compute per-p averages
    summary2 = df2.groupby("p").mean().reset_index()

    # plot accuracy vs p
    plt.figure()
    plt.plot(summary["p"], summary["accuracy"], marker="o", label="with R", color="b")
    plt.plot(summary2["p"], summary2["accuracy"], marker="o", label="without R", color="r")
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
    plt.plot(summary["p"], summary["loss"], marker="o", color="b", label="with R")
    plt.plot(summary2["p"], summary2["loss"], marker="o", color="r", label="without R")
    # plt.gca().invert_xaxis()
    plt.xlabel("p")
    plt.ylabel("Loss")
    plt.title("Average CANO Loss vs p")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(out_dir, "loss_vs_p.png"))
    plt.close()


if __name__ == "__main__":
    main()
