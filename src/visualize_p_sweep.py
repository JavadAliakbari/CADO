"""
Visualization script for p-sweep experiment results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import numpy as np


def plot_results(csv_path, output_dir=None):
    """
    Create visualizations for p-sweep experiment results.

    Args:
        csv_path: Path to the CSV file with results
        output_dir: Directory to save plots (if None, uses the same directory as csv)
    """
    # Load results
    df = pd.read_csv(csv_path)

    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.rcParams["font.size"] = 12

    # 1. Main plot: ARI vs p for all modes
    fig, ax = plt.subplots(figsize=(14, 8))

    modes = df["mode"].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(modes)))

    for mode, color in zip(modes, colors):
        mode_data = df[df["mode"] == mode]

        # Plot mean line
        ax.plot(
            mode_data["p"],
            mode_data["avg_ari"],
            marker="o",
            label=mode,
            linewidth=2.5,
            markersize=6,
            color=color,
            alpha=0.9,
        )

        # Add error bars (std)
        ax.fill_between(
            mode_data["p"],
            mode_data["avg_ari"] - mode_data["std_ari"],
            mode_data["avg_ari"] + mode_data["std_ari"],
            alpha=0.2,
            color=color,
        )

    ax.set_xlabel(
        "p (within-cluster connection probability)", fontsize=14, fontweight="bold"
    )
    ax.set_ylabel("ARI (Adjusted Rand Index)", fontsize=14, fontweight="bold")
    ax.set_title(
        "CADO Performance: ARI vs p for Different Modes",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.legend(loc="best", fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(f"{output_dir}/ari_vs_p_all_modes.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_dir}/ari_vs_p_all_modes.pdf", bbox_inches="tight")
    print(f"Saved: {output_dir}/ari_vs_p_all_modes.png")

    # 2. Heatmap of ARI values
    fig, ax = plt.subplots(figsize=(16, 6))

    # Pivot data for heatmap
    pivot_data = df.pivot(index="mode", columns="p", values="avg_ari")

    sns.heatmap(
        pivot_data,
        annot=False,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "ARI"},
        ax=ax,
    )

    ax.set_xlabel(
        "p (within-cluster connection probability)", fontsize=14, fontweight="bold"
    )
    ax.set_ylabel("Mode", fontsize=14, fontweight="bold")
    ax.set_title(
        "ARI Heatmap Across Modes and p Values", fontsize=16, fontweight="bold", pad=20
    )

    plt.tight_layout()
    plt.savefig(f"{output_dir}/ari_heatmap.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_dir}/ari_heatmap.pdf", bbox_inches="tight")
    print(f"Saved: {output_dir}/ari_heatmap.png")

    # 3. Box plots for each mode
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, mode in enumerate(modes):
        mode_data = df[df["mode"] == mode]

        axes[idx].errorbar(
            mode_data["p"],
            mode_data["avg_ari"],
            yerr=mode_data["std_ari"],
            marker="o",
            capsize=5,
            capthick=2,
            linewidth=2,
            markersize=8,
            alpha=0.8,
        )

        axes[idx].set_xlabel("p", fontsize=12, fontweight="bold")
        axes[idx].set_ylabel("ARI", fontsize=12, fontweight="bold")
        axes[idx].set_title(f"{mode}", fontsize=13, fontweight="bold")
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_ylim([0, 1.05])

    plt.suptitle(
        "ARI vs p for Each Mode (with Standard Deviation)",
        fontsize=16,
        fontweight="bold",
        y=1.00,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ari_vs_p_individual.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_dir}/ari_vs_p_individual.pdf", bbox_inches="tight")
    print(f"Saved: {output_dir}/ari_vs_p_individual.png")

    # 4. Bar plot comparing modes at specific p values
    # Select a few representative p values
    p_values_to_compare = [0.0, 0.05, 0.10, 0.15, 0.20]
    df_compare = df[df["p"].isin(p_values_to_compare)]

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(p_values_to_compare))
    width = 0.2

    for idx, mode in enumerate(modes):
        mode_data = df_compare[df_compare["mode"] == mode]
        offsets = x + (idx - len(modes) / 2 + 0.5) * width

        bars = ax.bar(
            offsets,
            mode_data["avg_ari"],
            width,
            label=mode,
            alpha=0.8,
            yerr=mode_data["std_ari"],
            capsize=5,
        )

    ax.set_xlabel("p", fontsize=14, fontweight="bold")
    ax.set_ylabel("ARI", fontsize=14, fontweight="bold")
    ax.set_title(
        "Mode Comparison at Selected p Values", fontsize=16, fontweight="bold", pad=20
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:.2f}" for p in p_values_to_compare])
    ax.legend(loc="best", fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0, 1.1])

    plt.tight_layout()
    plt.savefig(f"{output_dir}/mode_comparison_bars.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_dir}/mode_comparison_bars.pdf", bbox_inches="tight")
    print(f"Saved: {output_dir}/mode_comparison_bars.png")

    # 5. Standard deviation plot
    fig, ax = plt.subplots(figsize=(14, 8))

    for mode, color in zip(modes, colors):
        mode_data = df[df["mode"] == mode]
        ax.plot(
            mode_data["p"],
            mode_data["std_ari"],
            marker="s",
            label=mode,
            linewidth=2.5,
            markersize=6,
            color=color,
            alpha=0.9,
        )

    ax.set_xlabel(
        "p (within-cluster connection probability)", fontsize=14, fontweight="bold"
    )
    ax.set_ylabel("Standard Deviation of ARI", fontsize=14, fontweight="bold")
    ax.set_title(
        "Variability in CADO Performance Across Runs",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.legend(loc="best", fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/std_vs_p.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_dir}/std_vs_p.pdf", bbox_inches="tight")
    print(f"Saved: {output_dir}/std_vs_p.png")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    for mode in modes:
        mode_data = df[df["mode"] == mode]
        print(f"\n{mode}:")
        print(
            f"  Mean ARI: {mode_data['avg_ari'].mean():.4f} ± {mode_data['avg_ari'].std():.4f}"
        )
        print(
            f"  Min ARI:  {mode_data['avg_ari'].min():.4f} (at p={mode_data.loc[mode_data['avg_ari'].idxmin(), 'p']:.3f})"
        )
        print(
            f"  Max ARI:  {mode_data['avg_ari'].max():.4f} (at p={mode_data.loc[mode_data['avg_ari'].idxmax(), 'p']:.3f})"
        )
        print(f"  Avg Std:  {mode_data['std_ari'].mean():.4f}")

    print("\n" + "=" * 80)


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_p_sweep.py <path_to_csv>")
        print("\nOr provide the results directory:")
        print("python visualize_p_sweep.py results/p_sweep_YYYYMMDD_HHMMSS/")
        sys.exit(1)

    input_path = sys.argv[1]

    # Check if input is a directory or a file
    if os.path.isdir(input_path):
        # Look for CSV file in directory
        csv_files = [f for f in os.listdir(input_path) if f.endswith(".csv")]
        if not csv_files:
            print(f"No CSV files found in {input_path}")
            sys.exit(1)
        csv_path = os.path.join(input_path, csv_files[0])
        output_dir = input_path
    else:
        # Input is a file
        csv_path = input_path
        output_dir = os.path.dirname(csv_path)

    print(f"Loading results from: {csv_path}")
    print(f"Saving plots to: {output_dir}")

    plot_results(csv_path, output_dir)

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
