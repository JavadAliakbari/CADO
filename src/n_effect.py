#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import traceback
import sys
import argparse
import warnings
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

# Suppress specific sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from idea5 import CANO
from utils.SBM import SBM
from utils.utils import clustring_accuracy, data, save_path, save_config
from utils.algorithm_utils import create_random_atoms, run_spectral_clustering


# Helper function to run a single experiment with specific node count and p value
def run_single(run, p, node_scale):
    # Scale the node counts based on the node_scale parameter
    # We maintain the same proportions but scale the total count
    original_samples = data.samples_per_cluster
    scaled_samples = [int(n * node_scale) for n in original_samples]

    # Build SBM with scaled node counts
    c = len(scaled_samples)
    probs = np.full((c, c), data.q)
    np.fill_diagonal(probs, p)

    try:
        graph = SBM(
            scaled_samples,
            probs,
            data.num_features,
            data.num_signal_features,
            train_ratio=data.train_ratio,
        )

        # Random warm start
        orig_atoms, orig_coefs = create_random_atoms(
            sum(scaled_samples), data.num_features, c, c
        )

        # Run with different CANO configurations
        # Configuration 0: mu1=data.mu1, mu2=data.mu2 (default)
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

        # Configuration 1: mu1=0, mu2=data.mu2
        atoms, coefs = deepcopy(orig_atoms), deepcopy(orig_coefs)
        _, _, _, _, f_vals1, accs1, *_ = CANO(
            graph,
            atoms,
            coefs,
            mu1=0,
            lr_search=False,
            log_results=False,
            leave=False,
        )
        loss_final1 = f_vals1[-1]
        acc_final1 = accs1[-1]

        # Configuration 2: mu1=data.mu1, mu2=0
        atoms, coefs = deepcopy(orig_atoms), deepcopy(orig_coefs)
        _, _, _, _, f_vals2, accs2, *_ = CANO(
            graph,
            atoms,
            coefs,
            mu2=0,
            lr_search=False,
            log_results=False,
            leave=False,
        )
        loss_final2 = f_vals2[-1]
        acc_final2 = accs2[-1]

        # Configuration 3: mu1=0, mu2=0
        atoms, coefs = deepcopy(orig_atoms), deepcopy(orig_coefs)
        _, _, _, _, f_vals3, accs3, *_ = CANO(
            graph,
            atoms,
            coefs,
            mu1=0,
            mu2=0,
            lr_search=False,
            log_results=False,
            leave=False,
        )
        loss_final3 = f_vals3[-1]
        acc_final3 = accs3[-1]

        # Run spectral clustering for comparison
        cluster_labels = run_spectral_clustering(graph, c)
        sc_acc = clustring_accuracy(cluster_labels, graph.y)

        # Return results including the node scale
        return {
            "node_scale": node_scale,
            "total_nodes": sum(scaled_samples),
            "p": np.round(p, 3),
            "run": run,
            "loss0": np.round(loss_final0, 4),
            "acc0": np.round(acc_final0, 4),
            "loss1": np.round(loss_final1, 4),
            "acc1": np.round(acc_final1, 4),
            "loss2": np.round(loss_final2, 4),
            "acc2": np.round(acc_final2, 4),
            "loss3": np.round(loss_final3, 4),
            "acc3": np.round(acc_final3, 4),
            "sc_acc": np.round(sc_acc, 4),
        }
    except Exception as e:
        print(
            f"Error in run_single (node_scale={node_scale}, p={p:.3f}, run={run}): {str(e)}"
        )
        traceback.print_exc()
        # Return None to indicate failure
        return None


def process_combination(p, node_scale, out_dir="./", debug_mode=False):
    # Override data.p for this p value
    data.p = float(p)

    results = []

    # For debugging: run a single process directly
    if debug_mode:
        print(f"Running in debug mode for p={p:.3f}, node_scale={node_scale}")
        try:
            result = run_single(0, p, node_scale)
            if result:
                results.append(result)
                print("Debug run successful")
            else:
                print("Debug run failed")
                return None
        except Exception as e:
            print(f"Error in debug run: {str(e)}")
            traceback.print_exc()
            return None
    else:
        # Execute runs in parallel
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Adjust number of runs based on p value (more runs for difficult cases)
            total_runs = 100 if p <= 0.1 else 50

            futures = [
                executor.submit(run_single, run, p, node_scale)
                for run in range(total_runs)
            ]

            # Using tqdm to track completion
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc=f"p={p:.3f}, scale={node_scale} runs",
                leave=False,
            ):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"\nError in worker process: {str(e)}")
                    print("\nTraceback:")
                    traceback.print_exc()
                    print("\nContinuing with other tasks...")

    if not results:
        print(f"No successful results for p={p:.3f}, node_scale={node_scale}")
        return None

    # Save results to CSV
    result_file = os.path.join(out_dir, f"results_p{p:.3f}_n{node_scale:.2f}.csv")
    df = pd.DataFrame(results)
    df.to_csv(result_file, index=False)

    # Calculate mean metrics
    mean_metrics = {
        "acc0": df["acc0"].mean(),
        "acc1": df["acc1"].mean(),
        "acc2": df["acc2"].mean(),
        "acc3": df["acc3"].mean(),
        "sc_acc": df["sc_acc"].mean(),
    }

    return mean_metrics


def plot_n_effect(out_dir):
    """Enhanced plotting function that visualizes the effect of both p and node count."""
    # Find all result files
    import glob

    files = glob.glob(os.path.join(out_dir, "results_p*_n*.csv"))

    if not files:
        print(f"No result files found in {out_dir}")
        return

    records = []
    for fp in files:
        df = pd.read_csv(fp)
        records.append(df)

    # Combine all results
    all_results = pd.concat(records, ignore_index=True)

    # Save combined results
    all_results.to_csv(os.path.join(out_dir, "all_results.csv"), index=False)

    # Compute averages grouped by p and node_scale
    summary = all_results.groupby(["p", "node_scale"]).mean().reset_index()
    summary.to_csv(os.path.join(out_dir, "summary_results.csv"), index=False)

    # Get unique p values and node scales
    p_values = sorted(summary["p"].unique())
    node_scales = sorted(summary["node_scale"].unique())

    # 1. Plot accuracy vs p for different node scales
    plt.figure(figsize=(12, 8))
    cmap = get_cmap("viridis")
    colors = [cmap(i) for i in np.linspace(0, 1, len(node_scales))]

    for i, scale in enumerate(node_scales):
        scale_data = summary[summary["node_scale"] == scale]
        plt.plot(
            scale_data["p"],
            scale_data["acc1"],
            marker="o",
            label=f"n={scale:.2f}x ({int(scale_data['total_nodes'].mean())} nodes)",
            color=colors[i],
        )

    plt.gca().invert_xaxis()  # Show largest p at left
    plt.xlabel("In-cluster probability (p)")
    plt.ylabel("Accuracy")
    plt.title("CANO Accuracy vs p for Different Node Counts")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_vs_p_by_node_count.png"))
    plt.close()

    # 2. Plot accuracy vs node scale for different p values
    plt.figure(figsize=(12, 8))
    cmap = get_cmap("plasma")
    colors = [cmap(i) for i in np.linspace(0, 1, len(p_values))]

    for i, p in enumerate(p_values):
        p_data = summary[summary["p"] == p]
        # Sort by node_scale to ensure correct line plotting
        p_data = p_data.sort_values("node_scale")
        plt.plot(
            p_data["total_nodes"],
            p_data["acc1"],
            marker="o",
            label=f"p={p:.3f}",
            color=colors[i],
        )

    plt.xlabel("Total Number of Nodes")
    plt.ylabel("Accuracy")
    plt.title("CANO Accuracy vs Node Count for Different p Values")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_vs_node_count_by_p.png"))
    plt.close()

    # 3. Heatmap visualization showing accuracy as function of both p and node count
    plt.figure(figsize=(10, 8))

    # Create pivot table for heatmap
    pivot_table = summary.pivot(index="p", columns="node_scale", values="acc1")

    # Create heatmap
    plt.imshow(pivot_table, cmap="viridis", aspect="auto", interpolation="nearest")
    plt.colorbar(label="Accuracy")

    # Set the ticks and labels
    plt.xticks(range(len(node_scales)), [f"{scale:.2f}x" for scale in node_scales])
    plt.yticks(range(len(p_values)), [f"{p:.3f}" for p in p_values])

    plt.xlabel("Node Scale Factor")
    plt.ylabel("In-cluster probability (p)")
    plt.title("Accuracy Heatmap: p vs Node Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_heatmap.png"))
    plt.close()

    # 4. Compare with spectral clustering
    plt.figure(figsize=(12, 8))

    # Create subplots for different node scales
    fig, axes = plt.subplots(1, len(node_scales), figsize=(16, 6), sharey=True)
    if len(node_scales) == 1:
        axes = [axes]  # Convert to list for consistent indexing

    for i, scale in enumerate(node_scales):
        scale_data = summary[summary["node_scale"] == scale]
        scale_data = scale_data.sort_values("p")

        ax = axes[i]
        ax.plot(
            scale_data["p"], scale_data["acc1"], marker="o", label="CANO", color="blue"
        )
        ax.plot(
            scale_data["p"],
            scale_data["sc_acc"],
            marker="o",
            label="Spectral",
            color="green",
        )

        ax.set_title(f"Scale: {scale:.2f}x")
        ax.set_xlabel("p")
        ax.grid(True)

        if i == 0:
            ax.set_ylabel("Accuracy")
            ax.legend()

    plt.suptitle("CANO vs Spectral Clustering Across Node Scales")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cano_vs_spectral.png"))
    plt.close()


def main():
    # Define ranges for p values and node scales
    p_values = np.arange(0.05, 0.31, 0.05)  # From 0.05 to 0.3 in steps of 0.05
    node_scales = [0.5, 1.0, 1.5, 2.0]  # Scale factors for node counts

    # Create output directory
    path = "node_effect/"
    out_dir = os.path.join(save_path, path)
    os.makedirs(out_dir, exist_ok=True)

    # Save configuration
    save_config(out_dir, "config.yml")

    # Store experiment metadata
    metadata = {
        "p_values": p_values.tolist(),
        "node_scales": node_scales,
        "original_samples_per_cluster": data.samples_per_cluster,
    }

    with open(os.path.join(out_dir, "experiment_metadata.json"), "w") as f:
        import json

        json.dump(metadata, f, indent=2)

    # Process each p-value and node scale combination
    results = []

    # Use tqdm for progress tracking
    total_combinations = len(p_values) * len(node_scales)
    progress_bar = tqdm(total=total_combinations, desc="Processing combinations")

    for p in p_values:
        for scale in node_scales:
            try:
                print(f"\nProcessing p={p:.3f}, node_scale={scale}")
                metrics = process_combination(p, scale, out_dir=out_dir)

                if metrics:
                    result = {"p": p, "node_scale": scale, **metrics}
                    results.append(result)

                    # Update progress bar with latest metrics
                    progress_bar.set_postfix(
                        {
                            "p": p,
                            "scale": scale,
                            "acc": metrics["acc1"],
                            "sc_acc": metrics["sc_acc"],
                        }
                    )
            except Exception as e:
                print(
                    f"\nError processing combination p={p:.3f}, scale={scale}: {str(e)}"
                )
                print("\nTraceback:")
                traceback.print_exc()
                print("\nContinuing with next combination...")
            finally:
                progress_bar.update(1)

    progress_bar.close()

    # Save summary of all combinations
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(os.path.join(out_dir, "combinations_summary.csv"), index=False)

    # Generate plots
    try:
        plot_n_effect(out_dir)
        print(f"\nResults saved to {out_dir}")
    except Exception as e:
        print(f"\nError in plotting: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run node effect experiments")
    parser.add_argument(
        "--debug", action="store_true", help="Run in debug mode with a single process"
    )
    parser.add_argument(
        "--p", type=float, help="Specific p value to debug (only used with --debug)"
    )
    parser.add_argument(
        "--scale",
        type=float,
        help="Specific node scale to debug (only used with --debug)",
    )
    args = parser.parse_args()

    if args.debug:
        print("Running in debug mode")
        # Create debug output directory
        path = "node_effect_debug/"
        out_dir = os.path.join(save_path, path)
        os.makedirs(out_dir, exist_ok=True)

        # Set default values if not provided
        p_value = args.p if args.p is not None else 0.15
        node_scale = args.scale if args.scale is not None else 1.0

        print(f"Using p={p_value}, node_scale={node_scale}")

        # Run a single combination in debug mode
        process_combination(p_value, node_scale, out_dir=out_dir, debug_mode=True)
    else:
        main()
