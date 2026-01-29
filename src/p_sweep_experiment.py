"""
Experiment script to sweep SBM parameter p and test CADO under different modes.

Modes:
1. Graph + Label (mu1=0, mu2=mu2_default)
2. Graph + Feature (mu1=mu1_default, mu2=0)
3. Graph + Feature + Label (mu1=mu1_default, mu2=mu2_default)
4. Graph Only (mu1=0, mu2=0) - Subspace clustering

For each p value:
- Generate the same graph once
- Run 4 modes with different mu parameters
- Repeat 10 times and compute average
- Report Accuracy for cases without labels
"""

from copy import deepcopy
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import os
import sys
from multiprocessing import Pool, cpu_count
from functools import partial

# Ensure we're in the project root directory for correct imports and .env loading
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from CADO import CADO
from utils.algorithm_utils import (
    create_random_atoms,
    run_spectral_clustering,
)
from utils.SBM import SBM
from utils.utils import *


def run_single_experiment(graph, mode_name, mu1, mu2, n_runs=10, atoms=None):
    """
    Run CADO multiple times for a specific mode and compute average ACC.

    Args:
        graph: SBM graph object
        mode_name: Name of the mode
        mu1: Parameter mu1 value
        mu2: Parameter mu2 value
        n_runs: Number of runs to average over
        atoms: Initial atoms to use for all runs (optional)
    Returns:
        avg_acc: Average ACC across runs
        std_acc: Standard deviation of ACC across runs
    """
    accuracies = []
    n = graph.num_nodes
    num_features = graph.num_features
    c = graph.num_clusters

    for _ in range(n_runs):
        # Create random initial atoms for each run
        if atoms is None:
            atoms = create_random_atoms(n, num_features, c, c)
        else:
            atoms = deepcopy(atoms)

        # if mode_name == "Graph Only":
        #     cluster_labels = run_spectral_clustering(graph, c)
        #     ari = clustering_accuracy(graph.y, cluster_labels)
        #     accuracies.append(ari)
        # else:
        # Run CADO
        (
            final_atoms,
            final_graph,
            results,
            f_values,
            accuracies,
            epsilsons,
            components,
            opt_params,
            R_params,
        ) = CADO(
            graph,
            atoms,
            log_results=False,
            leave=False,
            mu1=mu1,
            mu2=mu2,
        )

        accuracies.append(accuracies[-1])

    avg_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)

    return avg_acc, std_acc


def process_single_p_value(p, modes, n_runs_per_mode, config_data):
    """
    Process a single p value with all modes. This function is designed to be run in parallel.

    Args:
        p: The p value to test
        modes: List of mode configurations
        n_runs_per_mode: Number of runs per mode
        config_data: Dictionary with configuration parameters

    Returns:
        List of result dictionaries for this p value
    """
    # Unpack configuration
    samples_per_cluster = config_data["samples_per_cluster"]
    num_features = config_data["num_features"]
    train_ratio = config_data["train_ratio"]
    gmm_center_min = config_data["gmm_center_min"]
    gmm_min_separation = config_data["gmm_min_separation"]
    gmm_sigma = config_data["gmm_sigma"]

    c = len(samples_per_cluster)
    n = sum(samples_per_cluster)
    q = 1 - p

    # Create SBM graph for this p value
    probs = np.full((c, c), q)
    np.fill_diagonal(probs, p)

    graph = SBM(
        samples_per_cluster,
        probs,
        num_features,
        train_ratio=train_ratio,
        gmm_center_min=gmm_center_min,
        gmm_min_separation=gmm_min_separation,
        gmm_sigma=gmm_sigma,
    )

    # Create initial atoms for this p value
    atoms = create_random_atoms(n, num_features, c, c)

    # Results for this p value
    p_results = []

    # Run each mode
    for mode in modes:
        mode_name = mode["name"]
        mu1 = mode["mu1"]
        mu2 = mode["mu2"]

        # Run experiment n_runs_per_mode times
        avg_acc, std_acc = run_single_experiment(
            graph, mode_name, mu1, mu2, n_runs=n_runs_per_mode, atoms=atoms
        )

        # Store results
        p_results.append(
            {
                "p": p,
                "q": q,
                "mode": mode_name,
                "mu1": mu1,
                "mu2": mu2,
                "avg_acc": avg_acc,
                "std_acc": std_acc,
                "n_runs": n_runs_per_mode,
            }
        )

    return p_results


def main(n_workers=None):
    """
    Main experiment function with parallel processing.

    Args:
        n_workers: Number of parallel workers. If None, uses cpu_count() - 1
    """

    # Parameters
    p_values = np.arange(0.5, 0.99, 0.05)  # 0 to 0.2 with step 0.01
    n_runs_per_mode = 20  # Number of runs per mode

    # Get default mu values from config
    mu1_default = data.mu1
    mu2_default = data.mu2

    # Define modes
    modes = [
        {"name": "Graph+Label", "mu1": 0.0, "mu2": mu2_default},
        {"name": "Graph+Feature", "mu1": mu1_default, "mu2": 0.0},
        {"name": "Graph+Feature+Label", "mu1": mu1_default, "mu2": mu2_default},
        {"name": "Graph Only", "mu1": 0.0, "mu2": 0.0},
    ]

    # Set number of workers
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)  # Leave one CPU free

    print(f"Using {n_workers} parallel workers (out of {cpu_count()} available CPUs)")

    # Results storage
    results_list = []

    # Create progress bar for p values
    print(f"Running experiment with {len(p_values)} p values and {len(modes)} modes")
    print(f"Each configuration will be run {n_runs_per_mode} times")
    print(f"Total runs: {len(p_values) * len(modes) * n_runs_per_mode}")

    # Prepare configuration data
    config_data = {
        "samples_per_cluster": data.samples_per_cluster,
        "num_features": data.num_features,
        "train_ratio": data.train_ratio,
        "gmm_center_min": data.gmm_center_min,
        "gmm_min_separation": data.gmm_min_separation,
        "gmm_sigma": data.gmm_sigma,
    }

    # Create partial function with fixed parameters
    process_func = partial(
        process_single_p_value,
        modes=modes,
        n_runs_per_mode=n_runs_per_mode,
        config_data=config_data,
    )

    # Run parallel processing
    print(f"\nStarting parallel processing...")
    with Pool(processes=n_workers) as pool:
        # Use imap for progress tracking
        for p_results in tqdm(
            pool.imap(process_func, p_values),
            total=len(p_values),
            desc="Processing p values",
        ):
            # Extend results with all modes from this p value
            results_list.extend(p_results)

            # Show last result
            if p_results:
                last_result = p_results[-1]
                tqdm.write(
                    f"Completed p={last_result['p']:.3f} | "
                    f"Last mode: {last_result['mode']} | "
                    f"Acc={last_result['avg_acc']:.4f}±{last_result['std_acc']:.4f}"
                )

    # Convert results to DataFrame
    df_results = pd.DataFrame(results_list)

    # Round numerical columns to 4 decimal places
    numerical_cols = ["p", "q", "mu1", "mu2", "avg_acc", "std_acc"]
    for col in numerical_cols:
        if col in df_results.columns:
            df_results[col] = df_results[col].round(4)

    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # results_dir = f"results/p_sweep_{timestamp}"
    os.makedirs(save_path, exist_ok=True)

    # Pivot to wide format: each row = p value, columns = modes
    df_wide = df_results.pivot(index="p", columns="mode", values="avg_acc")
    df_wide = df_wide.reset_index()
    df_wide.columns.name = None

    # Reorder columns for clarity
    column_order = [
        "p",
        "Graph Only",
        "Graph+Label",
        "Graph+Feature",
        "Graph+Feature+Label",
    ]
    existing_cols = [col for col in column_order if col in df_wide.columns]
    df_wide = df_wide[existing_cols]

    # Save wide format CSV with 4 decimal precision
    csv_path = f"{save_path}/p_sweep_results.csv"
    df_wide.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\nResults saved to: {csv_path}")

    # Also save detailed results (long format) for reference
    csv_path_detailed = f"{save_path}/p_sweep_results_detailed.csv"
    df_results.to_csv(csv_path_detailed, index=False, float_format="%.4f")
    print(f"Detailed results saved to: {csv_path_detailed}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(
        f"  - p values: {p_values[0]:.2f} to {p_values[-1]:.2f} (step: {p_values[1]-p_values[0]:.2f})"
    )
    print(f"  - Samples per cluster: {config_data['samples_per_cluster']}")
    print(f"  - Number of features: {config_data['num_features']}")
    print(f"  - Runs per configuration: {n_runs_per_mode}")
    print(f"  - Parallel workers: {n_workers}")
    print(f"\nModes tested:")
    for mode in modes:
        print(f"  - {mode['name']}: mu1={mode['mu1']}, mu2={mode['mu2']}")

    print(f"\n{'Mode':<25} {'Avg Accuracy':<15} {'Std Accuracy':<15}")
    print("-" * 55)
    for mode in modes:
        mode_data = df_results[df_results["mode"] == mode["name"]]
        avg_across_p = mode_data["avg_acc"].mean()
        std_across_p = mode_data["avg_acc"].std()
        print(f"{mode['name']:<25} {avg_across_p:<15.4f} {std_across_p:<15.4f}")

    print("\n" + "=" * 80)

    # Save summary to text file
    summary_path = f"{save_path}/summary.txt"
    with open(summary_path, "w") as f:
        f.write("EXPERIMENT SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(
            f"  - p values: {p_values[0]:.2f} to {p_values[-1]:.2f} (step: {p_values[1]-p_values[0]:.2f})\n"
        )
        f.write(f"  - Samples per cluster: {config_data['samples_per_cluster']}\n")
        f.write(f"  - Number of features: {config_data['num_features']}\n")
        f.write(f"  - Runs per configuration: {n_runs_per_mode}\n")
        f.write(f"  - Parallel workers: {n_workers}\n")
        f.write(f"\nModes tested:\n")
        for mode in modes:
            f.write(f"  - {mode['name']}: mu1={mode['mu1']}, mu2={mode['mu2']}\n")
        f.write(f"\nResults by mode (averaged across all p values):\n")
        f.write(f"{'Mode':<25} {'Avg ACC':<15} {'Std ACC':<15}\n")
        f.write("-" * 55 + "\n")
        for mode in modes:
            mode_data = df_results[df_results["mode"] == mode["name"]]
            avg_across_p = mode_data["avg_acc"].mean()
            std_across_p = mode_data["avg_acc"].std()
            f.write(f"{mode['name']:<25} {avg_across_p:<15.4f} {std_across_p:<15.4f}\n")

    print(f"Summary saved to: {summary_path}")

    return df_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run p-sweep experiment with parallel processing"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: cpu_count - 1)",
    )
    args = parser.parse_args()

    results_df = main(n_workers=args.workers)
