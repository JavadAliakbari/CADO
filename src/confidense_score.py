import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional, Any

from Graph_Atom.graph_func import GraphFunc
from utils.algorithm_utils import *
from utils.SBM import SBM
from utils.utils import *

# Set print options to show the entire array
np.set_printoptions(threshold=np.inf, suppress=False)


def get_timestamp():
    """
    Get current timestamp in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def calculate_f_statistics(graph: Graph, num_atoms_per_graph=100):
    """
    Calculate f statistics for a given graph by generating random atoms.
    Args:
        graph: An instance of a graph (e.g., SBM graph)
        num_atoms_per_graph: Number of random atoms to create per graph
    Returns:
        dict: A dictionary containing:
            - mean_f: Mean f value
            - median_f: Median f value
            - min_f: Minimum f value
            - max_f: Maximum f value
            - std_f: Standard deviation of f values
            - f_optimal: Optimal f value
            - f_gap: Difference between optimal and mean f values
            - pct_below_optimal: Percentage of random atoms with f values below optimal
    """
    # Calculate optimal value once for reference
    optimal_scores = get_confidence_scores(graph, calc_optimal=True)
    f_optimal = optimal_scores["f_optimal"]
    # Collect f values for this graph
    f_values = []

    # Generate random atoms and calculate f for each
    bar = tqdm(total=num_atoms_per_graph, desc="Calculating f values", leave=False)
    for _ in range(num_atoms_per_graph):
        scores = get_confidence_scores(graph, calc_optimal=False)
        f_values.append(scores["f_random"])

        bar.set_postfix(
            f_random=scores["f_random"],
            f_optimal=f_optimal,
        )
        bar.update(1)

    # Calculate percentage of random atoms with f values below optimal
    below_optimal_count = sum(f < f_optimal for f in f_values)
    pct_below_optimal = (below_optimal_count / len(f_values)) * 100

    return {
        "mean_f": np.mean(f_values),
        "median_f": np.median(f_values),
        "min_f": np.min(f_values),
        "max_f": np.max(f_values),
        "std_f": np.std(f_values),
        "f_optimal": f_optimal,
        "f_gap": f_optimal - np.mean(f_values),
        "pct_below_optimal": pct_below_optimal,
    }


def get_confidence_scores(graph: Graph, calc_optimal=False):
    """
    Calculate confidence scores for a given graph.

    Args:
        graph: An instance of a graph (e.g., SBM graph)
        calc_optimal: Whether to calculate optimal atoms and scores (default: False)

    Returns:
        dict: A dictionary containing:
            - f_random: Confidence score for random atoms
            - f_optimal: Confidence score for optimal atoms (if calc_optimal=True)
            - L_random: L component for random atoms
            - R_random: R component for random atoms
            - pi_random: pi component for random atoms
            - L_optimal: L component for optimal atoms (if calc_optimal=True)
            - R_optimal: R component for optimal atoms (if calc_optimal=True)
            - pi_optimal: pi component for optimal atoms (if calc_optimal=True)
    """
    n = graph.num_nodes
    c = graph.num_classes
    num_features = graph.num_features

    # Create random and optimal atoms
    atoms_random, coefs_random = create_random_atoms(n, num_features, c, c)

    # Map to simplex and calculate signature
    atoms_random, coefs_random = map_to_simplex(atoms_random, coefs_random)
    sig_random = atoms_random.calc_sig(coefs_random)

    # Initialize GraphFunc and calculate scores
    f = GraphFunc(graph=graph)

    # Get scores and components for random atoms
    f_random, L_random, R_random, pi_random = f(sig_random, components=True)

    if calc_optimal:
        atoms_optimal, coefs_optimal = create_optimal_atoms(graph)
        sig_optimal = atoms_optimal.calc_sig(coefs_optimal)
        f_optimal, L_optimal, R_optimal, pi_optimal = f(sig_optimal, components=True)
    else:
        # If optimal is not calculated, return only random values
        f_optimal = None
        L_optimal = None
        R_optimal = None
        pi_optimal = None

    return {
        "f_random": f_random,
        "f_optimal": f_optimal,
        "L_random": L_random,
        "R_random": R_random,
        "pi_random": pi_random,
        "L_optimal": L_optimal,
        "R_optimal": R_optimal,
        "pi_optimal": pi_optimal,
    }


def analyze_random_graphs_from_config(
    num_graphs=10, num_atoms_per_graph=100, save_results=True
):
    """
    Generate random graphs using config parameters and for each graph,
    create multiple random atoms and calculate mean f value.

    Args:
        num_graphs: Number of graphs to create
        num_atoms_per_graph: Number of random atoms to create per graph
        save_results: Whether to save results to CSV

    Returns:
        pandas.DataFrame: DataFrame containing results for each graph
    """
    # Data to collect
    results = []

    # Get current date for results folder
    timestamp = get_timestamp()
    out_dir = f"results/{timestamp}/random_graphs_analysis"
    if save_results and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Use parameters from config
    samples_per_cluster = data.samples_per_cluster
    num_features = data.num_features
    num_signal_features = data.num_signal_features
    p = data.p  # In-cluster probability
    q = data.q  # Between-cluster probability
    train_ratio = data.train_ratio

    # Iterate over number of graphs
    bar = tqdm(total=num_graphs, desc="Processing graphs")
    for graph_idx in range(num_graphs):
        # Create the graph with config parameters
        c = len(samples_per_cluster)
        probs = np.full((c, c), q)
        np.fill_diagonal(probs, p)
        graph = SBM(
            samples_per_cluster,
            probs,
            num_features,
            num_signal_features,
            train_ratio=train_ratio,
        )

        # Calculate statistics using our new function
        stats = calculate_f_statistics(graph, num_atoms_per_graph)

        # Store results
        graph_result = {
            "graph_idx": graph_idx,
            "n": graph.num_nodes,
            "c": graph.num_clusters,
            "p": p,
            "q": q,
            "num_features": num_features,
            "num_signal_features": num_signal_features,
            "mean_f": stats["mean_f"],
            "median_f": stats["median_f"],
            "min_f": stats["min_f"],
            "max_f": stats["max_f"],
            "std_f": stats["std_f"],
            "f_optimal": stats["f_optimal"],
            "f_gap": stats["f_gap"],
            "pct_below_optimal": stats["pct_below_optimal"],
        }

        results.append(graph_result)

        bar.set_postfix(
            graph_idx=graph_idx,
            mean_f=stats["mean_f"],
            f_optimal=stats["f_optimal"],
            f_gap=stats["f_gap"],
            pct_below_optimal=stats["pct_below_optimal"],
        )
        bar.update(1)

        # Optional: save the individual graph and its results
        # if save_results:
        #     graph_df = pd.DataFrame(f_values, columns=["f_value"])
        #     graph_df.to_csv(f"{out_dir}/graph_{graph_idx}_f_values.csv", index=False)

    # Create final DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    if save_results:
        results_df.to_csv(f"{out_dir}/graph_analysis_summary.csv", index=False)

        # Generate plots
        try:
            import matplotlib.pyplot as plt

            # Plot mean f vs optimal f for each graph
            plt.figure(figsize=(10, 6))
            plt.scatter(
                results_df["graph_idx"],
                results_df["mean_f"],
                label="Mean f",
                color="blue",
            )
            plt.scatter(
                results_df["graph_idx"],
                results_df["f_optimal"],
                label="Optimal f",
                color="green",
            )
            plt.fill_between(
                results_df["graph_idx"],
                results_df["min_f"],
                results_df["max_f"],
                alpha=0.2,
                color="blue",
            )
            plt.xlabel("Graph Index")
            plt.ylabel("f Value")
            plt.title("Mean f vs Optimal f for Each Random Graph")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{out_dir}/mean_vs_optimal_f.png")
            plt.close()

            # Plot the gap between optimal and mean
            plt.figure(figsize=(10, 6))
            plt.bar(results_df["graph_idx"], results_df["f_gap"])
            plt.xlabel("Graph Index")
            plt.ylabel("f Gap (Optimal - Mean)")
            plt.title("Gap between Optimal and Mean f Values")
            plt.grid(True)
            plt.savefig(f"{out_dir}/f_gap.png")
            plt.close()

            # Plot relationship between graph properties and f values
            fig, axs = plt.subplots(2, 2, figsize=(15, 10))

            axs[0, 0].scatter(results_df["p"], results_df["mean_f"])
            axs[0, 0].set_xlabel("p (in-cluster probability)")
            axs[0, 0].set_ylabel("Mean f")
            axs[0, 0].grid(True)

            axs[0, 1].scatter(results_df["q"], results_df["mean_f"])
            axs[0, 1].set_xlabel("q (between-cluster probability)")
            axs[0, 1].set_ylabel("Mean f")
            axs[0, 1].grid(True)

            axs[1, 0].scatter(results_df["p"] - results_df["q"], results_df["mean_f"])
            axs[1, 0].set_xlabel("p-q (cluster separation)")
            axs[1, 0].set_ylabel("Mean f")
            axs[1, 0].grid(True)

            axs[1, 1].scatter(results_df["c"], results_df["mean_f"])
            axs[1, 1].set_xlabel("Number of clusters")
            axs[1, 1].set_ylabel("Mean f")
            axs[1, 1].grid(True)

            plt.tight_layout()
            plt.savefig(f"{out_dir}/graph_properties_vs_f.png")
            plt.close()

        except Exception as e:
            print(f"Warning: Could not generate plots due to: {e}")

    return results_df


def sweep_p_parameter(
    p_values: List[float],
    num_atoms_per_graph: int = 100,
    save_results: bool = True,
    use_config_for_other_params: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Sweep over different values of p (in-cluster probability) and analyze how it affects confidence scores.

    Args:
        p_values: List of p values to sweep over
        num_atoms_per_graph: Number of random atoms to create per graph
        save_results: Whether to save results to CSV
        use_config_for_other_params: Whether to use config data for other parameters
        **kwargs: Additional parameters to override config values
            - q: Between-cluster probability
            - samples_per_cluster: List of number of samples per cluster
            - num_features: Number of features
            - num_signal_features: Number of signal features
            - train_ratio: Ratio of training samples

    Returns:
        pandas.DataFrame: DataFrame containing results for each p value
    """
    # Get current date for results folder
    timestamp = get_timestamp()
    out_dir = f"results/{timestamp}/p_sweep_analysis"
    if save_results and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Results to collect
    results = []

    # Use parameters from config or override with kwargs
    if use_config_for_other_params:
        samples_per_cluster = kwargs.get(
            "samples_per_cluster", data.samples_per_cluster
        )
        num_features = kwargs.get("num_features", data.num_features)
        num_signal_features = kwargs.get(
            "num_signal_features", data.num_signal_features
        )
        q = kwargs.get("q", data.q)  # Between-cluster probability
        train_ratio = kwargs.get("train_ratio", data.train_ratio)
    else:
        # Default values if not using config
        samples_per_cluster = kwargs.get("samples_per_cluster", [100, 100, 100])
        num_features = kwargs.get("num_features", 6)
        num_signal_features = kwargs.get("num_signal_features", 2)
        q = kwargs.get("q", 0.1)  # Between-cluster probability
        train_ratio = kwargs.get("train_ratio", 0.8)

    # Iterate over p values
    bar = tqdm(total=len(p_values), desc="Sweeping p values")
    for p_idx, p in enumerate(p_values):
        # Create the graph with current p value
        c = len(samples_per_cluster)
        probs = np.full((c, c), q)
        np.fill_diagonal(probs, p)

        graph = SBM(
            samples_per_cluster,
            probs,
            num_features,
            num_signal_features,
            train_ratio=train_ratio,
        )

        # Calculate statistics
        stats = calculate_f_statistics(graph, num_atoms_per_graph)

        # Store results
        result = {
            "p": p,
            "p_q_diff": p - q,
            "q": q,
            "num_clusters": c,
            "num_features": num_features,
            "num_signal_features": num_signal_features,
            "mean_f": stats["mean_f"],
            "median_f": stats["median_f"],
            "min_f": stats["min_f"],
            "max_f": stats["max_f"],
            "std_f": stats["std_f"],
            "f_optimal": stats["f_optimal"],
            "f_gap": stats["f_gap"],
            "pct_below_optimal": stats["pct_below_optimal"],
        }

        results.append(result)

        bar.set_postfix(
            p=p,
            mean_f=stats["mean_f"],
            f_optimal=stats["f_optimal"],
            f_gap=stats["f_gap"],
            pct_below_optimal=stats["pct_below_optimal"],
        )
        bar.update(1)

    # Create final DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    if save_results:
        results_df.to_csv(f"{out_dir}/p_sweep_analysis.csv", index=False)

        # Generate plots
        try:
            import matplotlib.pyplot as plt

            # Plot mean f vs optimal f
            plt.figure(figsize=(10, 6))
            plt.plot(results_df["p"], results_df["mean_f"], "o-", label="Mean f")
            plt.plot(results_df["p"], results_df["f_optimal"], "o-", label="Optimal f")
            plt.fill_between(
                results_df["p"],
                results_df["min_f"],
                results_df["max_f"],
                alpha=0.2,
                color="blue",
            )
            plt.xlabel("p (in-cluster probability)")
            plt.ylabel("f Value")
            plt.title("Effect of p on Confidence Scores")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{out_dir}/p_vs_f.png")
            plt.close()

            # Plot gap
            plt.figure(figsize=(10, 6))
            plt.plot(results_df["p"], results_df["f_gap"], "o-")
            plt.xlabel("p (in-cluster probability)")
            plt.ylabel("f Gap (Optimal - Mean)")
            plt.title("Effect of p on f Gap")
            plt.grid(True)
            plt.savefig(f"{out_dir}/p_vs_f_gap.png")
            plt.close()

            # Plot p vs p-q difference relation to mean_f
            plt.figure(figsize=(10, 6))
            plt.scatter(
                results_df["p"],
                results_df["mean_f"],
                s=100,
                c=results_df["p_q_diff"],
                cmap="viridis",
            )
            plt.colorbar(label="p-q difference")
            plt.xlabel("p (in-cluster probability)")
            plt.ylabel("Mean f")
            plt.title("Relationship between p and Mean f")
            plt.grid(True)
            plt.savefig(f"{out_dir}/p_pq_diff_vs_mean_f.png")
            plt.close()

            # Plot percentage below optimal
            plt.figure(figsize=(10, 6))
            plt.plot(
                results_df["p"], results_df["pct_below_optimal"], "o-", color="red"
            )
            plt.xlabel("p (in-cluster probability)")
            plt.ylabel("% of Random Atoms Below Optimal (%)")
            plt.title("Percentage of Random Atoms with f Values Below Optimal")
            plt.ylim(0, 100)  # Set y-axis from 0-100%
            plt.grid(True)
            plt.savefig(f"{out_dir}/p_vs_pct_below_optimal.png")
            plt.close()

        except Exception as e:
            print(f"Warning: Could not generate plots due to: {e}")

    return results_df


def sweep_q_parameter(
    q_values: List[float],
    num_atoms_per_graph: int = 100,
    save_results: bool = True,
    use_config_for_other_params: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Sweep over different values of q (between-cluster probability) and analyze how it affects confidence scores.

    Args:
        q_values: List of q values to sweep over
        num_atoms_per_graph: Number of random atoms to create per graph
        save_results: Whether to save results to CSV
        use_config_for_other_params: Whether to use config data for other parameters
        **kwargs: Additional parameters to override config values
            - p: In-cluster probability
            - samples_per_cluster: List of number of samples per cluster
            - num_features: Number of features
            - num_signal_features: Number of signal features
            - train_ratio: Ratio of training samples

    Returns:
        pandas.DataFrame: DataFrame containing results for each q value
    """
    # Get current date for results folder
    timestamp = get_timestamp()
    out_dir = f"results/{timestamp}/q_sweep_analysis"
    if save_results and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Results to collect
    results = []

    # Use parameters from config or override with kwargs
    if use_config_for_other_params:
        samples_per_cluster = kwargs.get(
            "samples_per_cluster", data.samples_per_cluster
        )
        num_features = kwargs.get("num_features", data.num_features)
        num_signal_features = kwargs.get(
            "num_signal_features", data.num_signal_features
        )
        p = kwargs.get("p", data.p)  # In-cluster probability
        train_ratio = kwargs.get("train_ratio", data.train_ratio)
    else:
        # Default values if not using config
        samples_per_cluster = kwargs.get("samples_per_cluster", [100, 100, 100])
        num_features = kwargs.get("num_features", 6)
        num_signal_features = kwargs.get("num_signal_features", 2)
        p = kwargs.get("p", 0.7)  # In-cluster probability
        train_ratio = kwargs.get("train_ratio", 0.8)

    # Iterate over q values
    for q_idx, q in enumerate(tqdm(q_values, desc="Sweeping q values")):
        # Create the graph with current q value
        c = len(samples_per_cluster)
        probs = np.full((c, c), q)
        np.fill_diagonal(probs, p)

        graph = SBM(
            samples_per_cluster,
            probs,
            num_features,
            num_signal_features,
            train_ratio=train_ratio,
        )

        # Calculate statistics
        stats = calculate_f_statistics(graph, num_atoms_per_graph)

        # Store results
        result = {
            "q": q,
            "p": p,
            "p_q_diff": p - q,
            "num_clusters": c,
            "num_features": num_features,
            "num_signal_features": num_signal_features,
            "mean_f": stats["mean_f"],
            "median_f": stats["median_f"],
            "min_f": stats["min_f"],
            "max_f": stats["max_f"],
            "std_f": stats["std_f"],
            "f_optimal": stats["f_optimal"],
            "f_gap": stats["f_gap"],
            "pct_below_optimal": stats["pct_below_optimal"],
        }

        results.append(result)

    # Create final DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    if save_results:
        results_df.to_csv(f"{out_dir}/q_sweep_analysis.csv", index=False)

        # Generate plots
        try:
            import matplotlib.pyplot as plt

            # Plot mean f vs optimal f
            plt.figure(figsize=(10, 6))
            plt.plot(results_df["q"], results_df["mean_f"], "o-", label="Mean f")
            plt.plot(results_df["q"], results_df["f_optimal"], "o-", label="Optimal f")
            plt.fill_between(
                results_df["q"],
                results_df["min_f"],
                results_df["max_f"],
                alpha=0.2,
                color="blue",
            )
            plt.xlabel("q (between-cluster probability)")
            plt.ylabel("f Value")
            plt.title("Effect of q on Confidence Scores")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{out_dir}/q_vs_f.png")
            plt.close()

            # Plot gap
            plt.figure(figsize=(10, 6))
            plt.plot(results_df["q"], results_df["f_gap"], "o-")
            plt.xlabel("q (between-cluster probability)")
            plt.ylabel("f Gap (Optimal - Mean)")
            plt.title("Effect of q on f Gap")
            plt.grid(True)
            plt.savefig(f"{out_dir}/q_vs_f_gap.png")
            plt.close()

            # Plot p-q relationship
            plt.figure(figsize=(10, 6))
            plt.scatter(
                results_df["q"],
                results_df["mean_f"],
                s=100,
                c=results_df["p_q_diff"],
                cmap="viridis",
            )
            plt.colorbar(label="p-q difference")
            plt.xlabel("q (between-cluster probability)")
            plt.ylabel("Mean f")
            plt.title("Relationship between q and Mean f")
            plt.grid(True)
            plt.savefig(f"{out_dir}/q_pq_diff_vs_mean_f.png")
            plt.close()

            # Plot percentage below optimal
            plt.figure(figsize=(10, 6))
            plt.plot(
                results_df["q"], results_df["pct_below_optimal"], "o-", color="red"
            )
            plt.xlabel("q (between-cluster probability)")
            plt.ylabel("% of Random Atoms Below Optimal (%)")
            plt.title("Percentage of Random Atoms with f Values Below Optimal")
            plt.ylim(0, 100)  # Set y-axis from 0-100%
            plt.grid(True)
            plt.savefig(f"{out_dir}/q_vs_pct_below_optimal.png")
            plt.close()

        except Exception as e:
            print(f"Warning: Could not generate plots due to: {e}")

    return results_df


def sweep_num_features(
    feature_values: List[int],
    num_atoms_per_graph: int = 100,
    save_results: bool = True,
    use_config_for_other_params: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Sweep over different numbers of features and analyze how it affects confidence scores.

    Args:
        feature_values: List of number of features to sweep over
        num_atoms_per_graph: Number of random atoms to create per graph
        save_results: Whether to save results to CSV
        use_config_for_other_params: Whether to use config data for other parameters
        **kwargs: Additional parameters to override config values
            - p: In-cluster probability
            - q: Between-cluster probability
            - samples_per_cluster: List of number of samples per cluster
            - num_signal_features: Number of signal features
            - train_ratio: Ratio of training samples

    Returns:
        pandas.DataFrame: DataFrame containing results for each number of features
    """
    # Get current date for results folder
    timestamp = get_timestamp()
    out_dir = f"results/{timestamp}/num_features_sweep_analysis"
    if save_results and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Results to collect
    results = []

    # Use parameters from config or override with kwargs
    if use_config_for_other_params:
        samples_per_cluster = kwargs.get(
            "samples_per_cluster", data.samples_per_cluster
        )
        num_signal_features = kwargs.get(
            "num_signal_features", data.num_signal_features
        )
        p = kwargs.get("p", data.p)
        q = kwargs.get("q", data.q)
        train_ratio = kwargs.get("train_ratio", data.train_ratio)
    else:
        # Default values if not using config
        samples_per_cluster = kwargs.get("samples_per_cluster", [100, 100, 100])
        num_signal_features = kwargs.get("num_signal_features", 2)
        p = kwargs.get("p", 0.7)
        q = kwargs.get("q", 0.1)
        train_ratio = kwargs.get("train_ratio", 0.8)

    # Iterate over number of features
    for num_features in tqdm(feature_values, desc="Sweeping feature count"):
        # Adjust num_signal_features if needed (can't have more signal features than total features)
        actual_num_signal_features = min(num_signal_features, num_features)

        # Create the graph
        c = len(samples_per_cluster)
        probs = np.full((c, c), q)
        np.fill_diagonal(probs, p)

        graph = SBM(
            samples_per_cluster,
            probs,
            num_features,
            actual_num_signal_features,
            train_ratio=train_ratio,
        )

        # Calculate statistics
        stats = calculate_f_statistics(graph, num_atoms_per_graph)

        # Store results
        result = {
            "num_features": num_features,
            "num_signal_features": actual_num_signal_features,
            "signal_ratio": actual_num_signal_features / num_features,
            "p": p,
            "q": q,
            "p_q_diff": p - q,
            "num_clusters": c,
            "mean_f": stats["mean_f"],
            "median_f": stats["median_f"],
            "min_f": stats["min_f"],
            "max_f": stats["max_f"],
            "std_f": stats["std_f"],
            "f_optimal": stats["f_optimal"],
            "f_gap": stats["f_gap"],
            "pct_below_optimal": stats["pct_below_optimal"],
        }

        results.append(result)

    # Create final DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    if save_results:
        results_df.to_csv(f"{out_dir}/num_features_sweep_analysis.csv", index=False)

        # Generate plots
        try:
            import matplotlib.pyplot as plt

            # Plot mean f vs optimal f
            plt.figure(figsize=(10, 6))
            plt.plot(
                results_df["num_features"], results_df["mean_f"], "o-", label="Mean f"
            )
            plt.plot(
                results_df["num_features"],
                results_df["f_optimal"],
                "o-",
                label="Optimal f",
            )
            plt.fill_between(
                results_df["num_features"],
                results_df["min_f"],
                results_df["max_f"],
                alpha=0.2,
                color="blue",
            )
            plt.xlabel("Number of Features")
            plt.ylabel("f Value")
            plt.title("Effect of Feature Count on Confidence Scores")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{out_dir}/features_vs_f.png")
            plt.close()

            # Plot gap
            plt.figure(figsize=(10, 6))
            plt.plot(results_df["num_features"], results_df["f_gap"], "o-")
            plt.xlabel("Number of Features")
            plt.ylabel("f Gap (Optimal - Mean)")
            plt.title("Effect of Feature Count on f Gap")
            plt.grid(True)
            plt.savefig(f"{out_dir}/features_vs_f_gap.png")
            plt.close()

            # Plot signal ratio effect
            plt.figure(figsize=(10, 6))
            plt.scatter(
                results_df["num_features"],
                results_df["mean_f"],
                s=100,
                c=results_df["signal_ratio"],
                cmap="viridis",
            )
            plt.colorbar(label="Signal Feature Ratio")
            plt.xlabel("Number of Features")
            plt.ylabel("Mean f")
            plt.title("Relationship between Feature Count and Mean f")
            plt.grid(True)
            plt.savefig(f"{out_dir}/features_signal_ratio_vs_mean_f.png")
            plt.close()

            # Plot percentage below optimal
            plt.figure(figsize=(10, 6))
            plt.plot(
                results_df["num_features"],
                results_df["pct_below_optimal"],
                "o-",
                color="red",
            )
            plt.xlabel("Number of Features")
            plt.ylabel("% of Random Atoms Below Optimal (%)")
            plt.title("Percentage of Random Atoms with f Values Below Optimal")
            plt.ylim(0, 100)  # Set y-axis from 0-100%
            plt.grid(True)
            plt.savefig(f"{out_dir}/features_vs_pct_below_optimal.png")
            plt.close()

        except Exception as e:
            print(f"Warning: Could not generate plots due to: {e}")

    return results_df


def sweep_num_signal_features(
    signal_feature_values: List[int],
    num_atoms_per_graph: int = 100,
    save_results: bool = True,
    use_config_for_other_params: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Sweep over different numbers of signal features and analyze how it affects confidence scores.

    Args:
        signal_feature_values: List of number of signal features to sweep over
        num_atoms_per_graph: Number of random atoms to create per graph
        save_results: Whether to save results to CSV
        use_config_for_other_params: Whether to use config data for other parameters
        **kwargs: Additional parameters to override config values
            - p: In-cluster probability
            - q: Between-cluster probability
            - samples_per_cluster: List of number of samples per cluster
            - num_features: Total number of features
            - train_ratio: Ratio of training samples

    Returns:
        pandas.DataFrame: DataFrame containing results for each number of signal features
    """
    # Get current date for results folder
    timestamp = get_timestamp()
    out_dir = f"results/{timestamp}/num_signal_features_sweep_analysis"
    if save_results and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Results to collect
    results = []

    # Use parameters from config or override with kwargs
    if use_config_for_other_params:
        samples_per_cluster = kwargs.get(
            "samples_per_cluster", data.samples_per_cluster
        )
        num_features = kwargs.get("num_features", data.num_features)
        p = kwargs.get("p", data.p)
        q = kwargs.get("q", data.q)
        train_ratio = kwargs.get("train_ratio", data.train_ratio)
    else:
        # Default values if not using config
        samples_per_cluster = kwargs.get("samples_per_cluster", [100, 100, 100])
        num_features = kwargs.get("num_features", 6)
        p = kwargs.get("p", 0.7)
        q = kwargs.get("q", 0.1)
        train_ratio = kwargs.get("train_ratio", 0.8)

    # Filter signal feature values to ensure they don't exceed total features
    valid_signal_feature_values = [
        v for v in signal_feature_values if v <= num_features
    ]

    # Iterate over number of signal features
    for num_signal_features in tqdm(
        valid_signal_feature_values, desc="Sweeping signal feature count"
    ):
        # Create the graph
        c = len(samples_per_cluster)
        probs = np.full((c, c), q)
        np.fill_diagonal(probs, p)

        graph = SBM(
            samples_per_cluster,
            probs,
            num_features,
            num_signal_features,
            train_ratio=train_ratio,
        )

        # Calculate statistics
        stats = calculate_f_statistics(graph, num_atoms_per_graph)

        # Store results
        result = {
            "num_signal_features": num_signal_features,
            "num_features": num_features,
            "signal_ratio": num_signal_features / num_features,
            "p": p,
            "q": q,
            "p_q_diff": p - q,
            "num_clusters": c,
            "mean_f": stats["mean_f"],
            "median_f": stats["median_f"],
            "min_f": stats["min_f"],
            "max_f": stats["max_f"],
            "std_f": stats["std_f"],
            "f_optimal": stats["f_optimal"],
            "f_gap": stats["f_gap"],
            "pct_below_optimal": stats["pct_below_optimal"],
        }

        results.append(result)

    # Create final DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    if save_results:
        results_df.to_csv(
            f"{out_dir}/num_signal_features_sweep_analysis.csv", index=False
        )

        # Generate plots
        try:
            import matplotlib.pyplot as plt

            # Plot mean f vs optimal f
            plt.figure(figsize=(10, 6))
            plt.plot(
                results_df["num_signal_features"],
                results_df["mean_f"],
                "o-",
                label="Mean f",
            )
            plt.plot(
                results_df["num_signal_features"],
                results_df["f_optimal"],
                "o-",
                label="Optimal f",
            )
            plt.fill_between(
                results_df["num_signal_features"],
                results_df["min_f"],
                results_df["max_f"],
                alpha=0.2,
                color="blue",
            )
            plt.xlabel("Number of Signal Features")
            plt.ylabel("f Value")
            plt.title("Effect of Signal Feature Count on Confidence Scores")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{out_dir}/signal_features_vs_f.png")
            plt.close()

            # Plot gap
            plt.figure(figsize=(10, 6))
            plt.plot(results_df["num_signal_features"], results_df["f_gap"], "o-")
            plt.xlabel("Number of Signal Features")
            plt.ylabel("f Gap (Optimal - Mean)")
            plt.title("Effect of Signal Feature Count on f Gap")
            plt.grid(True)
            plt.savefig(f"{out_dir}/signal_features_vs_f_gap.png")
            plt.close()

            # Plot signal ratio effect
            plt.figure(figsize=(10, 6))
            plt.scatter(
                results_df["num_signal_features"],
                results_df["mean_f"],
                s=100,
                c=results_df["signal_ratio"],
                cmap="viridis",
            )
            plt.colorbar(label="Signal Feature Ratio")
            plt.xlabel("Number of Signal Features")
            plt.ylabel("Mean f")
            plt.title("Relationship between Signal Feature Count and Mean f")
            plt.grid(True)
            plt.savefig(f"{out_dir}/signal_features_ratio_vs_mean_f.png")
            plt.close()

            # Plot percentage below optimal
            plt.figure(figsize=(10, 6))
            plt.plot(
                results_df["num_signal_features"],
                results_df["pct_below_optimal"],
                "o-",
                color="red",
            )
            plt.xlabel("Number of Signal Features")
            plt.ylabel("% of Random Atoms Below Optimal (%)")
            plt.title("Percentage of Random Atoms with f Values Below Optimal")
            plt.ylim(0, 100)  # Set y-axis from 0-100%
            plt.grid(True)
            plt.savefig(f"{out_dir}/signal_features_vs_pct_below_optimal.png")
            plt.close()

        except Exception as e:
            print(f"Warning: Could not generate plots due to: {e}")

    return results_df


def sweep_num_clusters(
    cluster_counts: List[int],
    cluster_size: int = 100,
    num_atoms_per_graph: int = 100,
    save_results: bool = True,
    use_config_for_other_params: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Sweep over different numbers of clusters and analyze how it affects confidence scores.

    Args:
        cluster_counts: List of number of clusters to sweep over
        cluster_size: Size of each cluster (if uniform)
        num_atoms_per_graph: Number of random atoms to create per graph
        save_results: Whether to save results to CSV
        use_config_for_other_params: Whether to use config data for other parameters
        **kwargs: Additional parameters to override config values
            - p: In-cluster probability
            - q: Between-cluster probability
            - num_features: Number of features
            - num_signal_features: Number of signal features
            - train_ratio: Ratio of training samples

    Returns:
        pandas.DataFrame: DataFrame containing results for each number of clusters
    """
    # Get current date for results folder
    timestamp = get_timestamp()
    out_dir = f"results/{timestamp}/num_clusters_sweep_analysis"
    if save_results and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Results to collect
    results = []

    # Use parameters from config or override with kwargs
    if use_config_for_other_params:
        num_features = kwargs.get("num_features", data.num_features)
        num_signal_features = kwargs.get(
            "num_signal_features", data.num_signal_features
        )
        p = kwargs.get("p", data.p)
        q = kwargs.get("q", data.q)
        train_ratio = kwargs.get("train_ratio", data.train_ratio)
    else:
        # Default values if not using config
        num_features = kwargs.get("num_features", 6)
        num_signal_features = kwargs.get("num_signal_features", 2)
        p = kwargs.get("p", 0.7)
        q = kwargs.get("q", 0.1)
        train_ratio = kwargs.get("train_ratio", 0.8)

    # Iterate over number of clusters
    for num_clusters in tqdm(cluster_counts, desc="Sweeping cluster count"):
        # Create samples_per_cluster with equal size clusters
        samples_per_cluster = [cluster_size] * num_clusters

        # Create the graph
        probs = np.full((num_clusters, num_clusters), q)
        np.fill_diagonal(probs, p)

        graph = SBM(
            samples_per_cluster,
            probs,
            num_features,
            num_signal_features,
            train_ratio=train_ratio,
        )

        # Calculate statistics
        stats = calculate_f_statistics(graph, num_atoms_per_graph)

        # Store results
        result = {
            "num_clusters": num_clusters,
            "total_nodes": num_clusters * cluster_size,
            "p": p,
            "q": q,
            "p_q_diff": p - q,
            "num_features": num_features,
            "num_signal_features": num_signal_features,
            "mean_f": stats["mean_f"],
            "median_f": stats["median_f"],
            "min_f": stats["min_f"],
            "max_f": stats["max_f"],
            "std_f": stats["std_f"],
            "f_optimal": stats["f_optimal"],
            "f_gap": stats["f_gap"],
            "pct_below_optimal": stats["pct_below_optimal"],
        }

        results.append(result)

    # Create final DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    if save_results:
        results_df.to_csv(f"{out_dir}/num_clusters_sweep_analysis.csv", index=False)

        # Generate plots
        try:
            import matplotlib.pyplot as plt

            # Plot mean f vs optimal f
            plt.figure(figsize=(10, 6))
            plt.plot(
                results_df["num_clusters"], results_df["mean_f"], "o-", label="Mean f"
            )
            plt.plot(
                results_df["num_clusters"],
                results_df["f_optimal"],
                "o-",
                label="Optimal f",
            )
            plt.fill_between(
                results_df["num_clusters"],
                results_df["min_f"],
                results_df["max_f"],
                alpha=0.2,
                color="blue",
            )
            plt.xlabel("Number of Clusters")
            plt.ylabel("f Value")
            plt.title("Effect of Cluster Count on Confidence Scores")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{out_dir}/clusters_vs_f.png")
            plt.close()

            # Plot gap
            plt.figure(figsize=(10, 6))
            plt.plot(results_df["num_clusters"], results_df["f_gap"], "o-")
            plt.xlabel("Number of Clusters")
            plt.ylabel("f Gap (Optimal - Mean)")
            plt.title("Effect of Cluster Count on f Gap")
            plt.grid(True)
            plt.savefig(f"{out_dir}/clusters_vs_f_gap.png")
            plt.close()

            # Plot relationship between total nodes and mean_f
            plt.figure(figsize=(10, 6))
            plt.scatter(
                results_df["num_clusters"],
                results_df["mean_f"],
                s=100,
                c=results_df["total_nodes"],
                cmap="viridis",
            )
            plt.colorbar(label="Total Nodes")
            plt.xlabel("Number of Clusters")
            plt.ylabel("Mean f")
            plt.title("Relationship between Cluster Count and Mean f")
            plt.grid(True)
            plt.savefig(f"{out_dir}/clusters_nodes_vs_mean_f.png")
            plt.close()

            # Plot percentage below optimal
            plt.figure(figsize=(10, 6))
            plt.plot(
                results_df["num_clusters"],
                results_df["pct_below_optimal"],
                "o-",
                color="red",
            )
            plt.xlabel("Number of Clusters")
            plt.ylabel("% of Random Atoms Below Optimal (%)")
            plt.title("Percentage of Random Atoms with f Values Below Optimal")
            plt.ylim(0, 100)  # Set y-axis from 0-100%
            plt.grid(True)
            plt.savefig(f"{out_dir}/clusters_vs_pct_below_optimal.png")
            plt.close()

        except Exception as e:
            print(f"Warning: Could not generate plots due to: {e}")

    return results_df


def sweep_cluster_size_distribution(
    size_distributions: List[List[int]],
    distribution_labels: List[str],
    num_atoms_per_graph: int = 100,
    save_results: bool = True,
    use_config_for_other_params: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Sweep over different cluster size distributions and analyze how it affects confidence scores.

    Args:
        size_distributions: List of lists, where each inner list is a cluster size distribution
        distribution_labels: List of labels for each distribution
        num_atoms_per_graph: Number of random atoms to create per graph
        save_results: Whether to save results to CSV
        use_config_for_other_params: Whether to use config data for other parameters
        **kwargs: Additional parameters to override config values
            - p: In-cluster probability
            - q: Between-cluster probability
            - num_features: Number of features
            - num_signal_features: Number of signal features
            - train_ratio: Ratio of training samples

    Returns:
        pandas.DataFrame: DataFrame containing results for each distribution
    """
    # Get current date for results folder
    timestamp = get_timestamp()
    out_dir = f"results/{timestamp}/cluster_size_distribution_analysis"
    if save_results and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Results to collect
    results = []

    # Use parameters from config or override with kwargs
    if use_config_for_other_params:
        num_features = kwargs.get("num_features", data.num_features)
        num_signal_features = kwargs.get(
            "num_signal_features", data.num_signal_features
        )
        p = kwargs.get("p", data.p)
        q = kwargs.get("q", data.q)
        train_ratio = kwargs.get("train_ratio", data.train_ratio)
    else:
        # Default values if not using config
        num_features = kwargs.get("num_features", 6)
        num_signal_features = kwargs.get("num_signal_features", 2)
        p = kwargs.get("p", 0.7)
        q = kwargs.get("q", 0.1)
        train_ratio = kwargs.get("train_ratio", 0.8)

    # Iterate over cluster size distributions
    for idx, (samples_per_cluster, label) in enumerate(
        zip(size_distributions, distribution_labels)
    ):
        # Create the graph
        c = len(samples_per_cluster)
        probs = np.full((c, c), q)
        np.fill_diagonal(probs, p)

        graph = SBM(
            samples_per_cluster,
            probs,
            num_features,
            num_signal_features,
            train_ratio=train_ratio,
        )

        # Calculate statistics
        stats = calculate_f_statistics(graph, num_atoms_per_graph)

        # Store results
        result = {
            "distribution_idx": idx,
            "distribution_label": label,
            "samples_per_cluster": str(
                samples_per_cluster
            ),  # Convert to string for storage
            "num_clusters": c,
            "total_nodes": sum(samples_per_cluster),
            "min_cluster_size": min(samples_per_cluster),
            "max_cluster_size": max(samples_per_cluster),
            "avg_cluster_size": np.mean(samples_per_cluster),
            "std_cluster_size": np.std(samples_per_cluster),
            "p": p,
            "q": q,
            "p_q_diff": p - q,
            "num_features": num_features,
            "num_signal_features": num_signal_features,
            "mean_f": stats["mean_f"],
            "median_f": stats["median_f"],
            "min_f": stats["min_f"],
            "max_f": stats["max_f"],
            "std_f": stats["std_f"],
            "f_optimal": stats["f_optimal"],
            "f_gap": stats["f_gap"],
            "pct_below_optimal": stats["pct_below_optimal"],
        }

        results.append(result)

    # Create final DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    if save_results:
        results_df.to_csv(
            f"{out_dir}/cluster_size_distribution_analysis.csv", index=False
        )

        # Generate plots
        try:
            import matplotlib.pyplot as plt

            # Plot mean f vs optimal f
            plt.figure(figsize=(12, 6))
            ind = np.arange(len(distribution_labels))
            width = 0.35

            plt.bar(ind, results_df["mean_f"], width, label="Mean f")
            plt.bar(ind + width, results_df["f_optimal"], width, label="Optimal f")

            plt.xlabel("Cluster Size Distribution")
            plt.ylabel("f Value")
            plt.title("Effect of Cluster Size Distribution on Confidence Scores")
            plt.xticks(ind + width / 2, distribution_labels, rotation=45)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{out_dir}/distribution_vs_f.png")
            plt.close()

            # Plot gap
            plt.figure(figsize=(12, 6))
            plt.bar(distribution_labels, results_df["f_gap"])
            plt.xlabel("Cluster Size Distribution")
            plt.ylabel("f Gap (Optimal - Mean)")
            plt.title("Effect of Cluster Size Distribution on f Gap")
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{out_dir}/distribution_vs_f_gap.png")
            plt.close()

            # Plot relationship between std of cluster size and mean_f
            plt.figure(figsize=(10, 6))
            plt.scatter(results_df["std_cluster_size"], results_df["mean_f"], s=100)
            for i, label in enumerate(distribution_labels):
                plt.annotate(
                    label,
                    (
                        results_df["std_cluster_size"].iloc[i],
                        results_df["mean_f"].iloc[i],
                    ),
                )
            plt.xlabel("Standard Deviation of Cluster Sizes")
            plt.ylabel("Mean f")
            plt.title("Relationship between Cluster Size Variation and Mean f")
            plt.grid(True)
            plt.savefig(f"{out_dir}/cluster_size_std_vs_mean_f.png")
            plt.close()

            # Plot percentage below optimal
            plt.figure(figsize=(12, 6))
            plt.bar(distribution_labels, results_df["pct_below_optimal"], color="red")
            plt.xlabel("Cluster Size Distribution")
            plt.ylabel("% of Random Atoms Below Optimal (%)")
            plt.title("Percentage of Random Atoms with f Values Below Optimal")
            plt.ylim(0, 100)  # Set y-axis from 0-100%
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{out_dir}/distribution_vs_pct_below_optimal.png")
            plt.close()

        except Exception as e:
            print(f"Warning: Could not generate plots due to: {e}")

    return results_df


def run_comprehensive_parameter_sweep(
    p_values: Optional[List[float]] = None,
    q_values: Optional[List[float]] = None,
    feature_values: Optional[List[int]] = None,
    signal_feature_values: Optional[List[int]] = None,
    cluster_counts: Optional[List[int]] = None,
    size_distributions: Optional[List[List[int]]] = None,
    distribution_labels: Optional[List[str]] = None,
    num_atoms_per_graph: int = 100,
    save_results: bool = True,
    use_config_for_other_params: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Run a comprehensive parameter sweep over multiple parameters.

    Args:
        p_values: List of p values to sweep over (default None means skip this sweep)
        q_values: List of q values to sweep over (default None means skip this sweep)
        feature_values: List of feature counts to sweep over (default None means skip this sweep)
        signal_feature_values: List of signal feature counts to sweep over (default None means skip this sweep)
        cluster_counts: List of cluster counts to sweep over (default None means skip this sweep)
        size_distributions: List of cluster size distributions to sweep over (default None means skip this sweep)
        distribution_labels: Labels for the size distributions (required if size_distributions is provided)
        num_atoms_per_graph: Number of random atoms to create per graph
        save_results: Whether to save results to CSV
        use_config_for_other_params: Whether to use config data for other parameters

    Returns:
        Dict[str, pd.DataFrame]: Dictionary of DataFrames containing results for each parameter sweep
    """
    results = {}

    # Get current date for results folder
    timestamp = get_timestamp()
    main_out_dir = f"results/{timestamp}/comprehensive_sweep_analysis"
    if save_results and not os.path.exists(main_out_dir):
        os.makedirs(main_out_dir)

    # Run p sweep if values provided
    if p_values is not None:
        print("Running p parameter sweep...")
        p_results = sweep_p_parameter(
            p_values=p_values,
            num_atoms_per_graph=num_atoms_per_graph,
            save_results=False,  # We'll save all results at the end
            use_config_for_other_params=use_config_for_other_params,
        )
        results["p_sweep"] = p_results

    # Run q sweep if values provided
    if q_values is not None:
        print("Running q parameter sweep...")
        q_results = sweep_q_parameter(
            q_values=q_values,
            num_atoms_per_graph=num_atoms_per_graph,
            save_results=False,
            use_config_for_other_params=use_config_for_other_params,
        )
        results["q_sweep"] = q_results

    # Run feature count sweep if values provided
    if feature_values is not None:
        print("Running feature count sweep...")
        feature_results = sweep_num_features(
            feature_values=feature_values,
            num_atoms_per_graph=num_atoms_per_graph,
            save_results=False,
            use_config_for_other_params=use_config_for_other_params,
        )
        results["feature_sweep"] = feature_results

    # Run signal feature count sweep if values provided
    if signal_feature_values is not None:
        print("Running signal feature count sweep...")
        signal_feature_results = sweep_num_signal_features(
            signal_feature_values=signal_feature_values,
            num_atoms_per_graph=num_atoms_per_graph,
            save_results=False,
            use_config_for_other_params=use_config_for_other_params,
        )
        results["signal_feature_sweep"] = signal_feature_results

    # Run cluster count sweep if values provided
    if cluster_counts is not None:
        print("Running cluster count sweep...")
        cluster_results = sweep_num_clusters(
            cluster_counts=cluster_counts,
            num_atoms_per_graph=num_atoms_per_graph,
            save_results=False,
            use_config_for_other_params=use_config_for_other_params,
        )
        results["cluster_sweep"] = cluster_results

    # Run cluster size distribution sweep if values provided
    if size_distributions is not None and distribution_labels is not None:
        print("Running cluster size distribution sweep...")
        distribution_results = sweep_cluster_size_distribution(
            size_distributions=size_distributions,
            distribution_labels=distribution_labels,
            num_atoms_per_graph=num_atoms_per_graph,
            save_results=False,
            use_config_for_other_params=use_config_for_other_params,
        )
        results["distribution_sweep"] = distribution_results

    # Save all results if requested
    if save_results:
        for sweep_name, df in results.items():
            df.to_csv(f"{main_out_dir}/{sweep_name}_results.csv", index=False)

        # Generate summary plot
        try:
            import matplotlib.pyplot as plt

            # Create a summary plot comparing parameter impacts
            plt.figure(figsize=(15, 10))

            # Plot bar chart comparing the mean gap across different parameter sweeps
            param_impact = {}
            param_pct_below_optimal = {}

            if "p_sweep" in results:
                param_impact["p"] = results["p_sweep"]["f_gap"].mean()
                param_pct_below_optimal["p"] = results["p_sweep"][
                    "pct_below_optimal"
                ].mean()

            if "q_sweep" in results:
                param_impact["q"] = results["q_sweep"]["f_gap"].mean()
                param_pct_below_optimal["q"] = results["q_sweep"][
                    "pct_below_optimal"
                ].mean()

            if "feature_sweep" in results:
                param_impact["num_features"] = results["feature_sweep"]["f_gap"].mean()
                param_pct_below_optimal["num_features"] = results["feature_sweep"][
                    "pct_below_optimal"
                ].mean()

            if "signal_feature_sweep" in results:
                param_impact["num_signal_features"] = results["signal_feature_sweep"][
                    "f_gap"
                ].mean()
                param_pct_below_optimal["num_signal_features"] = results[
                    "signal_feature_sweep"
                ]["pct_below_optimal"].mean()

            if "cluster_sweep" in results:
                param_impact["num_clusters"] = results["cluster_sweep"]["f_gap"].mean()
                param_pct_below_optimal["num_clusters"] = results["cluster_sweep"][
                    "pct_below_optimal"
                ].mean()

            if "distribution_sweep" in results:
                param_impact["cluster_distribution"] = results["distribution_sweep"][
                    "f_gap"
                ].mean()
                param_pct_below_optimal["cluster_distribution"] = results[
                    "distribution_sweep"
                ]["pct_below_optimal"].mean()

            if param_impact:
                # Plot f_gap impact
                plt.subplot(2, 1, 1)
                plt.bar(param_impact.keys(), param_impact.values())
                plt.xlabel("Parameter")
                plt.ylabel("Average f Gap (Optimal - Mean)")
                plt.title("Impact of Different Parameters on f Gap")
                plt.xticks(rotation=45)
                plt.grid(True)

                # Plot percentage below optimal
                plt.subplot(2, 1, 2)
                plt.bar(
                    param_pct_below_optimal.keys(),
                    param_pct_below_optimal.values(),
                    color="red",
                )
                plt.xlabel("Parameter")
                plt.ylabel("Avg % Below Optimal")
                plt.title(
                    "Impact of Different Parameters on % of f Values Below Optimal"
                )
                plt.xticks(rotation=45)
                plt.ylim(0, 100)
                plt.grid(True)

                plt.tight_layout()
                plt.savefig(f"{main_out_dir}/parameter_impact_summary.png")
                plt.close()

        except Exception as e:
            print(f"Warning: Could not generate summary plot due to: {e}")

    return results


if __name__ == "__main__":
    # Use this to analyze multiple graphs with many random atoms per graph using config parameters
    # results_df = analyze_random_graphs_from_config(
    #     num_graphs=10, num_atoms_per_graph=100
    # )
    # print(f"Analysis completed. Results summary:")
    # print(
    #     results_df[
    #         ["graph_idx", "c", "p", "q", "mean_f", "f_optimal", "f_gap"]
    #     ].to_string(index=False)
    # )
    p_values = np.linspace(0.01, 0.1, 10)  # [0.1, 0.2, ..., 0.9]
    p_results = sweep_p_parameter(p_values=p_values, num_atoms_per_graph=10000)

    # Example of running parameter sweeps:
    """
    # Sweep p parameter
    p_values = np.linspace(0.1, 0.9, 9)  # [0.1, 0.2, ..., 0.9]
    p_results = sweep_p_parameter(p_values=p_values, num_atoms_per_graph=50)
    
    # Sweep q parameter
    q_values = np.linspace(0.01, 0.3, 10)  # [0.01, 0.04, ..., 0.3]
    q_results = sweep_q_parameter(q_values=q_values, num_atoms_per_graph=50)
    
    # Sweep feature counts
    feature_values = [2, 4, 6, 8, 10, 15, 20]
    feature_results = sweep_num_features(feature_values=feature_values)
    
    # Sweep signal feature counts
    signal_feature_values = [1, 2, 3, 4, 5]
    signal_feature_results = sweep_num_signal_features(signal_feature_values=signal_feature_values)
    
    # Sweep cluster counts
    cluster_counts = [2, 3, 4, 5, 6, 8, 10]
    cluster_results = sweep_num_clusters(cluster_counts=cluster_counts)
    
    # Sweep cluster size distributions
    size_distributions = [
        [100, 100, 100],  # Balanced
        [50, 100, 150],   # Slightly imbalanced
        [30, 100, 170],   # Very imbalanced
        [20, 50, 100, 230], # Four clusters, highly imbalanced
        [80, 80, 80, 80, 80], # Five equal clusters
    ]
    distribution_labels = ["Balanced", "Slightly Imbal", "Very Imbal", "Four Clusters", "Five Equal"]
    distribution_results = sweep_cluster_size_distribution(
        size_distributions=size_distributions,
        distribution_labels=distribution_labels
    )
    
    # Or run a comprehensive sweep
    comprehensive_results = run_comprehensive_parameter_sweep(
        p_values=np.linspace(0.1, 0.9, 5),
        q_values=np.linspace(0.01, 0.3, 5),
        feature_values=[2, 6, 10, 15],
        signal_feature_values=[1, 2, 3, 4],
        cluster_counts=[2, 3, 5],
        size_distributions=[
            [100, 100, 100],  # Balanced
            [50, 100, 150],   # Slightly imbalanced
            [30, 100, 170],   # Very imbalanced
        ],
        distribution_labels=["Balanced", "Slightly Imbal", "Very Imbal"],
        num_atoms_per_graph=50
    )
    """
