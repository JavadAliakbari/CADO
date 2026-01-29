"""
Quick test script for train_ratio-sweep experiment.
Tests with single training ratio and runs to verify everything works.
"""

import numpy as np
import sys
import os

# Ensure we're in the project root directory for correct imports and .env loading
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from train_ratio_sweep_experiment import run_single_experiment
from utils.SBM import SBM
from utils.utils import data


def quick_test():
    """Run a quick test with minimal parameters."""

    print("=" * 80)
    print("QUICK TEST OF TRAINING RATIO SWEEP EXPERIMENT")
    print("=" * 80)

    # Test parameters
    train_ratio_test = 0.5  # Single training ratio
    n_runs = 2  # Just 2 runs

    print(f"\nTest configuration:")
    print(f"  - train_ratio: {train_ratio_test}")
    print(f"  - Number of runs per mode: {n_runs}")
    print(f"  - p value: {data.p}")
    print(f"  - q value: {data.q}")
    print(f"  - sigma: {data.gmm_sigma}")
    print(f"  - Samples per cluster: {data.samples_per_cluster}")

    # Create test graph
    samples_per_cluster = data.samples_per_cluster
    num_features = data.num_features
    c = len(samples_per_cluster)
    p = data.p
    q = data.q

    probs = np.full((c, c), q)
    np.fill_diagonal(probs, p)

    print(f"\nGenerating SBM graph with train_ratio={train_ratio_test}...")
    graph = SBM(
        samples_per_cluster,
        probs,
        num_features,
        train_ratio=train_ratio_test,
        gmm_center_min=data.gmm_center_min,
        gmm_min_separation=data.gmm_min_separation,
        gmm_sigma=data.gmm_sigma,
    )
    print(f"  - Graph created: {graph.num_nodes} nodes, {graph.num_clusters} clusters")
    print(f"  - Training nodes: {len(graph.train_nodes)} ({train_ratio_test*100:.0f}%)")
    print(
        f"  - Test nodes: {graph.num_nodes - len(graph.train_nodes)} ({(1-train_ratio_test)*100:.0f}%)"
    )

    # Get default mu values
    mu0_default = data.mu0
    mu1_default = data.mu1
    mu2_default = data.mu2

    # Define test modes
    modes = [
        {"name": "Feature+Label", "mu0": 0.0, "mu2": mu2_default},
        {"name": "Graph+Feature", "mu0": mu0_default, "mu2": 0.0},
        {
            "name": "Graph+Feature+Label",
            "mu0": mu0_default,
            "mu1": mu1_default,
            "mu2": mu2_default,
        },
        {"name": "Feature Only", "mu0": 0.0, "mu2": 0.0},
    ]

    print(f"\nTesting {len(modes)} modes...")
    print("-" * 80)

    results = []

    for i, mode in enumerate(modes, 1):
        mode_name = mode["name"]
        mu0 = mode["mu0"]
        mu2 = mode["mu2"]

        print(f"\n[{i}/{len(modes)}] Testing {mode_name}")
        print(f"      Parameters: mu0={mu0}, mu2={mu2}")

        try:
            avg_acc, std_acc = run_single_experiment(
                graph, mode_name, mu0, mu2, n_runs=n_runs
            )

            results.append(
                {
                    "mode": mode_name,
                    "avg_acc": avg_acc,
                    "std_acc": std_acc,
                    "status": "✓ SUCCESS",
                }
            )

            print(f"      Result: ACC = {avg_acc:.4f} ± {std_acc:.4f} ✓")

        except Exception as e:
            results.append(
                {
                    "mode": mode_name,
                    "avg_acc": None,
                    "std_acc": None,
                    "status": f"✗ FAILED: {str(e)}",
                }
            )
            print(f"      ✗ FAILED: {str(e)}")

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    print(f"\n{'Mode':<30} {'ACC':<20} {'Status':<20}")
    print("-" * 70)

    all_passed = True
    for result in results:
        if result["avg_acc"] is not None:
            acc_str = f"{result['avg_acc']:.4f} ± {result['std_acc']:.4f}"
        else:
            acc_str = "N/A"
            all_passed = False

        print(f"{result['mode']:<30} {acc_str:<20} {result['status']:<20}")

    print("\n" + "=" * 80)

    if all_passed:
        print("\n✓ ALL TESTS PASSED!")
        print("\nYou can now run the full experiment with:")
        print("  python train_ratio_sweep_experiment.py")
        print("\nOr with custom workers:")
        print("  python train_ratio_sweep_experiment.py --workers 4")
    else:
        print("\n✗ SOME TESTS FAILED!")
        print("\nPlease check the error messages above.")
        return 1

    print("\n" + "=" * 80)
    return 0


if __name__ == "__main__":
    exit_code = quick_test()
    sys.exit(exit_code)
