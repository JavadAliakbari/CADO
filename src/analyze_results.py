"""
Example analysis script showing how to work with p-sweep results.
"""

import pandas as pd
import numpy as np
import sys


def analyze_results(csv_path):
    """
    Perform detailed analysis on p-sweep experiment results.

    Args:
        csv_path: Path to the p_sweep_results.csv file
    """
    # Load data
    df = pd.read_csv(csv_path)

    print("=" * 80)
    print("P-SWEEP RESULTS ANALYSIS")
    print("=" * 80)

    # Basic info
    print(f"\nDataset Info:")
    print(f"  - Total rows: {len(df)}")
    print(
        f"  - P values tested: {df['p'].nunique()} (from {df['p'].min():.2f} to {df['p'].max():.2f})"
    )
    print(f"  - Modes tested: {df['mode'].nunique()}")
    print(f"  - Runs per configuration: {df['n_runs'].iloc[0]}")

    # Mode names
    modes = df["mode"].unique()
    print(f"\nModes: {', '.join(modes)}")

    # Overall statistics
    print("\n" + "-" * 80)
    print("OVERALL STATISTICS (averaged across all p values)")
    print("-" * 80)
    print(
        f"\n{'Mode':<30} {'Mean ARI':<12} {'Std ARI':<12} {'Min ARI':<12} {'Max ARI':<12}"
    )
    print("-" * 78)

    for mode in modes:
        mode_data = df[df["mode"] == mode]
        mean_ari = mode_data["avg_ari"].mean()
        std_ari = mode_data["avg_ari"].std()
        min_ari = mode_data["avg_ari"].min()
        max_ari = mode_data["avg_ari"].max()
        print(
            f"{mode:<30} {mean_ari:<12.4f} {std_ari:<12.4f} {min_ari:<12.4f} {max_ari:<12.4f}"
        )

    # Find best performing configurations
    print("\n" + "-" * 80)
    print("TOP 5 BEST CONFIGURATIONS")
    print("-" * 80)
    top5 = df.nlargest(5, "avg_ari")
    print(f"\n{'Rank':<6} {'Mode':<30} {'p':<8} {'ARI':<15}")
    print("-" * 60)
    for idx, (i, row) in enumerate(top5.iterrows(), 1):
        print(
            f"{idx:<6} {row['mode']:<30} {row['p']:<8.3f} {row['avg_ari']:.4f}±{row['std_ari']:.4f}"
        )

    # Worst configurations
    print("\n" + "-" * 80)
    print("TOP 5 WORST CONFIGURATIONS")
    print("-" * 80)
    bottom5 = df.nsmallest(5, "avg_ari")
    print(f"\n{'Rank':<6} {'Mode':<30} {'p':<8} {'ARI':<15}")
    print("-" * 60)
    for idx, (i, row) in enumerate(bottom5.iterrows(), 1):
        print(
            f"{idx:<6} {row['mode']:<30} {row['p']:<8.3f} {row['avg_ari']:.4f}±{row['std_ari']:.4f}"
        )

    # Mode comparison at specific p values
    print("\n" + "-" * 80)
    print("MODE COMPARISON AT KEY P VALUES")
    print("-" * 80)

    key_p_values = [0.0, 0.05, 0.10, 0.15, 0.20]
    available_p = [p for p in key_p_values if p in df["p"].values]

    for p_val in available_p:
        print(f"\np = {p_val:.2f}:")
        p_data = df[df["p"] == p_val].sort_values("avg_ari", ascending=False)
        for i, row in p_data.iterrows():
            print(
                f"  {row['mode']:<30} ARI = {row['avg_ari']:.4f} ± {row['std_ari']:.4f}"
            )

    # Performance improvement analysis
    print("\n" + "-" * 80)
    print("PERFORMANCE IMPROVEMENT: p=0.00 vs p=0.20")
    print("-" * 80)

    for mode in modes:
        p0_data = df[(df["mode"] == mode) & (df["p"] == 0.0)]
        p20_data = df[(df["mode"] == mode) & (df["p"] == 0.20)]

        if not p0_data.empty and not p20_data.empty:
            ari_0 = p0_data["avg_ari"].iloc[0]
            ari_20 = p20_data["avg_ari"].iloc[0]
            improvement = ari_20 - ari_0
            pct_improvement = (improvement / (ari_0 + 1e-10)) * 100

            print(f"\n{mode}:")
            print(f"  p=0.00: ARI = {ari_0:.4f}")
            print(f"  p=0.20: ARI = {ari_20:.4f}")
            print(f"  Improvement: {improvement:+.4f} ({pct_improvement:+.1f}%)")

    # Stability analysis (lowest standard deviation)
    print("\n" + "-" * 80)
    print("STABILITY ANALYSIS (configurations with lowest variability)")
    print("-" * 80)
    most_stable = df.nsmallest(5, "std_ari")
    print(f"\n{'Rank':<6} {'Mode':<30} {'p':<8} {'Std ARI':<12} {'Avg ARI':<12}")
    print("-" * 68)
    for idx, (i, row) in enumerate(most_stable.iterrows(), 1):
        print(
            f"{idx:<6} {row['mode']:<30} {row['p']:<8.3f} {row['std_ari']:<12.4f} {row['avg_ari']:<12.4f}"
        )

    # Statistical summary by mode
    print("\n" + "-" * 80)
    print("DETAILED MODE STATISTICS")
    print("-" * 80)

    for mode in modes:
        mode_data = df[df["mode"] == mode]
        print(f"\n{mode}:")
        print(f"  Number of p values: {len(mode_data)}")
        print(f"  Mean ARI: {mode_data['avg_ari'].mean():.4f}")
        print(f"  Median ARI: {mode_data['avg_ari'].median():.4f}")
        print(f"  Std Dev: {mode_data['avg_ari'].std():.4f}")
        print(
            f"  Min ARI: {mode_data['avg_ari'].min():.4f} (at p={mode_data.loc[mode_data['avg_ari'].idxmin(), 'p']:.3f})"
        )
        print(
            f"  Max ARI: {mode_data['avg_ari'].max():.4f} (at p={mode_data.loc[mode_data['avg_ari'].idxmax(), 'p']:.3f})"
        )
        print(f"  25th percentile: {mode_data['avg_ari'].quantile(0.25):.4f}")
        print(f"  75th percentile: {mode_data['avg_ari'].quantile(0.75):.4f}")
        print(f"  Average std across runs: {mode_data['std_ari'].mean():.4f}")

    # Correlation with p
    print("\n" + "-" * 80)
    print("CORRELATION BETWEEN p AND ARI")
    print("-" * 80)

    for mode in modes:
        mode_data = df[df["mode"] == mode]
        correlation = np.corrcoef(mode_data["p"], mode_data["avg_ari"])[0, 1]
        print(f"\n{mode}:")
        print(f"  Pearson correlation: {correlation:.4f}")
        if correlation > 0.7:
            print(f"  → Strong positive correlation (performance improves with p)")
        elif correlation > 0.3:
            print(f"  → Moderate positive correlation")
        elif correlation > -0.3:
            print(f"  → Weak or no correlation")
        else:
            print(f"  → Negative correlation (unexpected)")

    print("\n" + "=" * 80)


def export_summary_table(csv_path, output_path=None):
    """
    Export a summary table suitable for publication.

    Args:
        csv_path: Path to the p_sweep_results.csv file
        output_path: Path to save summary (if None, prints to console)
    """
    df = pd.read_csv(csv_path)

    # Create summary table
    summary_data = []
    modes = df["mode"].unique()

    for mode in modes:
        mode_data = df[df["mode"] == mode]
        summary_data.append(
            {
                "Mode": mode,
                "Mean ARI": f"{mode_data['avg_ari'].mean():.4f}",
                "Std ARI": f"{mode_data['avg_ari'].std():.4f}",
                "Min ARI": f"{mode_data['avg_ari'].min():.4f}",
                "Max ARI": f"{mode_data['avg_ari'].max():.4f}",
                "Avg Std (across runs)": f"{mode_data['std_ari'].mean():.4f}",
            }
        )

    summary_df = pd.DataFrame(summary_data)

    if output_path:
        summary_df.to_csv(output_path, index=False)
        print(f"\nSummary table saved to: {output_path}")
    else:
        print("\n" + "=" * 80)
        print("SUMMARY TABLE FOR PUBLICATION")
        print("=" * 80)
        print(summary_df.to_string(index=False))


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <path_to_csv>")
        print("\nExample:")
        print(
            "  python analyze_results.py results/p_sweep_20260129_120000/p_sweep_results.csv"
        )
        sys.exit(1)

    csv_path = sys.argv[1]

    try:
        analyze_results(csv_path)

        # Optionally export summary table
        output_dir = os.path.dirname(csv_path)
        summary_path = os.path.join(output_dir, "summary_table.csv")
        export_summary_table(csv_path, summary_path)

    except FileNotFoundError:
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error analyzing results: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import os

    main()
