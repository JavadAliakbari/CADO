# Sigma Sweep Experiment

## Overview

This experiment sweeps the **GMM sigma parameter** (feature noise/spread) from 0.1 to 2.0 to test how feature quality affects CADO's performance across different modes.

## What is Sigma?

- **Sigma (σ)**: Standard deviation of the Gaussian Mixture Model for generating node features
- **Low sigma (0.1-0.5)**: Features are tightly clustered → easier to separate
- **Medium sigma (0.5-1.0)**: Moderate feature spread → realistic setting
- **High sigma (1.0-2.0)**: Features are noisy/diffuse → harder to separate

## Quick Start

### Test First
```bash
python src/test_sigma_sweep.py
```

### Run Full Experiment
```bash
# With automatic CPU detection
python src/sigma_sweep_experiment.py

# With custom worker count
python src/sigma_sweep_experiment.py --workers 4
```

## Experiment Parameters

- **Sigma values**: 0.1 to 2.0 (step: 0.1) = 20 values
- **Modes**: 4 (Graph+Label, Graph+Feature, Graph+Feature+Label, Graph Only)
- **Runs per config**: 10
- **Total CADO runs**: 20 × 4 × 10 = 800
- **Parallel processing**: Uses multiple CPU cores
- **Duration**: ~15-40 minutes (depending on CPUs)

## Fixed Parameters

During the sweep, these remain constant:
- **p**: Within-cluster connection probability (from config)
- **q**: Between-cluster connection probability (from config)
- Graph structure parameters (samples_per_cluster, num_features, etc.)

## Four Modes

| Mode | mu1 | mu2 | Description |
|------|-----|-----|-------------|
| **Graph+Label** | 0 | default | Graph + labels only |
| **Graph+Feature** | default | 0 | Graph + features only |
| **Graph+Feature+Label** | default | default | All information |
| **Graph Only** | 0 | 0 | Baseline (no features/labels) |

## Expected Results

### Predicted Trends

1. **Low sigma (clean features)**:
   - Graph+Feature modes should excel
   - Graph+Feature+Label should achieve highest accuracy
   
2. **Medium sigma (realistic)**:
   - Performance degradation in feature-based modes
   - Graph structure becomes more important
   
3. **High sigma (noisy features)**:
   - Graph+Feature may struggle
   - Graph+Label may outperform Graph+Feature
   - Graph Only provides baseline

### Key Insights

This experiment reveals:
- **Feature robustness**: How well does CADO handle noisy features?
- **Mode comparison**: When do labels become more valuable than features?
- **Critical sigma**: At what noise level do features hurt rather than help?

## Output Files

```
results/YYYYMMDD_HHMMSS/
├── sigma_sweep_results.csv          # Wide format: sigma × modes
├── sigma_sweep_results_detailed.csv # Long format: all metadata
└── summary.txt                       # Experiment summary
```

### CSV Format (Wide)

```csv
sigma,Graph Only,Graph+Label,Graph+Feature,Graph+Feature+Label
0.1000,0.3351,0.8234,0.9567,0.9823
0.2000,0.3349,0.8156,0.9345,0.9712
...
```

### CSV Format (Detailed)

```csv
sigma,p,q,mode,mu1,mu2,avg_acc,std_acc,n_runs
0.1,0.08,0.05,Graph+Label,0.0,0.1,0.8234,0.0123,10
...
```

## Usage Examples

### Basic Run
```bash
python src/sigma_sweep_experiment.py
```

### Custom Sigma Range
Edit the file to change:
```python
sigma_values = np.arange(0.1, 2.1, 0.1)  # Line ~178
# Change to:
sigma_values = np.arange(0.1, 1.1, 0.05)  # More granular
```

### Fewer Runs (Faster)
```python
n_runs_per_mode = 5  # Line ~179 (change from 10)
```

### Serial Mode (Debugging)
```bash
python src/sigma_sweep_experiment.py --workers 1
```

## Visualization

After running, analyze with:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('results/YYYYMMDD_HHMMSS/sigma_sweep_results.csv')

# Plot all modes
plt.figure(figsize=(12, 6))
for col in df.columns[1:]:  # Skip sigma column
    plt.plot(df['sigma'], df[col], marker='o', label=col)

plt.xlabel('Sigma (Feature Noise)')
plt.ylabel('Accuracy')
plt.title('CADO Performance vs Feature Noise')
plt.legend()
plt.grid(True)
plt.savefig('sigma_vs_accuracy.png')
plt.show()
```

## Comparison with P-Sweep

| Aspect | P-Sweep | Sigma-Sweep |
|--------|---------|-------------|
| **Variable** | Graph density (p) | Feature noise (σ) |
| **Tests** | Graph structure quality | Feature quality |
| **Fixed** | σ, GMM parameters | p, q values |
| **Insight** | Graph connectivity impact | Feature robustness |
| **Expected winner** | Graph+All (high p) | Graph+All (low σ) |

## Combining Both Experiments

For comprehensive analysis:

```bash
# Run both experiments
python src/p_sweep_experiment.py
python src/sigma_sweep_experiment.py

# Compare results
python src/analyze_combined_sweeps.py  # (Create custom script)
```

This reveals interactions between graph structure and feature quality!

## Tips

1. **Start with test**: Always run `test_sigma_sweep.py` first
2. **Monitor memory**: Each worker uses ~70-100 MB
3. **Check results**: Look for critical sigma threshold where features stop helping
4. **Compare modes**: Graph+Feature should degrade faster than Graph+Label as sigma increases

## Troubleshooting

### Import Error
Scripts handle paths automatically. If issues persist:
```bash
cd /Users/javada/Desktop/CADO
python src/sigma_sweep_experiment.py
```

### Unexpected Results
- Verify sigma is actually changing in generated graphs
- Check that features are using the sigma parameter
- Compare with Graph Only baseline

### Slow Execution
```bash
# Reduce runs
python src/sigma_sweep_experiment.py  # Edit n_runs_per_mode

# Use fewer workers
python src/sigma_sweep_experiment.py --workers 2
```

## Scientific Questions

This experiment helps answer:

1. **Q**: How robust is CADO to feature noise?
   **A**: Compare performance degradation across sigma values

2. **Q**: When are labels more valuable than features?
   **A**: Find sigma where Graph+Label > Graph+Feature

3. **Q**: Is there an optimal feature noise level?
   **A**: Look for peak performance in middle range

4. **Q**: How does Graph Only baseline compare?
   **A**: Check if noisy features hurt more than help

---

**Created**: 2026-01-29  
**Status**: ✅ Ready to use  
**Parallel**: ✅ Multi-core support  
**Output**: CSV with 4 decimal precision
