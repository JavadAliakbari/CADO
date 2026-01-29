# Training Ratio Sweep Experiment

## Overview

This experiment sweeps the **training ratio parameter** from 0.1 to 0.9 to test how the amount of labeled training data affects CADO's performance across different modes.

## What is Training Ratio?

- **Training Ratio**: Fraction of nodes with labels available during training
- **Low ratio (0.1-0.3)**: Few labeled examples → semi-supervised challenge
- **Medium ratio (0.4-0.6)**: Balanced train/test split → realistic setting
- **High ratio (0.7-0.9)**: Abundant labels → supervised learning

## Quick Start

### Test First
```bash
python src/test_train_ratio_sweep.py
```

### Run Full Experiment
```bash
# With automatic CPU detection
python src/train_ratio_sweep_experiment.py

# With custom worker count
python src/train_ratio_sweep_experiment.py --workers 4
```

## Experiment Parameters

- **Training ratio values**: 0.1 to 0.9 (step: 0.1) = 9 values
- **Modes**: 4 (Feature+Label, Graph+Feature, Graph+Feature+Label, Feature Only)
- **Runs per config**: 10
- **Total CADO runs**: 9 × 4 × 10 = 360
- **Parallel processing**: Uses multiple CPU cores
- **Duration**: ~10-20 minutes (depending on CPUs)

## Fixed Parameters

During the sweep, these remain constant:
- **p**: Within-cluster connection probability
- **q**: Between-cluster connection probability
- **sigma**: GMM feature noise parameter
- Graph structure and feature parameters

## Four Modes

| Mode | mu0 | mu2 | Description |
|------|-----|-----|-------------|
| **Feature+Label** | 0 | default | Features + labels only |
| **Graph+Feature** | default | 0 | Graph + features only |
| **Graph+Feature+Label** | default | default | All information |
| **Feature Only** | 0 | 0 | Baseline (no graph) |

## Expected Results

### Predicted Trends

1. **Low training ratio (0.1-0.3)**:
   - Label-based modes struggle with few examples
   - Graph+Feature may perform relatively better
   - Semi-supervised learning challenge
   
2. **Medium training ratio (0.4-0.6)**:
   - Balanced performance across modes
   - Graph+Feature+Label should lead
   - Representative of real-world scenarios
   
3. **High training ratio (0.7-0.9)**:
   - Label-based modes achieve peak performance
   - Graph+Feature+Label should dominate
   - Diminishing returns as ratio increases

### Key Insights

This experiment reveals:
- **Label efficiency**: How much labeled data does CADO need?
- **Semi-supervised capability**: Can graph structure compensate for few labels?
- **Mode sensitivity**: Which modes are most affected by training ratio?
- **Critical threshold**: Minimum training ratio for good performance?

## Scientific Questions

1. **Q**: How many labeled examples does CADO need?
   **A**: Find minimum training ratio for acceptable performance

2. **Q**: Can graph structure help with few labels?
   **A**: Compare Graph+Feature vs Feature+Label at low ratios

3. **Q**: Is there a saturation point?
   **A**: Check if performance plateaus at high training ratios

4. **Q**: Which mode is most label-efficient?
   **A**: Compare accuracy curves across training ratios

## Output Files

```
results/YYYYMMDD_HHMMSS/
├── train_ratio_sweep_results.csv          # Wide format: ratio × modes
├── train_ratio_sweep_results_detailed.csv # Long format: all metadata
└── summary.txt                             # Experiment summary
```

### CSV Format (Wide)

```csv
train_ratio,Feature Only,Feature+Label,Graph+Feature,Graph+Feature+Label
0.1000,0.3351,0.4234,0.5567,0.6823
0.2000,0.3349,0.5156,0.6345,0.7712
...
```

## Usage Examples

### Basic Run
```bash
python src/train_ratio_sweep_experiment.py
```

### Custom Training Ratio Range
Edit the file to change:
```python
train_ratio_values = np.arange(0.1, 1.0, 0.1)  # Line ~179
# Change to:
train_ratio_values = np.arange(0.2, 0.9, 0.05)  # More granular
```

### Fewer Runs (Faster)
```python
n_runs_per_mode = 5  # Line ~180 (change from 10)
```

### Serial Mode (Debugging)
```bash
python src/train_ratio_sweep_experiment.py --workers 1
```

## Visualization

After running, analyze with:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('results/YYYYMMDD_HHMMSS/train_ratio_sweep_results.csv')

# Plot all modes
plt.figure(figsize=(12, 6))
for col in df.columns[1:]:  # Skip train_ratio column
    plt.plot(df['train_ratio'], df[col], marker='o', label=col, linewidth=2)

plt.xlabel('Training Ratio (Fraction of Labeled Nodes)')
plt.ylabel('Test Accuracy')
plt.title('CADO Performance vs Training Data Availability')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0.05, 0.95)
plt.ylim(0, 1.05)
plt.savefig('train_ratio_vs_accuracy.png', dpi=300)
plt.show()
```

## Comparison with Other Sweeps

| Aspect | P-Sweep | Sigma-Sweep | Train-Ratio-Sweep |
|--------|---------|-------------|-------------------|
| **Variable** | Graph density | Feature noise | Label availability |
| **Tests** | Graph quality | Feature quality | Supervision level |
| **Fixed** | σ, train_ratio | p, q, train_ratio | p, q, σ |
| **Insight** | Connectivity | Robustness | Label efficiency |
| **Range** | 0.0-0.2 | 0.1-2.0 | 0.1-0.9 |

## Practical Applications

### Real-world Scenarios

1. **Expensive labeling**: How much annotation is needed?
2. **Active learning**: Which samples to label first?
3. **Transfer learning**: Can we leverage unlabeled data?
4. **Budget constraints**: Minimum labels for deployment?

### Recommendations Based on Results

- **High budget**: Use train_ratio ≥ 0.7 for best performance
- **Medium budget**: train_ratio = 0.4-0.6 offers good balance
- **Low budget**: Rely on Graph+Feature if train_ratio < 0.3

## Combining Multiple Sweeps

For comprehensive analysis:

```bash
# Run all three experiments
python src/p_sweep_experiment.py
python src/sigma_sweep_experiment.py
python src/train_ratio_sweep_experiment.py

# Results show interactions between:
# - Graph structure (p)
# - Feature quality (sigma)
# - Label availability (train_ratio)
```

This reveals the complete picture of CADO's robustness!

## Tips

1. **Start with test**: Always run `test_train_ratio_sweep.py` first
2. **Monitor progress**: Each training ratio is processed in parallel
3. **Check train/test split**: Verify nodes are correctly split
4. **Compare to baseline**: Feature Only shows non-graph performance

## Troubleshooting

### Unexpected Performance at Low Ratios
- Check if test set is being used correctly
- Verify training nodes are actually labeled
- Compare with Feature Only baseline

### No Performance Gain with More Labels
- May indicate overfitting or saturation
- Check if test accuracy plateaus
- Verify graph structure is consistent

### All Modes Perform Similarly
- Training ratio may not be limiting factor
- Graph/features may dominate over labels
- Check other parameters (p, sigma)

---

**Created**: 2026-01-29  
**Status**: ✅ Ready to use  
**Parallel**: ✅ Multi-core support  
**Output**: CSV with 4 decimal precision
