# P-Sweep Experiment - Quick Start Guide

## Overview

The p-sweep experiment tests CADO's performance across different SBM (Stochastic Block Model) parameters by varying the within-cluster connection probability `p` from 0 to 0.2 with 0.01 increments.

## Files Created

1. **[p_sweep_experiment.py](src/p_sweep_experiment.py)** - Main experiment script
2. **[test_p_sweep.py](src/test_p_sweep.py)** - Quick test script  
3. **[visualize_p_sweep.py](src/visualize_p_sweep.py)** - Visualization script
4. **[run_p_sweep.sh](src/run_p_sweep.sh)** - Convenient bash runner (executable)
5. **[P_SWEEP_README.md](P_SWEEP_README.md)** - Detailed documentation

## Four Modes Tested

| Mode | mu1 | mu2 | Description |
|------|-----|-----|-------------|
| **Graph+Label** | 0.0 | default | Uses graph + labels, no features |
| **Graph+Feature** | default | 0.0 | Uses graph + features, no labels |
| **Graph+Feature+Label** | default | default | Uses all information sources |
| **Graph Only** | 0.0 | 0.0 | Subspace clustering baseline |

## Quick Start

### 1. Run Test First (Recommended)
```bash
cd src
./run_p_sweep.sh test
```
Or directly:
```bash
python src/test_p_sweep.py
```

### 2. Run Full Experiment
```bash
cd src
./run_p_sweep.sh full
```
Or directly:
```bash
python src/p_sweep_experiment.py
```

**Estimated time:** 1-2 hours (840 total CADO runs)

### 3. Generate Visualizations
```bash
python src/visualize_p_sweep.py results/p_sweep_YYYYMMDD_HHMMSS/
```

## Expected Output

### During Experiment
```
Running experiment with 21 p values and 4 modes
Each configuration will be run 10 times
Total runs: 840

Sweeping p values: 42%|████▏     | 9/21 [00:45<01:00, 5.0s/it, p=0.080, mode=Graph+Label, ARI=0.8234±0.0312]
```

### Results Directory Structure
```
results/p_sweep_20260129_HHMMSS/
├── p_sweep_results.csv           # Raw results data
├── summary.txt                    # Text summary
├── ari_vs_p_all_modes.png        # Main plot
├── ari_heatmap.png                # Heatmap visualization
├── ari_vs_p_individual.png       # Individual mode plots
├── mode_comparison_bars.png      # Bar chart comparison
└── std_vs_p.png                  # Variability plot
```

### CSV Format
```csv
p,q,mode,mu1,mu2,avg_ari,std_ari,n_runs
0.00,0.05,Graph+Label,0.0,0.1,0.7234,0.0412,10
0.00,0.05,Graph+Feature,0.5,0.0,0.6845,0.0523,10
...
```

## Customization

### Change Number of Runs
Edit [p_sweep_experiment.py](src/p_sweep_experiment.py):
```python
n_runs_per_mode = 20  # Line ~123 (change from 10 to 20)
```

### Change p Range
Edit [p_sweep_experiment.py](src/p_sweep_experiment.py):
```python
p_values = np.arange(0.0, 0.31, 0.01)  # Line ~120 (extend to 0.3)
```

### Modify Modes
Edit [p_sweep_experiment.py](src/p_sweep_experiment.py):
```python
modes = [
    {"name": "Graph+Label", "mu1": 0.0, "mu2": mu2_default},
    {"name": "Graph+Feature", "mu1": mu1_default, "mu2": 0.0},
    {"name": "Graph+Feature+Label", "mu1": mu1_default, "mu2": mu2_default},
    {"name": "Graph Only", "mu1": 0.0, "mu2": 0.0},
    {"name": "Custom Mode", "mu1": 0.5, "mu2": 0.5},  # Add custom modes
]
```

## Key Features

✅ **Progress tracking** with tqdm showing real-time ARI results  
✅ **Automated averaging** over multiple runs (default: 10)  
✅ **Same graph per p** - fair comparison across modes  
✅ **ARI computation** - handles cases with/without labels correctly  
✅ **CSV export** - easy to import into R, pandas, Excel  
✅ **Rich visualizations** - 5 different plot types in PNG and PDF  
✅ **Summary statistics** - mean, std, min, max ARI per mode  
✅ **Error handling** - graceful handling of failures  

## Understanding Results

### ARI (Adjusted Rand Index)
- **Range:** [0, 1]
- **1.0** = Perfect clustering
- **0.0** = Random clustering
- **Higher is better**

### Expected Patterns
1. Performance improves as `p` increases (stronger within-cluster connections)
2. "Graph+Feature+Label" should perform best (uses all information)
3. "Graph Only" may struggle without additional signals
4. Lower std_ari = more stable performance

### Example Summary
```
Mode                      Avg ARI         Std ARI        
-------------------------------------------------------
Graph+Label               0.8234          0.1234
Graph+Feature             0.7845          0.1456
Graph+Feature+Label       0.9123          0.0876
Graph Only                0.6543          0.1654
```

## Troubleshooting

### Import Errors
Scripts automatically handle directory setup. If issues persist:
```bash
cd /Users/javada/Desktop/CADO
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python src/p_sweep_experiment.py
```

### Out of Memory
Reduce number of runs or graph size in config.yml:
```yaml
samples_per_cluster: [300, 300, 300]  # Reduce from [500, 500, 500]
```

### Slow Execution
- Enable early stopping in config.yml (already enabled)
- Reduce algorithm.iters in config.yml
- Reduce number of p values tested

## Next Steps

1. ✅ Run test: `python src/test_p_sweep.py`
2. ✅ Run full experiment: `python src/p_sweep_experiment.py`
3. ✅ Generate plots: `python src/visualize_p_sweep.py results/p_sweep_*/`
4. 📊 Analyze CSV data with your preferred tools
5. 📝 Include results in your paper/report

## Questions?

- See [P_SWEEP_README.md](P_SWEEP_README.md) for detailed documentation
- Check individual file docstrings for implementation details
- Test completed successfully ✓

---

**Created:** 2026-01-29  
**Status:** Ready to use  
**Test Result:** ✓ All tests passed
