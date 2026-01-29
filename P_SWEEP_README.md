# P-Sweep Experiment Documentation

This experiment sweeps the SBM parameter `p` (within-cluster connection probability) from 0 to 0.2 with a step size of 0.01, testing CADO's performance under four different modes.

## Modes

1. **Graph + Label** (mu1=0, mu2=mu2_default)
   - Uses graph structure and label information
   - Feature information is disabled (mu1=0)

2. **Graph + Feature** (mu1=mu1_default, mu2=0)
   - Uses graph structure and feature information
   - Label information is disabled (mu2=0)
   - ARI computed from assignment matrix E

3. **Graph + Feature + Label** (mu1=mu1_default, mu2=mu2_default)
   - Uses all three sources of information
   - Full CADO mode

4. **Graph Only** (mu1=0, mu2=0)
   - Only uses graph structure
   - Equivalent to subspace clustering
   - ARI computed from assignment matrix E

## Running the Experiment

### Basic Usage

```bash
cd src
python p_sweep_experiment.py
```

This will:
- Sweep p from 0.0 to 0.2 with step 0.01 (21 values)
- Run each of the 4 modes
- Repeat each configuration 10 times
- Save results to `results/p_sweep_YYYYMMDD_HHMMSS/`

### Total Runs

- P values: 21 (from 0.0 to 0.2 with step 0.01)
- Modes: 4
- Repetitions per configuration: 10
- **Total CADO runs: 21 × 4 × 10 = 840**

### Progress Tracking

The script uses `tqdm` to show:
- Current p value being processed
- Current mode being tested
- Average ARI and standard deviation for each mode

## Output Files

The experiment creates a timestamped directory with:

### 1. `p_sweep_results.csv`
CSV file containing all results with columns:
- `p`: Within-cluster connection probability
- `q`: Between-cluster connection probability
- `mode`: Mode name
- `mu1`: Feature weight parameter
- `mu2`: Label weight parameter
- `avg_ari`: Average ARI across runs
- `std_ari`: Standard deviation of ARI
- `n_runs`: Number of runs averaged

### 2. `summary.txt`
Text summary with:
- Experimental configuration
- Mode descriptions
- Summary statistics (mean ARI per mode)

## Visualizing Results

After the experiment completes, visualize the results:

```bash
python visualize_p_sweep.py results/p_sweep_YYYYMMDD_HHMMSS/
```

Or specify the CSV directly:

```bash
python visualize_p_sweep.py results/p_sweep_YYYYMMDD_HHMMSS/p_sweep_results.csv
```

### Generated Plots

1. **ari_vs_p_all_modes.png** - Main plot showing ARI vs p for all modes with error bands
2. **ari_heatmap.png** - Heatmap of ARI values across modes and p values
3. **ari_vs_p_individual.png** - Individual plots for each mode with error bars
4. **mode_comparison_bars.png** - Bar chart comparing modes at selected p values
5. **std_vs_p.png** - Standard deviation plot showing variability

All plots are saved in both PNG (high resolution) and PDF formats.

## Configuration

The experiment uses parameters from `config/config.yml`:
- `samples_per_cluster`: Number of samples per cluster
- `num_features`: Number of features
- `train_ratio`: Ratio of training samples
- `q`: Between-cluster connection probability
- `mu1`, `mu2`: Default weight parameters
- `gmm_center_min`, `gmm_min_separation`, `gmm_sigma`: GMM parameters

## Customization

### Changing the Number of Runs

Edit `p_sweep_experiment.py`:
```python
n_runs_per_mode = 20  # Change from 10 to 20
```

### Changing the p Range

Edit `p_sweep_experiment.py`:
```python
p_values = np.arange(0.0, 0.31, 0.01)  # Now sweeps from 0 to 0.3
```

### Adding More Modes

Edit the `modes` list in `p_sweep_experiment.py`:
```python
modes = [
    # Existing modes...
    {"name": "Custom Mode", "mu1": 0.5, "mu2": 0.5},
]
```

## Expected Runtime

With default settings:
- CADO iterations: 100 (from config)
- Total runs: 840
- Estimated time per run: ~5-10 seconds
- **Total estimated runtime: ~70-140 minutes**

The actual runtime depends on:
- Early stopping (convergence detection)
- Hardware specifications
- Graph size and complexity

## Understanding the Results

### ARI (Adjusted Rand Index)
- Range: [0, 1]
- 1.0 = Perfect clustering
- 0.0 = Random clustering
- Higher is better

### Expected Patterns

1. **Graph Only** should perform worse as it lacks feature/label information
2. **Graph + Feature + Label** should perform best as it uses all information
3. Performance should generally improve as `p` increases (stronger within-cluster connections)
4. Standard deviation shows the stability of the algorithm across runs

## Troubleshooting

### Out of Memory
Reduce the number of runs or samples:
```python
n_runs_per_mode = 5  # Reduce from 10
```

### Slow Execution
- Check if early stopping is enabled in config
- Reduce `algorithm.iters` in config.yml
- Use fewer p values

### Import Errors
Make sure you're in the `src` directory:
```bash
cd src
python p_sweep_experiment.py
```

## Example Output

```
Running experiment with 21 p values and 4 modes
Each configuration will be run 10 times
Total runs: 840

Sweeping p values: 100%|████████████| 21/21 [01:23<00:00, 3.98s/it, p=0.200, mode=Graph Only, ARI=0.8234±0.0312]

Results saved to: results/p_sweep_20260129_120000/p_sweep_results.csv

================================================================================
EXPERIMENT SUMMARY
================================================================================

Configuration:
  - p values: 0.00 to 0.20 (step: 0.01)
  - q value: 0.05
  - Samples per cluster: [500, 500, 500]
  - Number of features: 6
  - Runs per configuration: 10

Modes tested:
  - Graph+Label: mu1=0.0, mu2=0.1
  - Graph+Feature: mu1=0.5, mu2=0.0
  - Graph+Feature+Label: mu1=0.5, mu2=0.1
  - Graph Only: mu1=0.0, mu2=0.0

Mode                      Avg ARI         Std ARI        
-------------------------------------------------------
Graph+Label               0.8234          0.1234
Graph+Feature             0.7845          0.1456
Graph+Feature+Label       0.9123          0.0876
Graph Only                0.6543          0.1654

================================================================================
```

## Citation

If you use this experiment in your research, please cite the CADO paper and include the experimental parameters used.
