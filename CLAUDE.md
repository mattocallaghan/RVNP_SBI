# RVNP-SBI - ICML 2026 Project Guide

**Last Updated**: January 6, 2026
**Status**: ✅ All implementation complete. Ready for experiments.

---

## What This Project Does

Robust Variational Neural Posterior (RVNP) for Simulation-Based Inference under model misspecification.

- **Goal**: Learn correction models to fix simulator errors
- **Baseline**: Neural Posterior Estimation (NPE)
- **Tasks**: CS, SIR, Pendulum, Spectra (4 tasks)
- **Nobs**: 1, 10, 100, 1000, 10000 (5 settings)
- **Methods**: RVNP-simple, RVNP-mu_hybrid, NPE (3 methods)

---

## Quick Start

### Run All Main Experiments (~60 total)
```bash
bash run_experiments_colab.sh              # Run everything
bash run_experiments_colab.sh --status     # Check progress
bash run_experiments_colab.sh --task=CS    # Run only CS task
bash run_experiments_colab.sh --plots-only # Generate plots
```

### Run Well-Specified Experiments
```bash
bash run_wellspec_experiments.sh           # No misspecification tests
bash run_wellspec_experiments.sh --status
```

### Run Ablation Study (CS task only)
```bash
bash run_ablation_study.sh                 # All ablations
bash run_ablation_study.sh --params-only   # Parameter size only
bash run_ablation_study.sh --shrinkage-only # Shrinkage prior only
bash run_ablation_study.sh --plots-only
```

### Single Experiment
```bash
python integrated_pipeline.py --task=SIR --method=RVNP-mu_hybrid --nobs=100
```

---

## Methods

### 1. RVNP-mu_hybrid ⭐ (Main Method)
- **Type**: Neural mean + neural covariance correction
- **Config**: `correction_type='mu_hybrid'`
- **Mean**: μ = x + μ_global + μ_θ(θ) where μ_θ is neural network
- **Shrinkage Prior**: L2 penalty on ||μ_θ(θ)||² (controlled by `lambda_shrinkage`)
- **Implementation**: `models/correction_model.py` (MuHybridCorrectionModel)

### 2. RVNP-simple
- **Type**: Diagonal covariance correction
- **Config**: `correction_type='simple'`
- **Mean**: μ = x (no neural mean)

### 3. NPE (Baseline)
- **Type**: No correction
- **Config**: `model.name='npe'`, `correction_type='none'`

**⚠️ DO NOT confuse**:
- `correction_type='hybrid'` = Neural covariance ONLY (deprecated) ❌
- `correction_type='mu_hybrid'` = Neural mean + covariance ✅

---

## Configs Available

### Main Task Configs
- ✅ **CS**: `configs/CS_task/ranpt_{1,10,100,1000,10000}_mu_hybrid.py`
- ✅ **CS**: `configs/CS_task/ranpt_{1,10,100,1000,10000}_simple.py`
- ✅ **CS**: `configs/CS_task/npe_cs_task.py`
- ✅ **SIR**: `configs/SIR/ranpt_{1,10,100,1000,10000}_{mu_hybrid,simple}.py`
- ✅ **SIR**: `configs/SIR/npe_sir_task.py`
- ✅ **Pendulum**: `configs/Pendulum/ranpt_{1,10,100,1000,10000}_{mu_hybrid,simple}.py`
- ✅ **Pendulum**: `configs/Pendulum/npe_pendulum_task.py`
- ✅ **Spectra**: `configs/Spectra/ranpt_{1,10,100}_mu_hybrid.py`
- ✅ **Spectra**: `configs/Spectra/npe_spectra_task.py`

### Well-Specified Configs
- Pattern: `configs/{task}/*_wellspec.py`
- Used by: `run_wellspec_experiments.sh`

### Ablation Study Configs (CS only, Nobs=100)
**Parameter Size**:
- `configs/CS_task/ablation_params_{tiny,small,medium,large,huge}.py`
- nn_block_dim = [16, 32, 52, 128, 256]

**Shrinkage Prior**:
- `configs/CS_task/ablation_shrink_{0,weak,medium,strong,verystrong}.py`
- lambda_shrinkage = [0.0, 0.01, 0.1, 1.0, 10.0]

---

## Metrics

1. **ACAUC** (Primary) - Average Coverage Area Under Curve
   - Continuous calibration metric
   - Ideal value: 1.0 (perfect calibration)
   - < 1.0: Under-covered (overconfident)
   - > 1.0: Over-covered (too conservative)

2. **AEPC** - Average Expected Posterior Coverage
   - Discrete calibration at specific α levels

3. **LPP** - Log Posterior Probability
   - Higher = better

4. **NRMSE** - Normalized Root Mean Square Error
   - Lower = better

5. **ESS** - Effective Sample Size
   - Higher = better

6. **Training Time** - Component-wise breakdown
   - Tracked via `time_tracker.py`

---

## Key Files

### Pipeline
- `integrated_pipeline.py` - Main training + evaluation + metrics pipeline
- `run_experiments_colab.sh` - Main experiment runner (resumable)
- `run_wellspec_experiments.sh` - Well-specified experiments
- `run_ablation_study.sh` - Ablation study runner
- `publication_plots.py` - Generate 4-panel comparison plots
- `ablation_plots.py` - Generate 2x2 ablation plots

### Training & Models
- `main_train_eval.py` - Single experiment entry point
- `normalizing_flow.py` - RANPT model implementation
- `models/correction_model.py` - Correction models (mu_hybrid at line 954)
- `losses.py` - Loss functions including shrinkage prior (lines 576-613)
- `evaluate.py` - Evaluation including ACAUC (lines 290-346)

### Utilities
- `time_tracker.py` - Training time tracking
- `model_utils.py` - Parameter counting
- `verify_configs.py` - Config validation

### Data
- `data_utils.py` - Data loading
- `Data/` - Auto-generated datasets

---

## Architecture Details

### RVNP Training Stages
1-2. Train simulator flow p(x_sim|θ)
3. Initialize posterior p_φ(θ|x_obs)
4. Coarse correction training
4.5. Widen posterior
5. Joint refinement (posterior + correction)
6. Final posterior tuning

### Shrinkage Prior (mu_hybrid only)
```python
# In losses.py
mean_shift_magnitudes = vmap(correction_model.get_mean_shift_magnitude)(theta_sim)
mean_shift_penalty = jnp.mean(mean_shift_magnitudes)  # ||μ_θ(θ)||²
shrinkage_loss = diagonal_penalty + mean_shift_penalty
total_loss += lambda_shrinkage * shrinkage_loss
```

### Parameter Counts
Models automatically log parameter breakdown:
- Posterior flow
- Simulator flow
- Correction model
- Embedding (if applicable)

Saved to: `{workspace}/parameter_breakdown.json`

---

## Output Directories

### Main Experiments
```
experiment_results/
  ├── metrics_database.csv       # All metrics (ACAUC, LPP, NRMSE, etc.)
  └── pipeline_state.json        # Resumability state

publication_plots/
  ├── CS_comparison.png          # 4-panel plot per task
  ├── SIR_comparison.png
  ├── Pendulum_comparison.png
  ├── Spectra_comparison.png
  └── summary_table.csv

output/
  └── {workspace_name}/          # Per-experiment checkpoints & logs
      ├── parameter_breakdown.json
      └── checkpoints/
```

### Well-Specified Experiments
```
experiment_results_wellspec/
  └── metrics_database.csv

publication_plots_wellspec/
  └── *.png
```

### Ablation Study
```
experiment_results_ablation/
  └── ablation_metrics.csv

publication_plots_ablation/
  └── ablation_study.png         # 2x2 plot
```

---

## Common Issues & Solutions

### Out of Memory
```bash
# Run by task instead of all at once
bash run_experiments_colab.sh --task=CS
```

### Colab Timeout
```bash
# Just re-run - resumable via pipeline_state.json
bash run_experiments_colab.sh
```

### Missing Data
```bash
# Auto-generated on first run
mkdir -p Data
```

### Config Not Found
```bash
# Verify configs exist
python verify_configs.py
```

### Check Progress
```bash
bash run_experiments_colab.sh --status
bash run_wellspec_experiments.sh --status
bash run_ablation_study.sh --status
```

---

## Task-Specific Parameters

### CS Task
- flow_dimension: 3
- cond_dim: 4
- embedding: None
- Misspecification: Simulator has incorrect dynamics

### SIR Task
- flow_dimension: 2
- cond_dim: 10
- embedding: None
- Misspecification: Simplified transmission model

### Pendulum Task
- flow_dimension: 3
- cond_dim: 10
- embedding: None
- Misspecification: Missing friction term

### Spectra Task
- flow_dimension: 3
- cond_dim: 5
- embedding: 'IM' (Information Maximizing)
- embedding_dim: 5
- vector_dim_inference: 300 (raw spectra)
- Misspecification: Simplified spectral model

---

## Experimental Matrix

### Main Experiments
- **Tasks**: 4 (CS, SIR, Pendulum, Spectra)
- **Methods**: 3 (RVNP-simple, RVNP-mu_hybrid, NPE)
- **Nobs**: 5 (1, 10, 100, 1000, 10000)
- **Total**: ~60 experiments

### Well-Specified Experiments
- Same matrix as main, but with correct simulator
- Purpose: Validate RVNP doesn't harm performance

### Ablation Study
- **Task**: CS only
- **Nobs**: 100 only
- **Ablations**: 10 (5 parameter sizes + 5 shrinkage strengths)

---

## Time Tracking

The `time_tracker.py` module tracks component-wise training times:

```python
from time_tracker import TrainingTimeTracker

tracker = TrainingTimeTracker()
tracker.start_stage("simulator_flow", "Training simulator p(x|θ)")
# ... training code ...
tracker.end_stage("simulator_flow")

tracker.print_summary()
tracker.save_to_json("workspace/training_time.json")
```

**Key Insight**: Simulator flow is trained ONCE and reused by all components, so it's counted only once in total time.

---

## Plotting

### Main Comparison Plots (4-panel)
```bash
bash run_experiments_colab.sh --plots-only
```

Panels:
1. ACAUC vs Nobs (calibration)
2. LPP vs Nobs (likelihood quality)
3. NRMSE vs Nobs (parameter estimation)
4. ESS vs Nobs (sample efficiency, SIR only)

### Ablation Plots (2x2)
```bash
bash run_ablation_study.sh --plots-only
```

Panels:
1. ACAUC vs nn_block_dim
2. NRMSE vs nn_block_dim
3. ACAUC vs lambda_shrinkage
4. NRMSE vs lambda_shrinkage

---

## Code Locations

### Key Implementation Lines
- `models/correction_model.py:954` - `get_mean_shift_magnitude()` for mu_hybrid
- `losses.py:576-613` - Shrinkage prior computation
- `evaluate.py:290-346` - ACAUC metric computation
- `normalizing_flow.py:1993-2011` - Parameter count logging
- `integrated_pipeline.py:156-200` - Config file discovery

### Model Classes
- `SimpleCorrectionModel` - Diagonal covariance, mean=x
- `HybridCorrectionModel` - Neural covariance, mean=x (deprecated)
- `MuHybridCorrectionModel` - Neural mean + neural covariance ⭐
- `FullNeuralCorrectionModel` - Fully neural (experimental)

---

## Development & Testing

### Verify All Configs
```bash
python verify_configs.py
```

### Test ACAUC Implementation
```bash
python test_acauc.py
```

### Test Time Tracker
```bash
python time_tracker.py
```

### Load Specific Config
```python
from configs.CS_task.ranpt_100_mu_hybrid import get_config
config = get_config()
print(config.model.correction_type)  # Should print: mu_hybrid
```

---

## Implementation Status

### ✅ Completed (January 6, 2026)

1. **ACAUC Metric**
   - Continuous calibration metric
   - 100 alpha levels, 10,000 samples
   - Integrated into pipeline and plots

2. **Parameter Counting**
   - Component-wise breakdown
   - JSON output per workspace
   - Logged during training

3. **Config Files**
   - All main task configs created
   - 10 ablation configs for CS
   - All verified and tested

4. **Time Tracking System**
   - `time_tracker.py` implemented
   - Ready for integration into training loop

5. **Well-Specified Experiments**
   - Runner script created
   - Pipeline support added
   - Uses existing wellspec configs

6. **Ablation Study**
   - 10 configs created
   - Runner script created
   - Plotting script created

7. **Documentation**
   - This file (CLAUDE.md)
   - SYSTEM_OVERVIEW.txt
   - IMPLEMENTATION_PLAN.md
   - SESSION_COMPLETION_SUMMARY.md

---

## Next Steps

1. **Run Main Experiments**:
   ```bash
   bash run_experiments_colab.sh
   ```

2. **Run Well-Specified Experiments**:
   ```bash
   bash run_wellspec_experiments.sh
   ```

3. **Run Ablation Study**:
   ```bash
   bash run_ablation_study.sh
   ```

4. **Generate All Plots**:
   ```bash
   bash run_experiments_colab.sh --plots-only
   bash run_wellspec_experiments.sh --plots-only
   bash run_ablation_study.sh --plots-only
   ```

5. **Analyze Results**:
   - `experiment_results/metrics_database.csv`
   - `experiment_results_wellspec/metrics_database.csv`
   - `experiment_results_ablation/ablation_metrics.csv`

---

## Contact & References

**Project**: RVNP-SBI for ICML 2026
**Framework**: JAX + Equinox
**Key Dependencies**: jax, equinox, optax, numpyro, pandas, matplotlib, seaborn

**Repository Structure**: See `SYSTEM_OVERVIEW.txt` for detailed architecture

---

**Status**: ✅ ALL FEATURES IMPLEMENTED - READY FOR EXPERIMENTS
