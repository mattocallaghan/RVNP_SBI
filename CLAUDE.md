# RVNP-SBI Production Guide

**Last Updated**: January 13, 2026  
**Status**: ✅ Production Ready - All fixes verified

---

## Overview

**RVNP** (Robust Variational Neural Posterior) for Simulation-Based Inference under model misspecification.

**Core Idea**: Learn correction models to fix simulator errors when the simulator doesn't match reality.

**Experiments**:
- 4 Tasks: CS, SIR, Pendulum, Spectra
- 4 Methods: RVNP-simple, RVNP-NN, NPE, NNPE
- 5 Sample sizes (nobs): 1, 10, 100, 1000, 10000
- Total: 72 main + 44 wellspec + 10 ablation experiments

---

## Quick Start

### Run All Experiments (Google Colab or Local)

```bash
# Test everything works
bash scripts/quick_smoke_test.sh

# Main experiments (72 total)
bash scripts/run_experiments_colab.sh
bash scripts/run_experiments_colab.sh --task=CS      # Single task
bash scripts/run_experiments_colab.sh --status       # Check progress

# Well-specified experiments (44 total - no misspecification)
bash scripts/run_wellspec_experiments.sh

# Ablation study (10 experiments - CS task only)
bash scripts/run_ablation_study.sh
```

### Generate Plots

```bash
bash scripts/run_experiments_colab.sh --plots-only
bash scripts/run_wellspec_experiments.sh --plots-only
bash scripts/run_ablation_study.sh --plots-only
```

---

## Methods

### 1. RVNP-NN (mu_hybrid) ⭐ Main Method

**Config**: `correction_type='mu_hybrid'`  
**Files**: `ranpt_{nobs}_mu_hybrid.py`

**Correction Model**:
```python
# Neural mean shift + neural covariance
μ(x,θ) = x + μ_global + μ_θ(θ)     # Mean correction
Σ(θ) = L_hybrid(θ) L_hybrid(θ)ᵀ    # Neural covariance
```

**Shrinkage Prior**:
```python
R_shrink = (1/K) Σ_k ||μ_θ(θ_k)||²
Loss = -L_IWAE + λ_shrinkage * R_shrink
```

**Implementation**: `models/correction_model.py` (MuHybridCorrectionModel)

### 2. RVNP-simple

**Config**: `correction_type='simple'`  
**Files**: `ranpt_{nobs}_simple.py`

**Correction Model**:
```python
μ = x            # No neural mean
Σ = Diag(σ²)     # Fixed diagonal covariance
```

### 3. NPE (Baseline)

**Config**: `model.name='npe'`  
**Files**: `npe_{task}_task.py`

Standard Neural Posterior Estimation - no correction.

### 4. NNPE (Noisy Baseline)

**Config**: `model.name='nnpe'`  
**Files**: `nnpe_{task}_task.py`

Noisy NPE with corruption - alternative robustness approach.

---

## Metrics

### Primary Metric: ACAUC

**Average Coverage Area Under Curve** - continuous calibration error metric

```python
ACAUC = (1/d) Σⱼ ∫₀¹ |α - 𝟙[θⱼ* ∈ Cᵅʲ]| dα
```

- **Ideal**: 0.0 (perfect calibration)
- **< 0.05**: Excellent calibration
- **0.05 - 0.1**: Good calibration
- **> 0.1**: Poor calibration (significant deviation from ideal)

**Implementation**: `src/evaluate.py:290-346`

### Secondary Metrics

1. **LPP** - Log Posterior Probability  
   Higher = better

2. **NRMSE** - Normalized Root Mean Square Error  
   Lower = better (parameter estimation accuracy)

3. **ESS** - Effective Sample Size  
   Higher = better (SIR task only)

### Output

All metrics saved to:
- `experiment_results/metrics_database.csv` (main)
- `experiment_results_wellspec/metrics_database.csv` (wellspec)
- `experiment_results_ablation/ablation_metrics.csv` (ablation)

---

## Training Architecture

### Stage 1: Embedding (High-D tasks only)

**Tasks**: Spectra (300-D spectra → 5-D embedding)  
**Method**: InfoMax (mutual information maximization)  
**Output**: Trained embedding f_ω(x)

### Stage 2: Simulator Flow

**Input**: Pre-generated (θ, x) ~ p(θ)p(x|θ)  
**Trains**: Simulator p(x|θ)  
**Method**: Maximum likelihood  
**Reuse**: Trained once per task, shared across all experiments  
**Filename**: `Data/{task}_weights/simulator_nlpe_rqs_posterior_{task}_{num_sims}.eqx`

### Stage 3: Joint Posterior + Correction

**Input**: ONLY observed data x_obs (no pre-generated data)  
**Trains**: Posterior q_φ(θ|x̂) and correction r_ψ(x̂|x,θ) jointly

**Training Loop**:
```python
for batch in x_obs:
    # ALL sampling happens inside loss function:
    θ₁,...,θ_K ~ q_φ(θ|x_obs)              # Sample from current posterior
    x_sim_k ~ p(x|θ_k)                     # Sample from trained simulator
    
    # Compute IWAE loss
    L_IWAE = log(1/K Σ_k p(x_obs|x_sim_k,θ_k)p(θ_k)/q_φ(θ_k|x_obs))
    
    # Compute shrinkage (mu_hybrid only)
    R_shrink = (1/K) Σ_k ||μ_θ(θ_k)||²
    
    loss = -L_IWAE + λ_shrinkage * R_shrink
    
    # Update both posterior φ and correction ψ
    update(loss)
```

**Key Point**: No sampling in training loop - only x_obs passed to loss function.

---

## Tasks

### CS (Cancer/Stromal)

- **flow_dimension**: 3
- **cond_dim**: 4
- **Misspecification**: Incorrect dynamics in simulator

### SIR (Epidemic)

- **flow_dimension**: 2
- **cond_dim**: 10
- **Misspecification**: Simplified transmission model

### Pendulum

- **flow_dimension**: 3
- **cond_dim**: 10
- **Misspecification**: Missing friction term

### Spectra

- **flow_dimension**: 3
- **cond_dim**: 5
- **embedding**: 'IM' (Information Maximizing)
- **embedding_dim**: 5
- **vector_dim_inference**: 300 (raw spectra)
- **Misspecification**: Simplified spectral model
- **Note**: Limited to nobs=[1,10,100] (real data constraint)

---

## Configuration Files

### Main Experiments (72 total)

**Pattern**: `configs/{task}/ranpt_{nobs}_{correction}.py`

**RVNP-NN** (mu_hybrid):
- CS, SIR, Pendulum: nobs = 1, 10, 100, 1000, 10000 (15 each)
- Spectra: nobs = 1, 10, 100 (3)
- **Total**: 18 configs

**RVNP-simple**:
- Same as above
- **Total**: 18 configs

**NPE**: `configs/{task}/npe_{task}_task.py`
- **Total**: 4 configs

**NNPE**: `configs/{task}/nnpe_{task}_task.py`
- **Total**: 4 configs

### Well-Specified Experiments (44 total)

**Pattern**: `configs/{task}/{task}_task_tests{nobs}_{correction}_shrink00_wellspec.py`

- RVNP configs: All nobs values (18 each)
- NPE/NNPE: Only nobs=100 (4 each)

### Ablation Study (10 configs, CS task only, nobs=100)

**Parameter Size** (`configs/CS_task/ablation_params_{size}.py`):
- tiny, small, medium, large, huge
- nn_block_dim = [16, 32, 52, 128, 256]

**Shrinkage Prior** (`configs/CS_task/ablation_shrink_{level}.py`):
- 0, weak, medium, strong, verystrong
- lambda_shrinkage = [0.0, 0.01, 0.1, 1.0, 10.0]

### All Configs Use Correct Model Class

```python
config.model.name = 'nlpe_rqs_posterior'  # ✅ Trains from scratch
```

**Not**:
```python
config.model.name = 'ranpt'  # ❌ Requires pre-trained models
```

---

## Key Implementation Files

### Pipeline & Runners

- `scripts/integrated_pipeline.py` - Main training + evaluation + metrics
  - `METHODS` list (lines 61-66): Defines which methods to run
  - `_find_config_file()` (lines 171-205): Config discovery logic
- `scripts/run_experiments_colab.sh` - Main experiment runner
- `scripts/run_wellspec_experiments.sh` - Well-specified runner
- `scripts/run_ablation_study.sh` - Ablation runner
- `scripts/quick_smoke_test.sh` - Verification script

### Training & Models

- `scripts/main_train_eval.py` - Single experiment entry point
- `src/normalizing_flow.py` - Model classes
  - `Rational_Quadratic_Spline_w_posterior` (line 71): Correct class for training
  - `RANPT` (line 1502): Legacy class (requires pre-trained models)
- `models/correction_model.py` - Correction models
  - `MuHybridCorrectionModel` (line 954): Neural mean + cov
  - `SimpleCorrectionModel`: Diagonal cov
- `src/losses.py` - Loss functions
  - Shrinkage prior (lines 576-613)
  - IWAE loss computation

### Evaluation & Plotting

- `src/evaluate.py` - Metrics computation
  - `compute_acauc()` (lines 290-346): ACAUC metric
- `scripts/publication_plots.py` - 4-panel comparison plots
- `scripts/ablation_plots.py` - 2x2 ablation plots

---

## Output Structure

### Experiment Results

```
experiment_results/
├── metrics_database.csv          # All metrics (ACAUC, LPP, NRMSE, ESS)
└── pipeline_state.json           # Resumability state

experiment_results_wellspec/
└── metrics_database.csv

experiment_results_ablation/
└── ablation_metrics.csv
```

### Plots

```
publication_plots/
├── CS_comparison.png             # 4-panel: ACAUC, LPP, NRMSE, ESS vs nobs
├── SIR_comparison.png
├── Pendulum_comparison.png
├── Spectra_comparison.png
└── summary_table.csv             # LaTeX-ready table

publication_plots_wellspec/
└── (same structure)

publication_plots_ablation/
└── ablation_study.png            # 2x2: ACAUC/NRMSE vs params/shrinkage
```

### Checkpoints

```
output/{workspace_name}/
├── checkpoints/
│   ├── checkpoint_posterior_0.eqx
│   └── checkpoint_simulator_0.eqx
└── parameter_breakdown.json      # Component-wise parameter counts
```

---

## Resumability

### How It Works

Pipeline saves state to `experiment_results/pipeline_state.json`:

```json
{
  "experiments": [
    {
      "id": "CS_RVNP-NN_100",
      "training_complete": true,
      "evaluation_complete": true,
      "evaluation_sir_complete": false
    }
  ],
  "last_updated": "2026-01-13 15:00:00"
}
```

### When Colab Times Out

**Just re-run the same command!**

```bash
bash scripts/run_experiments_colab.sh
```

The pipeline will:
1. Load `pipeline_state.json`
2. Skip completed experiments
3. Resume from where it left off
4. Update metrics database incrementally

**No manual intervention needed.**

---

## Troubleshooting

### Colab Timeout

**Solution**: Re-run the command - fully resumable

```bash
bash scripts/run_experiments_colab.sh
```

### Out of Memory

**Solution**: Run tasks one at a time

```bash
bash scripts/run_experiments_colab.sh --task=CS
bash scripts/run_experiments_colab.sh --task=SIR
```

### Training Failures

**Check**: Config uses correct model class

```bash
grep "config.model.name" configs/CS_task/ranpt_100_mu_hybrid.py
# Should output: config.model.name = 'nlpe_rqs_posterior'
```

### Config Not Found

**Verify**: Smoke test passes

```bash
bash scripts/quick_smoke_test.sh
```

---

## Recent Fixes (January 13, 2026)

See `PIPELINE_FIXES_SUMMARY.md` for full details.

### Fix 1: Model Class (75 configs)

Changed all RVNP configs from `'ranpt'` → `'nlpe_rqs_posterior'`

### Fix 2: Method Naming

Updated METHODS list to use `'mu_hybrid'` instead of `'NN'` for RVNP-NN

### Fix 3: Script Paths

Fixed runner scripts to check for `scripts/integrated_pipeline.py`

### Fix 4: Streamlined Methods

Pipeline now finds ONLY 4 user-specified methods (removed legacy configs)

### Verification

```bash
bash scripts/quick_smoke_test.sh
# ✅ Main: 72 experiments (4 methods × tasks × nobs)
# ✅ Wellspec: 44 experiments
# ✅ All smoke tests passing
```

---

## Expected Runtime

### Per Experiment (approximate)

- CS: ~15-30 minutes
- SIR: ~20-40 minutes
- Pendulum: ~15-25 minutes
- Spectra: ~30-60 minutes (has embeddings)

### Full Pipelines

- Main (72): ~20-40 hours
- Wellspec (44): ~15-30 hours
- Ablation (10): ~3-5 hours

**Tip**: Run in batches by task to fit within Colab time limits.

---

## Code Reference

### Key Line Numbers

**Pipeline**:
- `scripts/integrated_pipeline.py:61-66` - METHODS list
- `scripts/integrated_pipeline.py:171-205` - Config discovery

**Training**:
- `src/losses.py:576-613` - Shrinkage prior
- `models/correction_model.py:954` - get_mean_shift_magnitude()

**Evaluation**:
- `src/evaluate.py:290-346` - ACAUC computation

**Plotting**:
- `scripts/publication_plots.py:21-26` - METHOD_STYLES

---

## Summary

✅ **All systems verified and production-ready**  
✅ **72 main + 44 wellspec + 10 ablation experiments configured**  
✅ **Simulator reuse working (trained once per task)**  
✅ **Full resumability on Colab timeouts**  
✅ **Clean configs (legacy files removed)**  
✅ **ACAUC metric implemented and tested**  
✅ **4 methods: RVNP-simple, RVNP-NN, NPE, NNPE**

**Ready to run production experiments!**

---

**Questions?** See:
- `PIPELINE_FIXES_SUMMARY.md` - Recent fixes (Jan 13, 2026)
- `HOW_TO_RUN.md` - Detailed usage guide
- `scripts/quick_smoke_test.sh` - Verify everything works
