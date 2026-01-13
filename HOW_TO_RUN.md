# How to Run RVNP-SBI Experiments

**Last Updated**: January 13, 2026
**Status**: ✅ All systems verified and ready (FIXED: Config files, pipeline discovery, script paths - see PIPELINE_FIXES_SUMMARY.md)

---

## Quick Start (Google Colab)

### 1. Clone Repository

```bash
!git clone https://github.com/YOUR_USERNAME/RVNP_SBI.git
%cd RVNP_SBI

# IMPORTANT: All scripts must be run from the RVNP_SBI root directory!
# Do NOT cd into scripts/ subdirectory
```

### 2. Install Dependencies

```bash
!pip install -q jax jaxlib equinox optax numpyro pandas matplotlib seaborn flowjax paramax ml_collections absl-py
```

### 3. Run Experiments (Resumable!)

**IMPORTANT**: All commands must be run from the RVNP_SBI root directory (not from scripts/)

#### Option A: Run ALL Experiments (~60 experiments)

```bash
!bash scripts/run_experiments_colab.sh
```

#### Option B: Run Specific Task

```bash
!bash scripts/run_experiments_colab.sh --task=CS
```

#### Option C: Check Progress

```bash
!bash scripts/run_experiments_colab.sh --status
```

### 4. Generate Plots

```bash
!bash scripts/run_experiments_colab.sh --plots-only
```

---

## ✅ Verified: No Bad Fallbacks

### DReG Implementation

The DReG implementation has **NO try/except blocks or fallbacks**:

```python
# In src/losses.py, line 893:
if self.use_dreg:
    # DReG path - DIRECT call, no fallback!
    iwae_loss = _iwae_with_dreg(...)
else:
    # Standard IWAE path
    ...
```

**Verification**:
- ✅ Zero try/except in `_iwae_with_dreg()` function (lines 160-266)
- ✅ Clean if/else branching on `use_dreg` flag
- ✅ No silent fallback to standard IWAE

If DReG fails, **it will crash immediately** (by design!) rather than silently falling back.

---

## ✅ Verified: Colab Resumability

### How Resumability Works

The pipeline automatically saves state to `experiment_results/pipeline_state.json`:

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
  "last_updated": "2026-01-11 10:30:15"
}
```

### When Colab Times Out

**Just re-run the same command!**

```bash
!bash run_experiments_colab.sh
```

The pipeline will:
1. ✅ Load `pipeline_state.json`
2. ✅ Skip completed experiments
3. ✅ Resume from where it left off
4. ✅ Update metrics database incrementally

**No manual intervention needed!**

---

## Configuration Files

All config files now have the `use_dreg` parameter:

### Default Setting (Backward Compatible)

```python
# configs/default_flow.py
config.training.use_dreg = False  # Standard IWAE (default)
```

### Enable DReG

```python
# In any config file, or via command line:
config.training.use_dreg = True  # DReG for variance reduction
```

### Override from Command Line

```bash
python scripts/main_train_eval.py \
  --mode=train \
  --config=configs/CS_task/ranpt_100_mu_hybrid.py \
  --config.training.use_dreg=True \
  --config.training.n_iters=300
```

---

## Experiment Options

### 1. Main Experiments (with misspecification)

```bash
# All tasks (CS, SIR, Pendulum, Spectra)
!bash scripts/run_experiments_colab.sh

# Single task
!bash scripts/run_experiments_colab.sh --task=CS

# Check status
!bash scripts/run_experiments_colab.sh --status

# Generate plots only
!bash scripts/run_experiments_colab.sh --plots-only
```

**Output**:
- `experiment_results/metrics_database.csv`
- `experiment_results/pipeline_state.json`
- `publication_plots/*.png`

### 2. Well-Specified Experiments (no misspecification)

```bash
# All well-specified experiments
!bash scripts/run_wellspec_experiments.sh

# Single task
!bash scripts/run_wellspec_experiments.sh --task=SIR

# Check status
!bash scripts/run_wellspec_experiments.sh --status
```

**Output**:
- `experiment_results_wellspec/metrics_database.csv`
- `publication_plots_wellspec/*.png`

### 3. Ablation Study (CS task only)

```bash
# All ablations
!bash scripts/run_ablation_study.sh

# Parameter size only
!bash scripts/run_ablation_study.sh --params-only

# Shrinkage prior only
!bash scripts/run_ablation_study.sh --shrinkage-only

# Check status
!bash scripts/run_ablation_study.sh --status
```

**Output**:
- `experiment_results_ablation/ablation_metrics.csv`
- `publication_plots_ablation/ablation_study.png`

---

## Single Experiment (For Testing)

### Run One Experiment Directly

```bash
python scripts/integrated_pipeline.py \
  --task=CS \
  --method=RVNP-mu_hybrid \
  --nobs=100
```

### With DReG Enabled

```bash
python scripts/integrated_pipeline.py \
  --task=CS \
  --method=RVNP-mu_hybrid \
  --nobs=100 \
  --use_dreg
```

---

## Testing DReG

### Verify DReG Implementation

```bash
# Unit tests (mathematical properties)
python scripts/test_dreg_simple.py

# Integration test (with actual models)
python scripts/test_dreg_loss_only.py
```

**Expected Output**:
```
🎉 DReG IMPLEMENTATION IS WORKING!

Verified:
  - Squared normalized weights (w_k²)
  - Stopped gradients on weight computation
  - Separate objectives for model vs inference
  - All gradients are finite (no NaN)
```

---

## Monitoring Progress

### Check Pipeline State

```bash
cat experiment_results/pipeline_state.json | python -m json.tool
```

### Check Metrics Database

```bash
# View all results
cat experiment_results/metrics_database.csv

# Count completed experiments
wc -l experiment_results/metrics_database.csv
```

### Check Logs

```bash
# Real-time monitoring
tail -f experiment_results/pipeline.log

# Search for errors
grep -i "error" experiment_results/pipeline.log
```

---

## Troubleshooting

### Colab Times Out

**Solution**: Just re-run the command!

```bash
!bash scripts/run_experiments_colab.sh
```

The pipeline will resume automatically.

### Out of Memory

**Solution**: Run tasks one at a time:

```bash
!bash scripts/run_experiments_colab.sh --task=CS
# Wait for completion, then:
!bash scripts/run_experiments_colab.sh --task=SIR
```

### Missing Data

**Solution**: Data is auto-generated on first run.

```bash
# Verify data directory exists
ls -la Data/
```

### Import Errors

**Solution**: Ensure all dependencies are installed:

```bash
!pip install -q jax jaxlib equinox optax numpyro pandas matplotlib seaborn flowjax paramax ml_collections absl-py
```

---

## Expected Runtime

### Per Experiment (approximate)

- **CS task**: ~15-30 minutes
- **SIR task**: ~20-40 minutes
- **Pendulum task**: ~15-25 minutes
- **Spectra task**: ~30-60 minutes (has embeddings)

### Full Pipeline

- **Main experiments (~60)**: ~20-40 hours
- **Well-specified experiments (~60)**: ~20-40 hours
- **Ablation study (10)**: ~3-5 hours

**Tip**: Run in batches by task to fit within Colab time limits.

---

## Output Files

### Metrics Database

```
experiment_results/metrics_database.csv
```

Columns:
- `task`, `method`, `correction_type`, `nobs`
- `acauc_mean`, `acauc_std` - Calibration metric
- `lpp_median`, `lpp_std` - Log posterior probability
- `nrmse_mean`, `nrmse_std` - Parameter estimation error
- `ess_mean` - Effective sample size (SIR only)
- `use_sir` - Boolean flag for sampling method
- `timestamp` - Completion time

### Plots

```
publication_plots/
├── CS_comparison.png          # 4-panel comparison
├── SIR_comparison.png
├── Pendulum_comparison.png
├── Spectra_comparison.png
└── summary_table.csv          # LaTeX-ready table
```

### Checkpoints

```
output/{workspace_name}/
├── checkpoints/
│   ├── checkpoint_posterior_0.eqx
│   └── checkpoint_simulator_0.eqx
└── parameter_breakdown.json
```

---

## Advanced Usage

### Custom Experiment

```python
from scripts.integrated_pipeline import RVNPIntegratedPipeline

# Create pipeline
pipeline = RVNPIntegratedPipeline(results_dir='my_results')

# Generate experiments
experiments = pipeline.generate_experiment_list()

# Filter to specific experiments
cs_experiments = [e for e in experiments if e.task == 'CS' and e.nobs == 100]

# Run filtered experiments
for exp in cs_experiments:
    pipeline.run_single_experiment(exp)
    pipeline.run_evaluation(exp, use_sir=False)
    pipeline.run_evaluation(exp, use_sir=True)

# Generate plots
pipeline.generate_all_plots()
```

### Enable DReG for Specific Experiments

```python
# Modify config before running
import ml_collections

config = ml_collections.ConfigDict()
config.training = ml_collections.ConfigDict()
config.training.use_dreg = True  # Enable DReG

# Run experiment with modified config
# (see scripts/integrated_pipeline.py for full example)
```

---

## Summary

✅ **No fallbacks** - DReG crashes immediately if it fails  
✅ **Fully resumable** - Colab timeout? Just re-run!  
✅ **Config files ready** - `use_dreg` in all configs  
✅ **Tested and verified** - All tests passing  

**Ready for production experiments!**

---

## Quick Reference

**IMPORTANT**: All commands must be run from RVNP_SBI root directory

| Command | Purpose |
|---------|---------|
| `bash scripts/run_experiments_colab.sh` | Run all experiments |
| `bash scripts/run_experiments_colab.sh --task=CS` | Run single task |
| `bash scripts/run_experiments_colab.sh --status` | Check progress |
| `bash scripts/run_experiments_colab.sh --plots-only` | Generate plots |
| `python scripts/test_dreg_loss_only.py` | Test DReG |
| `cat experiment_results/pipeline_state.json` | View state |

---

**Questions?** Check the documentation:
- `PIPELINE_FIXES_SUMMARY.md` - **IMPORTANT: Jan 13 2026 fixes to configs and pipeline**
- `DREG_FINAL_STATUS.md` - DReG implementation details
- `CLAUDE.md` - Full project guide
- `IMPLEMENTATION_SUMMARY.md` - Complete feature overview

