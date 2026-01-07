# RVNP-SBI: Robust Neural Variational Posterior Estimation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4+-orange.svg)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository implements Robust Variational Neural Posterior estimation from the paper: "Robust amortised simulation-based inference under model misspecification using variational inference", O'Callaghan et al 2026

## Algorithms Implemented

This codebase implements several simulation-based inference algorithms for robust posterior estimation under model misspecification:

- **RVNP-mu_hybrid** - Primary method: Neural mean + neural covariance correction
- **RVNP-simple** - Baseline: Fixed diagonal covariance correction
- **NPE (Neural Posterior Estimation)** - Standard baseline (no correction)
- **NNPE (Noisy Neural Posterior Estimation)** - Noisy posterior baseline from Ward et al

**RVNP-mu_hybrid is the primary algorithm** of this repository, designed to handle significant model misspecification by learning both parameter-dependent mean shifts and adaptive covariance structures.

## Table of Contents
- [Citation](#citation)
- [Quick Start](#quick-start)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
- [RVNP Configuration](#rvnp-configuration)
- [Tasks and Simulators](#tasks-and-simulators)
- [Data Requirements](#data-requirements)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{o2025robust,
  title={Robust variational neural posterior estimation for simulation-based inference},
  author={O'Callaghan, Matthew and Mandel, Kaisey S and Gilmore, Gerry},
  journal={arXiv preprint arXiv:2509.05724},
  year={2025}
}
```

## Quick Start

**📚 Additional Documentation**: [QUICKSTART.md](QUICKSTART.md) | [CLAUDE.md](CLAUDE.md) | [EXPERIMENTS_README.md](EXPERIMENTS_README.md)

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Create Data Directory:**
   ```bash
   mkdir Data
   ```

3. **Install Julia (for SIR model):**
   ```bash
   # Install Julia and required packages
   julia -e 'using Pkg; Pkg.add(["LinearAlgebra", "StochasticDiffEq", "NPZ", "ArgParse"])'
   ```

4. **Run RVNP on CS Task:**
   ```bash
   python main_train_eval.py --config=configs/CS_task/cs_task.py --mode=train
   ```

## Directory Structure

```
RVNP_SBI/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── main_train_eval.py       # Main training script
├── run_lib.py               # Training pipeline implementation
├── datasets.py              # Dataset loading and generation
├── utils.py                 # Utility functions and task implementations
├── losses.py                # Loss function definitions
├── activation_functions.py  # Custom activation functions
├── normalizing_flow.py      # Normalizing flow implementations
│
├── configs/                 # Configuration files for different experiments
│   ├── default_flow.py      # Base configuration
│   ├── CS_task/            # Compressed Sensing task configs
│   ├── Gaussian_Mixture/   # Gaussian mixture model configs  ### not implemented in paper
│   ├── SIR/               # SIR epidemiological model configs
│   ├── Pendulum/          # Pendulum dynamics configs
│   ├── Spectra/           # Spectral analysis configs
│   └── gaussian_embedding/ # Gaussian embedding configs
│
├── models/                  # Neural network model implementations
│   ├── __init__.py
│   ├── correction_model.py  # Correction models for misspecification
│   ├── embeddings.py        # Embedding networks
│   ├── priors.py           # Prior distributions
│   
│
├── Julia/                   # Julia implementations
│   └── SIR.jl              # SIR model simulator in Julia
│
└── Data/                   # Data directory (create manually)
    └── [generated datasets will be stored here]
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Julia 1.6+ (for SIR model)

### Python Dependencies
Install all required Python packages:
```bash
pip install -r requirements.txt
```

### Julia Setup (for SIR model)
```bash
# Install Julia packages
julia -e 'using Pkg; Pkg.add(["LinearAlgebra", "StochasticDiffEq", "NPZ", "ArgParse", "Random"])'
```

## Usage

### Basic RVNP Training
```bash
python main_train_eval.py --config=path/to/config.py --mode=train
```

## RVNP Configuration

### Example: Compressed Sensing Task with RVNP

The RVNP algorithm uses a multi-stage training approach with variational posterior estimation and correction models. Here's a breakdown of key configuration parameters for the Compressed Sensing (CS) task:

```python
# configs/CS_task/cs_task.py

def get_config():
    config = base_get_config()
    
    # === DATASET CONFIGURATION ===
    config.data.dataset = 'CS'                                 # Training dataset (cancer cell simulation)
    config.data.inference_dataset = 'CS_inference'             # Inference dataset
    config.data.num_simulations = 100000                       # Training simulations
    config.data.num_tests = 1                                  # Test observations
    config.data.num_iid = 1                                    # IID samples per observation
    config.data.vector_dim = 4                                 # Summary statistics dimension
    
    # === MODEL ARCHITECTURE ===
    config.model.name = 'nlpe_rqs_posterior'                   # RVNP model type
    config.model.flow_dimension = 3                            # Parameter dimension 
    config.model.cond_dim = 4                                  # Conditioning dimension (summary stats)
    config.model.nn_depth_bnaf = 5                            # Neural network depth
    config.model.nn_block_dim = 52                            # Block dimension
    config.model.activation = 'tanh'                          # Activation function
    
    # === RVNP MULTI-STAGE TRAINING ===
    config.model.train_simulator = False                       # Stage 2: Skip simulator training for CS, should be true on first run

    
    # === CORRECTION MODEL (RVNP Core Feature) ===
    config.model.correction_type = 'simple'                    # Correction model type:
                                                                # 'simple' - Fixed diagonal covariance (RVNP-simple)
                                                                # 'mu_hybrid' - Neural mean + covariance (RVNP-mu_hybrid, PRIMARY)
                                                                # 'none' - No correction (NPE baseline)
    config.model.initial_correction_variance = -3              # Initial correction variance (log scale)
    
    # === VARIATIONAL TRAINING PARAMETERS ===
    config.model.lambda_variational = 1.0                      # Variational loss weight
    config.model.K_obs_samples = 30                            # Observation samples for training
    config.model.lambda_shrinkage = 0.0                        # Shrinkage toward no misspecification, legacy
    config.model.use_variational = True                        # Enable variational terms, always true
    config.model.variational_temperature = 1.0                 # Temperature for log p samples
    config.model.use_kl_term = True                            # KL divergence term
    config.model.lambda_kl = 1.0                               # KL term weight
    config.model.use_iw_elbo = True                            # Importance-weighted ELBO
    
    # === SIMULATOR SAMPLING (RVNP Enhancement) ===
    config.model.use_simulator_sampling = True                 # Fresh simulator sampling during correction
    config.model.simulator_samples_per_theta = 32              # Samples per parameter during correction
    config.model.use_reparam_contrastive = True                # Reparameterization sampling
    
    # === TRAINING SCHEDULE ===
    config.training.use_initialization = False                 # Skip initialization for CS task
    config.training.init_epochs = 15                           # Initialization epochs
    config.training.warmup_epochs = 5                          # Warmup stages (1-4.5)
    config.training.final_epochs = 0                           # Main training (stage 5)
    config.training.final_posterior_epochs = 0                 # Final posterior tuning (stage 6)
    
    # === OPTIMIZATION ===
    config.training.batch_size = 1024                          # Batch size
    config.training.n_iters = 300                              # Training iterations
    config.training.max_patience = 100                         # Early stopping patience
    config.optim.lr = 1e-3                                     # Learning rate
    config.optim.optimizer = 'Adam'                            # Optimizer
    config.optim.weight_decay = 1e-5                           # Weight decay
    config.optim.grad_clip = 10.0                              # Gradient clipping
    
    # === CS-SPECIFIC PARAMETERS ===
    config.model.clip_theta_to_bounds = True                   # Clip theta to training bounds
    config.model.amortized_training = True                     # Amortized training across observations
    
    return config
```

### Key RVNP Parameters Explained

**Multi-Stage Training:**
- RVNP uses a 5-stage approach to build robust posteriors
- Each stage focuses on different aspects: simulator, posterior, correction

**Correction Model:**
- Core innovation of RVNP for handling model misspecification
- `simple`: Fixed diagonal covariance (RVNP-simple baseline)
- `mu_hybrid`: Neural mean + covariance correction (PRIMARY method)
- Helps adjust for discrepancies between simulator and reality

**Variational Training:**
- `lambda_variational`: Controls strength of variational objective
- `variational_temperature`: Temperature for importance weighting
- `K_obs_samples`: Number of observation samples for robust training

**Simulator Sampling:**
- `use_simulator_sampling`: Enables fresh data generation during training
- `simulator_samples_per_theta`: Diversity of samples per parameter

### RVNP vs Other Methods

| Method | Mean Correction | Covariance Correction | Variational | Use Case |
|--------|----------------|----------------------|-------------|----------|
| NPE | ❌ | ❌ | ❌ | Well-specified simulator |
| NNPE | ❌ | ❌ | ✅ | Noisy observations |
| RVNP-simple | ❌ | ✅ (Fixed Diagonal) | ✅ | Minimal misspecification |
| RVNP-mu_hybrid | ✅ (Neural) | ✅ (Hybrid) | ✅ | Significant misspecification (PRIMARY) |

## Tasks and Simulators

### Built-in Simulators

1. **Gaussian Embedding** - Simple Gaussian data with feature extraction
2. **Gaussian Mixture** - Mixture of Gaussian distributions
3. **Compressed Sensing (CS)** - Spatial cancer cell distribution modeling
4. **SIR Model** - Epidemiological dynamics (requires Julia)
5. **Pendulum** - Simple harmonic oscillator dynamics

## Data Requirements

### Data Directory Setup
**Important:** Create a `Data/` directory in the project root:
```bash
mkdir Data
```

This directory will store:
- Generated simulation datasets
- Cached statistics (means, standard deviations)
- Preprocessed data files

### Spectra Task - External Data Required
The spectra task requires externally-prepared data files:
- `Data/isochrone_informed_data_raw_col.npz` (for training)
- `Data/bprp_spectra_data.npz` (for inference)

These files should contain:
- `features`: Stellar parameters (age, metallicity, etc.)
- `fluxes`: Spectral flux measurements

**Note:** These data files are not included in the repository and must be prepared separately for stellar spectroscopy applications.

### SIR Model - Julia Environment
The SIR model requires:
- Julia installation with required packages
- Proper Julia environment path configuration
- Write permissions for temporary files during simulation

---

 

