Running Experiments
===================

This guide covers running the full experimental matrix and analyzing results.

Experimental Matrix
-------------------

The codebase is designed to run ~60 experiments organized as:

- **Tasks**: 4 (CS, SIR, Pendulum, Spectra)
- **Methods**: 3 (RVNP-simple, RVNP-NN, NPE)
- **Nobs**: 5 (1, 10, 100, 1000, 10000)

Main Experiments
----------------

Run all main experiments::

    bash run_experiments_colab.sh

Check progress::

    bash run_experiments_colab.sh --status

Run specific task::

    bash run_experiments_colab.sh --task=CS

Generate plots only::

    bash run_experiments_colab.sh --plots-only

Well-Specified Experiments
---------------------------

Test RVNP performance without misspecification::

    bash run_wellspec_experiments.sh

This validates that RVNP doesn't harm performance when the simulator is correct.

Ablation Study
--------------

Run ablation study (CS task only, Nobs=100)::

    bash run_ablation_study.sh

Studies two factors:

1. **Parameter Size**: Effect of ``nn_block_dim`` (16, 32, 52, 128, 256)
2. **Shrinkage Prior**: Effect of ``lambda_shrinkage`` (0.0, 0.01, 0.1, 1.0, 10.0)

Results
-------

Results are saved to:

- ``experiment_results/metrics_database.csv`` - All metrics
- ``publication_plots/`` - 4-panel comparison plots per task
- ``output/{workspace}/`` - Individual experiment checkpoints

Metrics in database:

- **ACAUC** - Primary calibration metric
- **AEPC** - Discrete calibration
- **LPP** - Log posterior probability
- **NRMSE** - Parameter estimation error
- **ESS** - Effective sample size (SIR only)
- **Training time** - Component-wise breakdown

Resumability
------------

All experiment runners are resumable via ``pipeline_state.json``.

If experiments are interrupted (e.g., Colab timeout), simply re-run::

    bash run_experiments_colab.sh

The pipeline will skip completed experiments and continue from where it stopped.

Troubleshooting
---------------

Out of Memory
~~~~~~~~~~~~~

Run by task instead of all at once::

    bash run_experiments_colab.sh --task=CS
    bash run_experiments_colab.sh --task=SIR
    bash run_experiments_colab.sh --task=Pendulum
    bash run_experiments_colab.sh --task=Spectra

Missing Data
~~~~~~~~~~~~

Data is auto-generated on first run::

    mkdir -p Data

Config Not Found
~~~~~~~~~~~~~~~~

Verify all configs exist::

    python scripts/verify_configs.py

For complete experiment configuration details, see the `README <https://github.com/USERNAME/RVNP_SBI/blob/main/README.md>`_.
