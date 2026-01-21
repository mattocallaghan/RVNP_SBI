Getting Started
===============

Installation
------------

Prerequisites
~~~~~~~~~~~~~

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Julia 1.6+ (for SIR model)

Python Dependencies
~~~~~~~~~~~~~~~~~~~

Install all required Python packages::

    pip install -r requirements.txt

Julia Setup
~~~~~~~~~~~

For the SIR epidemiological model task::

    julia -e 'using Pkg; Pkg.add(["LinearAlgebra", "StochasticDiffEq", "NPZ", "ArgParse", "Random"])'

Data Directory
~~~~~~~~~~~~~~

Create the data directory for generated datasets::

    mkdir Data

Basic Usage
-----------

Training a Single Experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Train RVNP-NN on the Compressed Sensing task with 100 observations::

    python scripts/main_train_eval.py --config=configs/CS_task/ranpt_100_mu_hybrid.py --mode=train

Train NPE baseline::

    python scripts/main_train_eval.py --config=configs/CS_task/npe_cs_task.py --mode=train

Running All Experiments
~~~~~~~~~~~~~~~~~~~~~~~~

Run the complete experimental matrix (60 experiments)::

    bash run_experiments_colab.sh

Check experiment progress::

    bash run_experiments_colab.sh --status

Run experiments for a specific task only::

    bash run_experiments_colab.sh --task=CS

Generate publication plots::

    bash run_experiments_colab.sh --plots-only

Available Tasks
---------------

The codebase supports four simulation-based inference tasks:

1. **Compressed Sensing (CS)**: Spatial cancer cell distribution modeling

   - Parameter dimension: 3
   - Observation dimension: 4
   - Misspecification: Incorrect spatial dynamics

2. **SIR Model**: Epidemiological dynamics (requires Julia)

   - Parameter dimension: 2
   - Observation dimension: 10
   - Misspecification: Simplified transmission model

3. **Pendulum**: Harmonic oscillator dynamics

   - Parameter dimension: 3
   - Observation dimension: 10
   - Misspecification: Missing friction term

4. **Spectra**: Stellar spectroscopy (requires external data)

   - Parameter dimension: 3
   - Observation dimension: 300 (raw) → 5 (embedded)
   - Misspecification: Simplified spectral model
   - Embedding: Information Maximizing (IM) network

Methods
-------

RVNP-NN (Primary)
~~~~~~~~~~~~~~~~~~~~~~~~~

Neural mean and covariance correction::

    correction_type = 'NN'

Key features:

- Mean: :math:`\mu(x,\\theta) = x + \mu_{global} + \mu_\\theta(\\theta)`
- Covariance: :math:`\Sigma(\\theta) = L_{hybrid}(\\theta) L_{hybrid}(\\theta)^T`
- Shrinkage prior: :math:`\|\mu_\\theta(\\theta)\|^2` regularization

Use when: Significant parameter-dependent misspecification

RVNP-simple
~~~~~~~~~~~

Diagonal covariance correction::

    correction_type = 'simple'

Key features:

- Mean: :math:`\mu(x,\\theta) = x` (identity)
- Covariance: Fixed diagonal
- Simpler, faster training

Use when: Minimal misspecification, need faster inference

NPE (Baseline)
~~~~~~~~~~~~~~

Standard neural posterior estimation::

    model.name = 'npe'
    correction_type = 'none'

No correction model. Assumes well-specified simulator.

Configuration
-------------

Experiments are configured via Python config files in ``configs/``.

Example config structure for RVNP-NN::

    def get_config():
        config = base_get_config()

        # Dataset
        config.data.dataset = 'CS'
        config.data.num_simulations = 100000
        config.data.num_iid = 100  # Number of observations

        # Model
        config.model.name = 'nlpe_rqs_posterior'  # RVNP
        config.model.correction_type = 'NN'
        config.model.flow_dimension = 3  # Parameter dimension
        config.model.cond_dim = 4  # Observation dimension

        # Training
        config.training.n_iters = 300
        config.optim.lr = 1e-3

        return config

See ``CLAUDE.md`` for detailed configuration guide.

Next Steps
----------

- See :doc:`theory` for mathematical foundations
- See :doc:`api/index` for API reference
- See :doc:`guides/index` for advanced guides
- Read the `README <https://github.com/USERNAME/RVNP_SBI/blob/main/README.md>`_ for complete usage instructions
