RVNP-SBI Documentation
======================

**Robust Variational Neural Posterior for Simulation-Based Inference**

This documentation covers the RVNP-SBI codebase, implementing robust neural variational posterior estimation under model misspecification using variational inference.

.. note::
   This is the API reference documentation. For quick start guides and experiment instructions, see:

   - `QUICKSTART.md <https://github.com/USERNAME/RVNP_SBI/blob/main/QUICKSTART.md>`_
   - `CLAUDE.md <https://github.com/USERNAME/RVNP_SBI/blob/main/CLAUDE.md>`_
   - `EXPERIMENTS_README.md <https://github.com/USERNAME/RVNP_SBI/blob/main/EXPERIMENTS_README.md>`_

Overview
--------

RVNP-SBI implements several simulation-based inference algorithms:

- **RVNP-mu_hybrid** (Primary): Neural mean + neural covariance correction
- **RVNP-simple**: Diagonal covariance correction
- **NPE**: Neural posterior estimation baseline
- **NNPE**: Noisy neural posterior estimation

The key innovation is the correction model that learns to adjust for model misspecification between the simulator and reality.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   api/index
   guides/index

Quick Start
-----------

Install dependencies::

    pip install -r requirements.txt

Run RVNP on the Compressed Sensing task::

    python main_train_eval.py --config=configs/CS_task/ranpt_100_mu_hybrid.py --mode=train

Or run all main experiments::

    bash run_experiments_colab.sh

Citation
--------

If you use this code in your research, please cite::

    @article{ocallaghan2025robust,
      title={Robust amortised simulation-based inference under model misspecification using variational inference},
      author={O'Callaghan, M and others},
      journal={ICML},
      year={2026}
    }

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
