RVNP-SBI Documentation
======================

**Robust Variational Neural Posterior for Simulation-Based Inference**

This documentation covers the RVNP-SBI codebase, implementing robust neural variational posterior estimation under model misspecification using variational inference.

.. note::
   **Documentation Navigation**:

   - **Theory & Background**: See :doc:`theory` for mathematical foundations
   - **Quick Start & Usage**: See the `README <https://github.com/USERNAME/RVNP_SBI/blob/main/README.md>`_
   - **API Reference**: Browse the modules below for detailed class and function documentation

Overview
--------

RVNP-SBI implements several simulation-based inference algorithms:

- **RVNP-NN** (Primary): Neural mean + neural covariance correction
- **RVNP-simple**: Diagonal covariance correction
- **NPE**: Neural posterior estimation baseline
- **NNPE**: Noisy neural posterior estimation

The key innovation is the correction model that learns to adjust for model misspecification between the simulator and reality.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   theory
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

If you use this code in your research, please cite:

.. code-block:: bibtex

    @article{o2025robust,
      title={Robust variational neural posterior estimation for simulation-based inference},
      author={O'Callaghan, Matthew and Mandel, Kaisey S and Gilmore, Gerry},
      journal={arXiv preprint arXiv:2509.05724},
      year={2025}
    }

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
