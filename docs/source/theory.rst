Mathematical Theory
===================

This page provides the mathematical foundation for RVNP (Robust Variational Neural Posterior) estimation.

Problem Setup
-------------

Simulation-Based Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In simulation-based inference (SBI), we have:

- Parameters of interest: :math:`\theta \in \Theta \subseteq \mathbb{R}^d`
- Observations: :math:`x \in \mathcal{X} \subseteq \mathbb{R}^p`
- Simulator: :math:`x \sim p(x|\theta)` (can generate data given parameters)
- Prior: :math:`p(\theta)`
- Goal: Infer posterior :math:`p(\theta|x_{\text{obs}})`

The challenge: We can sample from :math:`p(x|\theta)` but cannot evaluate its density.

Model Misspecification
~~~~~~~~~~~~~~~~~~~~~~~

**Key Problem**: The simulator :math:`p_{\text{sim}}(x|\theta)` may differ from the true data-generating process :math:`p_{\text{true}}(x|\theta)`.

This leads to:

- Biased posterior estimates
- Poor calibration (under/over-coverage)
- Incorrect scientific conclusions

**Examples of Misspecification**:

- Missing friction term in physical simulator
- Simplified transmission dynamics in epidemiological models
- Incorrect spatial dynamics in cell distribution models

Standard Methods
----------------

Neural Posterior Estimation (NPE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Standard NPE learns an amortized posterior :math:`q_\phi(\theta|x)` by maximizing:

.. math::

    \mathcal{L}_{\text{NPE}} = \mathbb{E}_{\theta \sim p(\theta)} \mathbb{E}_{x \sim p_{\text{sim}}(x|\theta)} [\log q_\phi(\theta|x)]

**Assumption**: :math:`p_{\text{sim}}(x|\theta) = p_{\text{true}}(x|\theta)` (no misspecification)

**Failure Mode**: When misspecified, NPE learns :math:`q_\phi(\theta|x)` that matches the *wrong* simulator, leading to biased inference on real observations.

RVNP Approach
-------------

Correction Model
~~~~~~~~~~~~~~~~

RVNP introduces a **correction model** :math:`r_\psi(\hat{x}|x,\theta)` that transforms simulator outputs to match real observations:

.. math::

    r_\psi(\hat{x}|x,\theta) = \mathcal{N}(\hat{x}; \mu_\psi(x,\theta), \Sigma_\psi(\theta))

where:

- :math:`x`: Original simulator output
- :math:`\hat{x}`: Corrected observation
- :math:`\psi`: Correction model parameters

RVNP-simple (Baseline)
^^^^^^^^^^^^^^^^^^^^^^^

Fixed diagonal covariance correction:

.. math::

    \mu_\psi(x,\theta) &= x \\
    \Sigma_\psi(\theta) &= \text{diag}(\sigma_1^2, \ldots, \sigma_p^2)

**Use when**: Minimal misspecification, only need variance adjustment

RVNP-NN (Neural Network, Primary)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Neural mean and neural covariance correction:

.. math::

    \mu_\psi(x,\theta) &= x + \mu_{\text{global}} + \mu_\theta(\theta) \\
    \Sigma_\psi(\theta) &= L_{\text{hybrid}}(\theta) L_{\text{hybrid}}(\theta)^T \\
    L_{\text{hybrid}}(\theta) &= L_{\text{global}} + \text{diag}(\sigma_{\text{local}}(\theta))

where:

- :math:`\mu_{\text{global}} \in \mathbb{R}^p`: Learnable global bias
- :math:`\mu_\theta: \Theta \to \mathbb{R}^p`: Neural network for parameter-dependent bias
- :math:`L_{\text{global}} \in \mathbb{R}^{p \times p}`: Global Cholesky factor (lower triangular)
- :math:`\sigma_{\text{local}}: \Theta \to \mathbb{R}^p`: Neural network for local scaling

**Use when**: Significant parameter-dependent misspecification

Variational Objective
~~~~~~~~~~~~~~~~~~~~~

RVNP jointly trains the posterior :math:`q_\phi(\theta|\hat{x})` and correction :math:`r_\psi(\hat{x}|x,\theta)` by minimizing:

.. math::

    \mathcal{L}(\phi,\psi) = -\mathbb{E}_\theta \mathbb{E}_{x \sim p_{\text{sim}}(x|\theta)} \mathbb{E}_{\hat{x} \sim r_\psi(\hat{x}|x,\theta)} [\log q_\phi(\theta|\hat{x})]
    + \lambda_{\text{KL}} \cdot \text{KL}(r_\psi(\hat{x}|x,\theta) \| p_{\text{sim}}(x|\theta))
    + \lambda_{\text{shrinkage}} \cdot \mathcal{R}_{\text{shrink}}(\psi)

**Loss Components**:

1. **Posterior NLL** (first term):

   - Trains posterior to assign high probability to true :math:`\theta` given corrected observations
   - Averaged over :math:`\theta`, simulator outputs :math:`x`, and corrected samples :math:`\hat{x}`

2. **KL Divergence** (second term):

   - Prevents correction from deviating too far from simulator
   - Ensures corrected distribution remains plausible
   - Weighted by :math:`\lambda_{\text{KL}}`

3. **Shrinkage Prior** (third term):

   .. math::

       \mathcal{R}_{\text{shrink}}(\psi) = \mathbb{E}_\theta[\|\mu_\theta(\theta)\|^2]

   - Regularizes neural mean shift toward zero
   - Prevents overfitting when mean misspecification is minimal
   - Weighted by :math:`\lambda_{\text{shrinkage}}`
   - **Important**: Only penalizes the mean neural network output, NOT the covariance

Multi-Stage Training
~~~~~~~~~~~~~~~~~~~~

RVNP uses a multi-stage training pipeline:

**Stage 1**: Train embedding networks (if high-dimensional observations)

**Stage 2**: Train simulator flow :math:`p_{\text{sim}}(x|\theta)` via maximum likelihood

**Stage 3**: Initialize posterior :math:`q_\phi(\theta|x)` (optional)

**Stage 4**: Joint training of :math:`q_\phi` and :math:`r_\psi` using full loss

This staged approach ensures stable learning and prevents collapse.

Calibration Metrics
-------------------

ACAUC (Primary)
~~~~~~~~~~~~~~~

**Average Coverage Area Under Curve** measures continuous calibration:

.. math::

    \text{ACAUC} = \frac{1}{d} \sum_{j=1}^{d} \int_{0}^{1} \mathbb{1}[\theta_j^* \in C_\alpha^j] \, d\alpha

where:

- :math:`d`: Parameter dimension
- :math:`\theta_j^*`: True value of parameter :math:`j`
- :math:`C_\alpha^j`: :math:`\alpha`-level credible interval for dimension :math:`j`
- :math:`\mathbb{1}[\cdot]`: Indicator function

**Interpretation**:

- ACAUC = 1.0: Perfect calibration
- ACAUC < 1.0: Under-coverage (overconfident posterior)
- ACAUC > 1.0: Over-coverage (too conservative)

Other Metrics
~~~~~~~~~~~~~

**AEPC** (Average Expected Posterior Coverage):
    Discrete calibration at specific :math:`\alpha` levels (e.g., 0.95)

**LPP** (Log Posterior Probability):
    Measures likelihood quality: :math:`\mathbb{E}[\log q_\phi(\theta^*|x_{\text{obs}})]`

**NRMSE** (Normalized Root Mean Square Error):
    Parameter estimation accuracy: :math:`\sqrt{\mathbb{E}[\|\theta^* - \hat{\theta}\|^2]} / \|\theta^*\|`

**ESS** (Effective Sample Size):
    Sample efficiency (SIR task only)

Implementation Details
----------------------

Normalizing Flows
~~~~~~~~~~~~~~~~~

Both :math:`p_{\text{sim}}(x|\theta)` and :math:`q_\phi(\theta|x)` are implemented as normalizing flows:

- **Architecture**: Rational quadratic splines (RQS)
- **Invertibility**: Enables efficient density evaluation and sampling
- **Expressiveness**: Universal approximation of continuous distributions

Optimization
~~~~~~~~~~~~

- **Optimizer**: Adam with learning rate :math:`10^{-3}`
- **Batch size**: 1024
- **Gradient clipping**: Max norm 10.0
- **Early stopping**: Patience 100 epochs on validation loss

Shrinkage Prior Tuning
~~~~~~~~~~~~~~~~~~~~~~~

Recommended values for :math:`\lambda_{\text{shrinkage}}`:

- 0.0: No regularization (may overfit)
- 0.01: Weak shrinkage (mild misspecification)
- 0.1: Medium shrinkage (moderate misspecification, **default**)
- 1.0: Strong shrinkage (minimal expected misspecification)
- 10.0: Very strong (near well-specified)

Tune via cross-validation on held-out calibration data.

References
----------

**Primary Paper**:
    O'Callaghan, Matthew, Mandel, Kaisey S., and Gilmore, Gerry (2025).
    "Robust variational neural posterior estimation for simulation-based inference."
    *arXiv preprint arXiv:2509.05724*.

**Related Work**:
    - Hermans, J., et al. (2021). "Likelihood-free inference with amortized approximate
      likelihood ratios." *ICML 2021*.
    - Ward, D., et al. (2022). "Robust neural posterior estimation and statistical model
      criticism." *NeurIPS 2022*.
    - Greenberg, D., et al. (2019). "Automatic posterior transformation for likelihood-free
      inference." *ICML 2019*.

**Normalizing Flows**:
    - Durkan, C., et al. (2019). "Neural spline flows." *NeurIPS 2019*.

For implementation details, see :doc:`api/index`.

For practical usage, see the `README <https://github.com/USERNAME/RVNP_SBI/blob/main/README.md>`_.
