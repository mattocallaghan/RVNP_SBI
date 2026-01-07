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

Importance-Weighted Variational Objective
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RVNP jointly trains the posterior :math:`q_\phi(\theta|\hat{x})` and correction :math:`r_\psi(\hat{x}|x,\theta)` using an **importance-weighted variational objective**:

.. math::

    \mathcal{L}(\phi,\psi) = -\text{ELBO}(\phi,\psi) + \lambda_{\text{shrinkage}} \cdot \mathcal{R}_{\text{shrink}}(\psi)

where the ELBO is computed using importance weighting and the shrinkage regularizes the correction model.

**Loss Computation** (all inside ``_kl_divergence`` method):

Given observed data :math:`x_{\text{obs}}`:

1. **Sample** :math:`\theta \sim q_\phi(\theta|x_{\text{obs}})` from current posterior
2. **Sample** :math:`x_{\text{sim}} \sim p_{\text{sim}}(x|\theta)` from trained simulator
3. **Compute IW-ELBO**:

   .. math::

       \text{ELBO} = \mathbb{E}_{\theta \sim q_\phi(\theta|x_{\text{obs}})} \left[ \text{logsumexp}_{x_{\text{sim}}} \log r_\psi(x_{\text{obs}}|x_{\text{sim}},\theta) \right]

   - Uses multiple samples per :math:`\theta` for importance weighting
   - Evaluates correction model's ability to map simulator outputs to observations

4. **Compute Shrinkage Prior**:

   .. math::

       \mathcal{R}_{\text{shrink}}(\psi) = \mathbb{E}_{\theta \sim q_\phi(\theta|x_{\text{obs}})}[\|\mu_\theta(\theta)\|^2]

   - Uses the SAME :math:`\theta` samples from step 1
   - Regularizes neural mean shift toward zero
   - Prevents overfitting when mean misspecification is minimal
   - Weighted by :math:`\lambda_{\text{shrinkage}}`
   - **Important**: Only penalizes the mean neural network output, NOT the covariance

5. **Return**: :math:`-\text{ELBO} + \lambda_{\text{shrinkage}} \cdot \mathcal{R}_{\text{shrink}}(\psi)`

Multi-Stage Training
~~~~~~~~~~~~~~~~~~~~

RVNP uses a 3-stage training pipeline:

**Stage 1: Embedding Training** (optional, for high-dimensional observations)

- **Data**: Pre-generated simulations :math:`(\\theta, x) \sim p(\\theta)p_{\\text{sim}}(x|\\theta)`
- **Trains**: Embedding network :math:`f_\\omega(x)`, discriminator, decoder
- **Method**: InfoMax (mutual information maximization)
- **Output**: Trained :math:`f_\\omega` that compresses high-dimensional :math:`x` to low-dimensional embeddings

**Stage 2: Simulator Flow Training**

- **Data**: Pre-generated simulations :math:`(\\theta, x) \sim p(\\theta)p_{\\text{sim}}(x|\\theta)`
- **Trains**: Simulator flow :math:`p_{\\text{sim}}(x|\\theta)`
- **Method**: Maximum likelihood on simulated data
- **Output**: Trained simulator that generates :math:`x \sim p_{\\text{sim}}(x|\\theta)` for any :math:`\\theta`

**Stage 3: Joint Posterior + Correction Training**

- **Data**: ONLY observed data :math:`x_{\\text{obs}}` (no pre-generated simulations used)
- **Trains**: Posterior :math:`q_\\phi(\\theta|\hat{x})` and correction :math:`r_\\psi(\hat{x}|x,\\theta)` jointly
- **Method**: RVNP Loss (importance-weighted ELBO + shrinkage regularization)
- **Training Loop**:

  * Pass :math:`x_{\\text{obs}}` to RVNPLoss function
  * ALL sampling happens inside ``_kl_divergence`` method:

    1. Sample :math:`\\theta \sim q_\\phi(\\theta|x_{\\text{obs}})` from current posterior
    2. Sample :math:`x_{\\text{sim}} \sim p_{\\text{sim}}(x|\\theta)` from trained simulator (Stage 2)
    3. Compute IW-ELBO with corrected observations :math:`\hat{x} \sim r_\\psi(\hat{x}|x_{\\text{sim}},\\theta)`
    4. Compute shrinkage regularization: :math:`\lambda_{\\text{shrinkage}} \cdot \mathbb{E}_{\\theta}[\|\mu_\\theta(\\theta)\|^2]` using sampled :math:`\\theta`
    5. Return :math:`-\\text{ELBO} + \\text{shrinkage}`

  * Update :math:`\\phi` (posterior) and :math:`\\psi` (correction) via gradient descent
  * **Key point**: No sampling in training loop - only :math:`x_{\\text{obs}}` passed to loss

This staged approach ensures:

- Stable learning of complex components
- Efficient reuse of trained simulator in Stage 3
- No need for massive pre-generated datasets in final training stage

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
