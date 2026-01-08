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

Error Model
~~~~~~~~~~~

RVNP introduces an **error model** :math:`p_\alpha(x_{\text{obs}}|x_{\text{sim}},\theta)` that bridges the simulation-to-reality gap:

.. math::

    p_\alpha(x_{\text{obs}}|x_{\text{sim}},\theta) = \mathcal{N}(x_{\text{obs}}; x_{\text{sim}}, \xi(\theta;\alpha))

where:

- :math:`x_{\text{sim}}`: Simulator output
- :math:`x_{\text{obs}}`: Real observation
- :math:`\xi(\theta;\alpha)`: Covariance matrix parametrized by :math:`\alpha`
- :math:`\theta`: Parameters of interest

**Paper Formulation** (O'Callaghan et al., 2025, Equations 15-16):

RVNP (Default):

.. math::

    \xi(\theta;\alpha) = \text{Diag}(\text{NN}(\theta;\alpha)) + \Lambda

where :math:`\text{NN}(\theta;\alpha)` outputs diagonal components and :math:`\Lambda` are learned global off-diagonal components.

RVNP-G (Global):

.. math::

    \xi(\theta;\alpha) = \alpha

A full-rank Gaussian covariance matrix (via Cholesky decomposition) that is constant across :math:`\theta`.

**Implementation Note**: The current implementation uses a **neural mean + neural covariance** variant (``correction_type='mu_hybrid'``) that differs from the paper:

.. math::

    p_\psi(\hat{x}|x,\theta) &= \mathcal{N}(\hat{x}; \mu_\psi(x,\theta), \Sigma_\psi(\theta)) \\
    \mu_\psi(x,\theta) &= x + \mu_{\text{global}} + \mu_\theta(\theta) \\
    \Sigma_\psi(\theta) &= L_{\text{hybrid}}(\theta) L_{\text{hybrid}}(\theta)^T

This includes a learnable neural mean shift :math:`\mu_\theta(\theta)` regularized by the shrinkage prior :math:`\mathcal{R}_{\text{shrink}}(\psi) = \frac{1}{K}\sum_k \|\mu_\theta(\theta_k)\|^2`.

Importance-Weighted Variational Objective
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RVNP jointly trains the posterior :math:`q_\phi(\theta|\hat{x})` and correction :math:`r_\psi(\hat{x}|x,\theta)` using an **importance-weighted objective**:

.. math::

    \mathcal{L}(\phi,\psi) = -\mathcal{L}_{\text{IWAE}}(\phi,\psi) + \lambda_{\text{shrinkage}} \cdot \mathcal{R}_{\text{shrink}}(\psi)

**Loss Computation** (all inside ``_kl_divergence`` method):

Given observed data :math:`x_{\text{obs}}`:

1. **Sample** :math:`\theta_1, \ldots, \theta_K \sim q_\phi(\theta|x_{\text{obs}})` (K samples per observation)

2. **For each** :math:`\theta_k`, **sample** :math:`x_{\text{sim}}^{(1)}, \ldots, x_{\text{sim}}^{(N)} \sim p_{\text{sim}}(x|\theta_k)` (N samples per theta)

3. **Compute Importance-Weighted Autoencoder Loss**:

   .. math::

       \mathcal{L}_{\text{IWAE}} = \frac{1}{K} \sum_{k=1}^{K} \log \left( \frac{1}{N} \sum_{n=1}^{N} \frac{r_\psi(x_{\text{obs}}|x_{\text{sim}}^{(n)}, \theta_k) \, p(\theta_k)}{q_\phi(\theta_k|x_{\text{obs}})} \right)

   Equivalently (using logsumexp for numerical stability):

   .. math::

       \mathcal{L}_{\text{IWAE}} = \frac{1}{K} \sum_{k=1}^{K} \left[ \text{logsumexp}_{n=1}^{N} \left( \log r_\psi(x_{\text{obs}}|x_{\text{sim}}^{(n)}, \theta_k) \right) - \log N - \log q_\phi(\theta_k|x_{\text{obs}}) + \log p(\theta_k) \right]

   where:

   - :math:`r_\psi(x_{\text{obs}}|x_{\text{sim}}, \theta)`: Correction model likelihood
   - :math:`p(\theta)`: Prior distribution
   - :math:`q_\phi(\theta|x_{\text{obs}})`: Posterior approximation
   - logsumexp provides tighter bound than single-sample ELBO

4. **Compute Shrinkage Regularization**:

   .. math::

       \mathcal{R}_{\text{shrink}}(\psi) = \frac{1}{K} \sum_{k=1}^{K} \|\mu_\theta(\theta_k)\|^2

   where :math:`\theta_k` are the SAME samples from step 1, and :math:`\mu_\theta(\theta)` is the neural mean correction function.

   - Regularizes neural mean shift toward zero
   - Prevents overfitting when mean misspecification is minimal
   - **Important**: Only penalizes the mean neural network output, NOT the covariance

5. **Total Loss**:

   .. math::

       \mathcal{L}(\phi,\psi) = -\mathcal{L}_{\text{IWAE}} + \lambda_{\text{shrinkage}} \cdot \mathcal{R}_{\text{shrink}}(\psi)

Multi-Stage Training
~~~~~~~~~~~~~~~~~~~~

RVNP uses a 3-stage training pipeline:

**Stage 1: Embedding Training** (optional, for high-dimensional observations)

- **Data**: Pre-generated simulations :math:`(\theta, x) \sim p(\theta)p_{\text{sim}}(x|\theta)`
- **Trains**: Embedding network :math:`f_\omega(x)`, discriminator, decoder
- **Method**: InfoMax (mutual information maximization)
- **Output**: Trained :math:`f_\omega` that compresses high-dimensional :math:`x` to low-dimensional embeddings

**Stage 2: Simulator Flow Training**

- **Data**: Pre-generated simulations :math:`(\theta, x) \sim p(\theta)p_{\text{sim}}(x|\theta)`
- **Trains**: Simulator flow :math:`p_{\text{sim}}(x|\theta)`
- **Method**: Maximum likelihood on simulated data
- **Output**: Trained simulator that generates :math:`x \sim p_{\text{sim}}(x|\theta)` for any :math:`\theta`

**Stage 3: Joint Posterior + Correction Training**

- **Data**: ONLY observed data :math:`x_{\text{obs}}` (no pre-generated simulations used)
- **Trains**: Posterior :math:`q_\phi(\theta|\hat{x})` and correction :math:`r_\psi(\hat{x}|x,\theta)` jointly
- **Method**: RVNP Loss (:math:`-\mathcal{L}_{\text{IWAE}}` + shrinkage regularization)
- **Training Loop**:

  * Pass :math:`x_{\text{obs}}` to RVNPLoss function
  * ALL sampling happens inside ``_kl_divergence`` method:

    1. Sample :math:`\theta_1, \ldots, \theta_K \sim q_\phi(\theta|x_{\text{obs}})` from current posterior
    2. For each :math:`\theta_k`, sample :math:`x_{\text{sim}}^{(n)} \sim p_{\text{sim}}(x|\theta_k)` from trained simulator (Stage 2)
    3. Compute :math:`\mathcal{L}_{\text{IWAE}}` using correction model :math:`r_\psi(x_{\text{obs}}|x_{\text{sim}},\theta)`
    4. Compute shrinkage: :math:`\mathcal{R}_{\text{shrink}}(\psi) = \frac{1}{K}\sum_{k}\|\mu_\theta(\theta_k)\|^2` using sampled :math:`\theta_k`
    5. Return :math:`-\mathcal{L}_{\text{IWAE}} + \lambda_{\text{shrinkage}} \cdot \mathcal{R}_{\text{shrink}}(\psi)`

  * Update :math:`\phi` (posterior) and :math:`\psi` (correction) via gradient descent
  * **Key point**: No sampling in training loop - only :math:`x_{\text{obs}}` passed to loss

This staged approach ensures:

- Stable learning of complex components
- Efficient reuse of trained simulator in Stage 3
- No need for massive pre-generated datasets in final training stage

Calibration Metrics
-------------------

**Paper Metrics** (O'Callaghan et al., 2025, Section 4.1):

The paper uses the following metrics:

**AEPC** (Average Expected Posterior Coverage):
    .. math::

        \alpha := \int_0^1 [\text{EPC}(\gamma) - \gamma] d\gamma

    where :math:`\text{EPC}(\gamma) = \mathbb{E}_{\theta^*,x_{\text{obs}}}[\mathbb{1}\{\theta^* \in \text{HDR}_{q_\phi(\theta|x_{\text{obs}})}(1-\gamma)\}]`

**AEMPC** (Average Expected Marginal Posterior Coverage):
    .. math::

        \alpha_{(\text{marginal})} := \frac{1}{m} \sum_{i=1}^m \int_0^1 [\text{EMPC}(\gamma)_i - \gamma] d\gamma

**LPP** (Log Posterior Probability):
    .. math::

        \text{LPP} := \mathbb{E}[\log q_\phi(\theta^*|x_{\text{obs}})]

**NRMSE** (Normalized Root Mean Square Error):
    .. math::

        \text{NRMSE} = \frac{1}{N_{\text{obs}}} \sum_{j=1}^{N_{\text{obs}}} \frac{\sqrt{\frac{1}{S}\sum_{s=1}^S (\theta_j^* - \theta_j^{(s)})^2}}{\text{Std}(\theta_{\text{prior}})}

**Implementation Metrics**:

The current implementation uses **ACAUC** (Average Coverage Area Under Curve) as the primary calibration metric, which differs from the paper:

.. math::

    \text{ACAUC} = \frac{1}{d} \sum_{j=1}^{d} \int_{0}^{1} \mathbb{1}[\theta_j^* \in C_\alpha^j] \, d\alpha

where:

- :math:`d`: Parameter dimension
- :math:`\theta_j^*`: True value of parameter :math:`j`
- :math:`C_\alpha^j`: :math:`\alpha`-level credible interval for dimension :math:`j`

**Interpretation**:

- ACAUC = 1.0: Perfect calibration
- ACAUC < 1.0: Under-coverage (overconfident posterior)
- ACAUC > 1.0: Over-coverage (too conservative)

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
