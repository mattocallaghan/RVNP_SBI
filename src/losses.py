"""Loss functions for RVNP training."""

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from jax.lax import stop_gradient
from jax.random import split
from jax.scipy.special import logsumexp
from jaxtyping import Array, ArrayLike, Float, PRNGKeyArray
from functools import partial

import equinox as eqx
import paramax
import optax
from flowjax.distributions import AbstractDistribution

from .models.correction_model import (
    CorrectionModel, SimpleCorrectionModel, DiagonalNeuralCorrectionModel,
    HybridCorrectionModel, FullNeuralCorrectionModel, MuHybridCorrectionModel,
    GlobalCorrectionModel
)
from jax import custom_vjp
from typing import Tuple, Any


class MaximumLikelihoodLoss:
    """Loss for fitting a flow with maximum likelihood (negative log likelihood).

    This loss can be used to learn either conditional or unconditional distributions.
    """
    
    def __init__(self, min_loss_bound: float = None):
        """
        Args:
            min_loss_bound: Optional lower bound for the loss to prevent collapse.
                           If provided, loss will be clipped to this minimum value.
        """
        self.min_loss_bound = min_loss_bound

    @eqx.filter_jit
    def __call__(
        self,
        params: AbstractDistribution,
        static: AbstractDistribution,
        x: Array,
        condition: Array | None = None,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, ""]:
        """Compute the loss. Key is ignored (for consistency of API)."""
        dist = paramax.unwrap(eqx.combine(params, static))
        loss = -jax.vmap(dist.log_prob)(x, condition).mean()
        
        # Apply hard clipping if minimum bound specified
        if self.min_loss_bound is not None:
            loss = jnp.maximum(loss, self.min_loss_bound)
            
        return loss



class ShannonLossEmbedding(eqx.Module):
    """
    InfoMax / Jensen-Shannon embedding loss + geometry regularizers that
    encourage parameter-sensitive, well-conditioned latent directions.
    
    Optionally includes decoder reconstruction terms for both synthetic and real data.

    Inputs expected:
      - x: synthetic batch (B, ...)
      - x_real: real batch (B_real, ...)
      - condition: theta batch (B, p) aligned with x rows
      - key: PRNGKey
    """

    # hyperparameters (tune these)
    lambda_geom: float = 1.0       # weight for geometry log-det loss (encourage sensitivity)
    lambda_scale: float = 1.0      # weight for trace/scale penalty (prevent blow-up)
    lambda_covdiag: float = 0.1    # weight for diagonal covariance (encourage use of dims)
    lambda_reconstruction: float = 0.0  # weight for decoder reconstruction loss
    use_decoder: bool = False      # enable/disable decoder reconstruction term
    ridge: float = 1e-3            # ridge for theta pseudo-inverse
    eps: float = 1e-6              # numerical stabilizer for slogdet
    target_trace: float = None     # if None, use batch trace baseline (set if you want fixed scale)
    num_shuffles: int = 30         # number of permutations for marginal term (as before)

    # note: params/static are not stored here; they are passed into __call__
    def __call__(self, params, static, x, x_real, condition, key):
        # Unpack modules (same as your code)
        params_embedding, params_discriminator, params_decoder = params
        static_embedding, static_discriminator, static_decoder = static
        embedding = eqx.combine(params_embedding, static_embedding)
        discriminator = eqx.combine(params_discriminator, static_discriminator)
        decoder = eqx.combine(params_decoder, static_decoder)

        batch_size = x.shape[0]

        # -------------------------
        # Compute embeddings z and z_real
        # -------------------------
        key, subkey = split(key)
        keys = jax.random.split(subkey, batch_size)
        p_embedding = lambda x_i, k: embedding(x_i, key=k, inference=True)
        # note: your original code vmap'd with shape x[:, jnp.newaxis, :] — preserve that if needed
        z = vmap(p_embedding)(x[:, jnp.newaxis, :], keys)        # (B, z_dim)
        key, subkey = split(key)
        real_keys = jax.random.split(subkey, x_real.shape[0])
        z_real = vmap(p_embedding)(x_real[:, jnp.newaxis, :], real_keys)  # (B_real, z_dim)

        # -------------------------
        # Decoder reconstruction loss (optional)
        # -------------------------
        reconstruction_loss = 0.0
        if self.use_decoder and self.lambda_reconstruction > 0.0:
            # Reconstruct synthetic data from embeddings
            key, subkey = split(key)
            keys_recon = jax.random.split(subkey, batch_size)
            p_decoder = lambda z_i, k: decoder(z_i, key=k, inference=True)
            x_reconstructed = vmap(p_decoder)(z, keys_recon)  # (B, x_dim)
            
            # Reconstruct real data from embeddings
            key, subkey = split(key)
            real_keys_recon = jax.random.split(subkey, x_real.shape[0])
            #x_real_reconstructed = vmap(p_decoder)(z_real, real_keys_recon)  # (B_real, x_dim)
            
            # Compute MSE reconstruction losses
            # Remove the channel dimension for comparison if it exists
            x_flat = x.reshape(x.shape[0], -1)  # Flatten to (B, x_dim)
            #x_real_flat = x_real.reshape(x_real.shape[0], -1)  # Flatten to (B_real, x_dim)
            x_reconstructed_flat = x_reconstructed.reshape(x_reconstructed.shape[0], -1)
            #x_real_reconstructed_flat = x_real_reconstructed.reshape(x_real_reconstructed.shape[0], -1)
            
            reconstruction_loss_synthetic = jnp.mean((x_flat - x_reconstructed_flat)**2)
            #reconstruction_loss_real = jnp.mean((x_real_flat - x_real_reconstructed_flat)**2)
            reconstruction_loss = self.lambda_reconstruction * (reconstruction_loss_synthetic )

        # -------------------------
        # Shannon / InfoMax 
        # -------------------------
        # Joint logits
        key, subkey = split(key)
        keys = jax.random.split(subkey, batch_size)
        logits_joint = jax.vmap(discriminator)(z, condition, key=keys)  # (B,)

        # Marginal terms via shuffles
        key, subkey = split(key)
        perm_keys = jax.random.split(subkey, self.num_shuffles)

        def marginal_term_fn(perm_key):
            perm_key, key_inner = split(perm_key)
            perm = jax.random.permutation(perm_key, condition.shape[0])
            condition_shuffled = condition[perm]
            keys_inner = jax.random.split(key_inner, batch_size)
            logits_marginal = jax.vmap(discriminator)(z, condition_shuffled, key=keys_inner)
            return -jax.nn.softplus(logits_marginal).mean()

        marginal_terms = vmap(marginal_term_fn)(perm_keys)
        marginal_term_avg = jnp.mean(marginal_terms)
        joint_term = -jax.nn.softplus(-logits_joint).mean()
        mi_lower_bound = joint_term + marginal_term_avg
        shannon_loss = -mi_lower_bound  # minimize negative lower bound



        # -------------------------
        # Total loss combine
        # -------------------------
        total_loss = shannon_loss + reconstruction_loss #+ geometry_loss + scale_loss + covdiag_loss + reconstruction_loss

        # Return losses (helpful to debug)
        metrics = {
            "total_loss": total_loss,
            "shannon_loss": shannon_loss,
            "reconstruction_loss": reconstruction_loss,
        }

        return total_loss#, metrics




# ============================================================================
# DReG (Doubly Reparameterized Gradient) Helper Functions
# ============================================================================

def _compute_normalized_weights(log_weights: Array) -> Array:
    """Compute normalized importance weights from log weights.

    Args:
        log_weights: Log importance weights, shape (n_obs, K) or (K,)

    Returns:
        Normalized weights that sum to 1: exp(lw - logsumexp(lw))

    Example:
        >>> log_weights = jnp.array([[1.0, 2.0, 3.0]])
        >>> weights = _compute_normalized_weights(log_weights)
        >>> jnp.sum(weights)  # Should be 1.0
    """
    log_weights_normalized = log_weights - logsumexp(log_weights, axis=-1, keepdims=True)
    return jnp.exp(log_weights_normalized)



def _iwae_elbo_computation(
    correction_model,
    simulator_flow,
    x_obs_real: Array,
    thetas_sampled: Array,
    keys_sim: Array,
    n_sim_samples: int,
) -> Array:
    """Compute ELBO terms (inner logsumexp over x_sim samples).

    This function is shared between forward and backward passes in DReG.
    It samples from the simulator and computes the ELBO for each (obs, theta) pair:

    ELBO_k = logsumexp_n[log p(x_obs|x_sim^n, θ_k)] - log N

    Args:
        correction_model: Correction model r_ψ(x_obs|x_sim, θ)
        simulator_flow: Trained simulator p(x|θ)
        x_obs_real: Observed data, shape (n_obs, obs_dim)
        thetas_sampled: Sampled theta values, shape (n_obs, K, theta_dim)
        keys_sim: Random keys for simulator sampling, shape (n_obs, K)
        n_sim_samples: Number of simulator samples N per theta

    Returns:
        ELBO values, shape (n_obs, K)
    """
    # Sample x_sim from simulator for each theta
    def sample_from_simulator_single(theta_single, key_single):
        return simulator_flow.sample(key_single, (n_sim_samples,), condition=theta_single)

    # Nested vmap: (n_obs, K) -> (n_obs, K, N, x_dim)
    x_sim_samples = vmap(vmap(sample_from_simulator_single))(thetas_sampled, keys_sim)

    # Compute ELBO for each (obs, theta) pair
    def compute_elbo_single_obs_theta(x_obs_single, x_sim_samples_k, theta_k):
        """Compute ELBO for single observation and single theta."""
        # Prepare theta for broadcasting to match x_sim batch size
        theta_1d = jnp.atleast_1d(theta_k)
        theta_batch = jnp.broadcast_to(
            theta_1d[None, :], (n_sim_samples, theta_1d.shape[-1])
        )

        # Compute log p(x_obs|x_sim^n, θ_k) for all N samples
        log_p_obs_given_sim = vmap(
            lambda x_sim, theta: correction_model.log_prob(x_obs_single, x_sim, theta)
        )(x_sim_samples_k, theta_batch)

        # ELBO_k = logsumexp_n[log p(x_obs|x_sim^n, θ_k)] - log N
        elbo_k = logsumexp(log_p_obs_given_sim) - jnp.log(n_sim_samples)
        return elbo_k

    def compute_elbo_single_obs(x_obs_single, x_sim_samples_all_k, thetas_all_k):
        """Compute ELBO for all K thetas given single observation."""
        return vmap(
            lambda x_sim_k, theta_k: compute_elbo_single_obs_theta(
                x_obs_single, x_sim_k, theta_k
            )
        )(x_sim_samples_all_k, thetas_all_k)

    # Vectorize over observations: (n_obs, K)
    elbos = vmap(compute_elbo_single_obs)(x_obs_real, x_sim_samples, thetas_sampled)
    return elbos


# ============================================================================
# DReG Implementation - Simple Direct Approach
# ============================================================================

def _iwae_with_dreg(
    flow, correction_model, simulator_flow, prior_log,
    x_obs_real, key, K, N
):
    """IWAE loss with DReG gradient estimator.

    This uses a simple direct approach: compute IWAE loss for the forward pass,
    then compute DReG-weighted objectives that JAX autodiff will differentiate correctly.

    The key DReG insight: Use (w_k)² weights with stop_gradient on weight computation.

    Args:
        flow: Posterior flow q(θ|x)
        correction_model: Correction model r(x̂|x,θ)
        simulator_flow: Trained simulator p(x|θ)
        prior_log: Prior log probability function
        x_obs_real: Observed data
        key: Random key
        K: Number of importance samples
        N: Number of simulator samples per theta

    Returns:
        IWAE loss (for forward) with DReG gradients (for backward)
    """
    n_obs = x_obs_real.shape[0]

    # ============================================================================
    # Forward pass: Standard IWAE
    # ============================================================================

    # 1. Sample theta ~ q_φ(θ|x_obs)
    key_theta, key = jax.random.split(key)
    keys_theta = jax.random.split(key_theta, n_obs)

    def sample_thetas(x_single, k):
        return flow.sample_and_log_prob(k, (K,), condition=x_single)

    thetas_sampled, log_p_posterior = vmap(sample_thetas)(x_obs_real, keys_theta)

    # 2. Compute prior log prob
    prior_logp = vmap(vmap(prior_log))(thetas_sampled)

    # 3. Generate simulator keys
    key_sim, key = jax.random.split(key)
    keys_sim = jax.random.split(key_sim, n_obs * K).reshape(n_obs, K)

    # 4. Compute ELBO
    elbos = _iwae_elbo_computation(
        correction_model, simulator_flow, x_obs_real,
        thetas_sampled, keys_sim, N
    )

    # 5. Compute IWAE loss (for forward value)
    log_weights = elbos - log_p_posterior
    iwae_per_obs = logsumexp(log_weights, axis=-1) + jnp.mean(prior_logp, axis=-1) - jnp.log(K)
    iwae_loss = -jnp.mean(iwae_per_obs)

    # ============================================================================
    # DReG: Recompute with stopped gradients for backward pass
    # ============================================================================

    # Recompute log q(θ|x) with STOPPED flow parameters (key DReG component)
    # Partition flow to stop gradients on parameters only
    params_flow, static_flow = eqx.partition(flow, eqx.is_inexact_array)
    params_flow_stopped = jax.tree_map(jax.lax.stop_gradient, params_flow)
    flow_stopped = eqx.combine(params_flow_stopped, static_flow)

    def compute_log_q_stopped(x_single, thetas_k):
        return vmap(lambda theta: flow_stopped.log_prob(theta, condition=x_single))(thetas_k)

    log_p_posterior_stopped = vmap(compute_log_q_stopped)(x_obs_real, thetas_sampled)

    # Compute normalized importance weights with stopped gradients
    log_weights_stopped = elbos - log_p_posterior_stopped
    grad_weights = _compute_normalized_weights(log_weights_stopped)

    # SQUARE the weights (THE KEY DReG DIFFERENCE from STL!)
    grad_weights_squared = grad_weights ** 2

    # Stop gradients on sampled thetas (treat them as fixed for gradient computation)
    thetas_stopped = jax.lax.stop_gradient(thetas_sampled)

    # Recompute log q WITHOUT stop_gradient (we want gradients through flow params!)
    def compute_log_q_for_grad(x_single, thetas_k):
        return vmap(lambda theta: flow.log_prob(theta, condition=x_single))(thetas_k)

    log_q_for_grad = vmap(compute_log_q_for_grad)(x_obs_real, thetas_stopped)

    # DReG objective for flow: weighted by SQUARED weights
    dreg_log_weights_flow = elbos - log_q_for_grad
    dreg_weighted_flow = grad_weights_squared * dreg_log_weights_flow
    dreg_sum_flow = jnp.sum(dreg_weighted_flow, axis=-1)
    prior_contrib = jnp.mean(prior_logp, axis=-1)
    dreg_objective_flow = jnp.mean(dreg_sum_flow + prior_contrib)

    # Standard IWAE objective for correction: weighted by single weights
    iwae_weighted_corr = grad_weights * elbos
    iwae_sum_corr = jnp.sum(iwae_weighted_corr, axis=-1)
    iwae_objective_corr = jnp.mean(iwae_sum_corr + prior_contrib)

    # Combine: return IWAE loss but with DReG objective gradients
    # This works because JAX will autodiff through dreg_objective_flow and iwae_objective_corr
    combined_objective = dreg_objective_flow + iwae_objective_corr

    # Return standard IWAE loss for forward, but structured so gradients come from DReG
    # Use stop_gradient trick: stopped_loss + (objective - stopped_objective)
    # This gives forward value of iwae_loss but backward gradients of combined_objective
    dreg_loss = jax.lax.stop_gradient(iwae_loss) + (combined_objective - jax.lax.stop_gradient(combined_objective))

    return -dreg_loss  # Negative because we're minimizing loss







class RVNPLoss:
    """
    RVNP Loss: Robust Variational Neural Posterior loss function.

    Implements the importance-weighted variational objective with shrinkage and entropy regularization:

    L(φ,ψ) = IW-ELBO + λ_shrinkage * ||μ_θ(θ)||² + λ_entropy * H(r_ψ)

    where IW-ELBO is the importance-weighted evidence lower bound computed via importance_weighted_ae_loss.
    """
    
    def __init__(
        self,
        lambda_variational: float = 1.0,  # Weight for IW-ELBO term
        lambda_kl: float = 1.0,  # Weight for KL divergence within ELBO
        lambda_shrinkage: float = 0.0,  # Shrinkage prior on mean neural network
        lambda_entropy: float = 0.0,  # Entropy regularization for covariance
        simulator_samples_per_theta: int = 32,  # Number of theta samples from posterior per observation
        n_sim_samples_per_theta: int = 32,  # Number of x_sim samples per theta in KL divergence
        prior: AbstractDistribution = None,  # Prior distribution
        empirical_bias: Array = None,  # Empirical bias for shrinkage prior (optional)
        use_dreg: bool = True,  # Use DReG gradient estimator for variance reduction
    ):
        self.lambda_variational = lambda_variational
        self.lambda_kl = lambda_kl
        self.lambda_shrinkage = lambda_shrinkage
        self.lambda_entropy = lambda_entropy
        self.simulator_samples_per_theta = simulator_samples_per_theta
        self.n_sim_samples_per_theta = n_sim_samples_per_theta
        self.prior = prior
        self.empirical_bias = empirical_bias
        self.use_dreg = use_dreg

    @eqx.filter_jit
    def __call__(
        self,
        params_flow: AbstractDistribution,
        static_flow: AbstractDistribution,
        params_embedding: eqx.Module,
        static_embedding: eqx.Module,
        params_correction: eqx.Module,
        static_correction: eqx.Module,
        simulator_flow: AbstractDistribution,  # Complete simulator flow (not params/static)
        x_obs: Array,
        key: PRNGKeyArray,
        embedding_stats: dict = None,
    ) -> Float[Array, ""]:
        """Compute RVNP loss: importance-weighted autoencoder objective with shrinkage regularization.

        Implements the RVNP loss function:

        .. math::

            \mathcal{L}(\phi,\psi) = -\mathcal{L}_{\text{IWAE}}(\phi,\psi) + \lambda_{\text{shrinkage}} \cdot \mathcal{R}_{\text{shrink}}(\psi)

        where:

        - :math:`\mathcal{L}_{\text{IWAE}}`: Importance-weighted autoencoder loss (computed in importance_weighted_ae_loss)
        - :math:`\mathcal{R}_{\text{shrink}}(\psi) = \frac{1}{K}\sum_{k=1}^K \|\mu_\theta(\theta_k)\|^2`: Shrinkage regularization
        - :math:`q_\phi(\theta|\hat{x})`: Posterior flow (amortized inference)
        - :math:`r_\psi(\hat{x}|x,\theta)`: Correction model
        - :math:`p_{\text{sim}}(x|\theta)`: Simulator flow (trained separately)

        Args:
            params_flow: Trainable parameters of posterior flow
            static_flow: Static (non-trainable) parameters of posterior flow
            params_embedding: Trainable parameters of embedding network (optional)
            static_embedding: Static parameters of embedding network
            params_correction: Trainable parameters of correction model
            static_correction: Static parameters of correction model
            simulator_flow: Pre-trained simulator flow (complete model)
            x_obs: Observed data, shape (n_obs, obs_dim).
                   **ONLY data input** - all sampling happens inside importance_weighted_ae_loss
            key: JAX random key for stochastic sampling
            embedding_stats: Optional dict with 'mean' and 'std' for embedding normalization

        Returns:
            Scalar loss = :math:`-\mathcal{L}_{\text{IWAE}} + \lambda_{\text{shrinkage}} \cdot \mathcal{R}_{\text{shrink}}`

        Notes:
            All sampling occurs inside importance_weighted_ae_loss:

            1. Sample :math:`\theta_1, \ldots, \theta_K \sim q_\phi(\theta|x_{\text{obs}})`
            2. For each :math:`\theta_k`, sample :math:`x_{\text{sim}}^{(n)} \sim p_{\text{sim}}(x|\theta_k)`
            3. Compute :math:`\mathcal{L}_{\text{IWAE}}` using correction model
            4. Compute :math:`\mathcal{R}_{\text{shrink}}(\psi)` using sampled :math:`\theta_k`
            5. Return :math:`-\mathcal{L}_{\text{IWAE}} + \lambda_{\text{shrinkage}} \cdot \mathcal{R}_{\text{shrink}}`
        """

        ###########
        ## pull in the parameters and static parameters
        # Normalizing flow (posterior)
        flow = paramax.unwrap(eqx.combine(params_flow, static_flow))
        correction_model = paramax.unwrap(eqx.combine(params_correction, static_correction))
        n_obs = x_obs.shape[0]
        # ======== ======== ======== ======== ======== ======== ========
        # ========              EMBEDDING SECTION               ========
        # ======== ======== ======== ======== ======== ======== ========
        if params_embedding is not None:
            embedding = paramax.unwrap(eqx.combine(params_embedding, static_embedding))
            key_embed_obs, key = jax.random.split(key)
            keys_obs = jax.random.split(key_embed_obs, n_obs)
            x_obs_embedded = jax.lax.stop_gradient(
                vmap(lambda x, k: embedding(x, key=k, inference=True))(x_obs[:, jnp.newaxis, :], keys_obs)
            )
            x_obs_processed = (x_obs_embedded - embedding_stats['mean']) / embedding_stats['std']
        else:
            x_obs_processed = x_obs
        # ======== ======== ======== ======== ======== ======== ========
        # ========              RVNP LOSS                       ========
        # ======== ======== ======== ======== ======== ======== ========
        # L_IWAE (Importance-Weighted Autoencoder loss) with shrinkage regularization
        # All sampling (θ_k ~ q_φ(θ|x_obs) and x_sim ~ p(x|θ)) happens inside importance_weighted_ae_loss
        # Returns: -L_IWAE + λ_shrinkage * R_shrink
        key_variational, key = jax.random.split(key)
        reverse = False  # Use forward KL divergence, legacy code
        loss = self.lambda_variational * self.importance_weighted_ae_loss(
            correction_model, simulator_flow, flow, reverse, x_obs_processed, key_variational,
            n_samples=self.simulator_samples_per_theta, kl_weight=self.lambda_kl,
            prior_log=self.prior.log_prob, lambda_shrinkage=self.lambda_shrinkage
        )

        return loss


    @eqx.filter_jit
    def _compute_shrinkage_prior(self, correction_model, theta_sim: Array) -> Float[Array, ""]:
        """
        Compute shrinkage prior that assumes low misspecification:
        - Diagonal elements should be small (low variance)
        - Off-diagonal elements should be even smaller (very low correlations)
        
        Prior assumption: misspecification is minimal with very weak correlations
        
        Args:
            correction_model: Correction model with learnable covariance
            
        Returns:
            Shrinkage penalty with stronger penalty on off-diagonal elements
        """
        if isinstance(correction_model, SimpleCorrectionModel):
            # Get the full covariance matrix
            covariance_matrix = correction_model.get_covariance_matrix()
            
            # Extract diagonal and off-diagonal elements
            diagonal_elements = jnp.diag(covariance_matrix)
            
            # Get off-diagonal elements (upper + lower triangular, excluding diagonal)
            dim = correction_model.dim
            # Create mask for off-diagonal elements
            off_diag_mask = 1.0 - jnp.eye(dim)
            off_diagonal_elements = covariance_matrix * off_diag_mask
            
            # Shrinkage losses with different strengths
            # Diagonal elements: penalize toward zero, weighted by empirical bias magnitude
            diagonal_penalty=(jnp.log(2.0)-jnp.log(jnp.pi)- jnp.log(0.1)- jnp.log1p((((jax.nn.softplus(diagonal_elements) + 1e-6 - 0) / 0.1)) ** 2))
            diagonal_penalty=-diagonal_penalty
            
            # Weight each diagonal component by the absolute value of empirical bias
            if self.empirical_bias is not None:
                bias_weights = 0.3/jnp.abs(self.empirical_bias)
                diagonal_penalty = diagonal_penalty * bias_weights
            
            diagonal_penalty = diagonal_penalty.sum(-1)
            # Off-diagonal elements: penalize heavily toward zero (correlations should be very small)
            off_diagonal_penalty = jnp.sum(off_diagonal_elements**2,-1)
            
            # Combined loss: stronger penalty on off-diagonal elements
            shrinkage_loss = (diagonal_penalty).mean()  # 10x stronger penalty on correlations
            
        elif isinstance(correction_model, GlobalCorrectionModel):
            # GlobalCorrectionModel - similar to SimpleCorrectionModel but with mean shift
            covariance_matrix = correction_model.get_covariance_matrix()
            
            # Extract diagonal and off-diagonal elements
            diagonal_elements = jnp.diag(covariance_matrix)
            
            # Get off-diagonal elements (upper + lower triangular, excluding diagonal)
            dim = correction_model.dim
            off_diag_mask = 1.0 - jnp.eye(dim)
            off_diagonal_elements = covariance_matrix * off_diag_mask
            
            # Shrinkage penalties (same as SimpleCorrectionModel)
            diagonal_penalty = jnp.mean(diagonal_elements**2)
            off_diagonal_penalty = jnp.mean(off_diagonal_elements**2) * 10  # 10x stronger penalty on correlations
            
            # Combined loss: stronger penalty on off-diagonal elements
            shrinkage_loss = (diagonal_penalty + off_diagonal_penalty).mean()
            
        elif isinstance(correction_model, (DiagonalNeuralCorrectionModel, HybridCorrectionModel, MuHybridCorrectionModel, FullNeuralCorrectionModel)):
            # For neural correction models, apply same diagonal shrinkage as SimpleCorrectionModel
            # plus network parameter regularization

            # Get diagonal elements using the same approach as SimpleCorrectionModel
            if isinstance(correction_model, DiagonalNeuralCorrectionModel):
                # DiagonalNeuralCorrectionModel doesn't have get_covariance_matrix, skip diagonal penalty
                diagonal_penalty = 0.0
            else:
                # For theta-dependent models, compute diagonal penalty across all theta_sim values
                # Use vmap to compute covariance matrices for all theta values in the batch
                get_covariance_batch = jax.vmap(correction_model.get_covariance_matrix)
                covariance_matrices = get_covariance_batch(theta_sim)  # (batch_size, dim, dim)

                # Extract diagonal elements across all samples
                diagonal_elements = jnp.diagonal(covariance_matrices, axis1=1, axis2=2)  # (batch_size, dim)

                # Apply same diagonal penalty as SimpleCorrectionModel (vectorized across batch)
                diagonal_penalty_batch=(jnp.log(2.0)-jnp.log(jnp.pi)- jnp.log(0.1)- jnp.log1p((((jax.nn.softplus(diagonal_elements) + 1e-6 - 0) / 0.1)) ** 2))
                diagonal_penalty_batch=-diagonal_penalty_batch

                # Weight each diagonal component by the absolute value of empirical bias
                if self.empirical_bias is not None:
                    bias_weights = 1.0/jnp.abs(self.empirical_bias)
                    diagonal_penalty_batch = diagonal_penalty_batch * bias_weights[None, :]  # Broadcast across batch

                diagonal_penalty = diagonal_penalty_batch.sum(-1).mean()  # Average across batch and sum across dimensions

            # Shrinkage prior: only apply to mean neural network output (not covariance)
            mean_shift_penalty = 0.0
            if isinstance(correction_model, MuHybridCorrectionModel):
                # Shrinkage prior on neural mean shift: ||μ_θ(θ)||²
                # Encourages mean shift network to output zero (no mean misspecification)
                get_mean_magnitude_batch = jax.vmap(correction_model.get_mean_shift_magnitude)
                mean_shift_magnitudes = get_mean_magnitude_batch(theta_sim)  # (batch_size,)
                mean_shift_penalty = jnp.mean(mean_shift_magnitudes)

            # Only apply shrinkage to mean neural network, not covariance
            shrinkage_loss = mean_shift_penalty 
            
        else:
            # For other correction models, use fallback
            raise NotImplementedError (f"Shrinkage prior not implemented for {type(correction_model)}")
        
        return shrinkage_loss


    @eqx.filter_jit
    def importance_weighted_ae_loss(
        self,
        correction_model,
        simulator_flow,
        flow,
        reverse,                # Posterior flow for theta sampling
        x_obs_real: Array,     # (n_obs, dim) - real observations
        key: PRNGKeyArray,
        n_samples: int = 100 ,  # Number of samples to draw from simulator
        kl_weight: float = 1.0,  # Weight for KL term (default 1.0)
        prior_log: any = None,  # Prior distribution (if needed)
        lambda_shrinkage: float = 0.0,  # Weight for shrinkage prior on mean correction

    ) -> Float[Array, ""]:
        """
        Compute importance-weighted autoencoder loss (L_IWAE) with shrinkage regularization.

        This method performs ALL sampling internally:

        1. Sample :math:`\\theta_1, \ldots, \\theta_K \sim q_\phi(\\theta|x_{\\text{obs}})` (K samples per obs)
        2. For each :math:`\\theta_k`, sample :math:`x_{\\text{sim}}^{(1)}, \ldots, x_{\\text{sim}}^{(N)} \sim p(x|\\theta_k)` (N samples per theta)
        3. Compute L_IWAE using importance weighting:

           .. math::

               \mathcal{L}_{\\text{IWAE}} = \\frac{1}{K} \sum_{k=1}^{K} \left[ \\text{logsumexp}_{n=1}^{N} \log r_\psi(x_{\\text{obs}}|x_{\\text{sim}}^{(n)}, \\theta_k) - \log N - \log q_\phi(\\theta_k|x_{\\text{obs}}) + \log p(\\theta_k) \\right]

        4. Compute shrinkage regularization:

           .. math::

               \mathcal{R}_{\\text{shrink}} = \\frac{1}{K} \sum_{k=1}^{K} \|\\mu_\\theta(\\theta_k)\|^2

        5. Return :math:`-\mathcal{L}_{\\text{IWAE}} + \lambda_{\\text{shrinkage}} \cdot \mathcal{R}_{\\text{shrink}}`

        Args:
            correction_model: Correction model :math:`r_\psi(x_{\\text{obs}}|x_{\\text{sim}}, \\theta)`
            simulator_flow: Trained simulator :math:`p(x|\\theta)`
            flow: Posterior flow :math:`q_\phi(\\theta|x_{\\text{obs}})` for sampling
            reverse: Whether to use reverse KL (default: False)
            x_obs_real: Observed data, shape (n_obs, obs_dim)
            key: JAX random key for sampling
            n_samples: Number of x_sim samples per theta (N in formula)
            kl_weight: Weight for KL divergence term (default: 1.0)
            prior_log: Prior log probability function :math:`\log p(\\theta)`
            lambda_shrinkage: Shrinkage weight :math:`\lambda_{\\text{shrinkage}}` (default: 0.0)

        Returns:
            Scalar loss = :math:`-\mathcal{L}_{\\text{IWAE}} + \lambda_{\\text{shrinkage}} \cdot \mathcal{R}_{\\text{shrink}}`
        """

        # ========================================================================
        # DReG PATH: Use stop_gradient for variance-reduced gradients
        # ========================================================================
        if self.use_dreg:
            # Compute IWAE with DReG gradient estimator
            # This uses jax.lax.stop_gradient and squared weights directly
            iwae_loss = _iwae_with_dreg(
                flow, correction_model, simulator_flow, prior_log,
                x_obs_real, key,
                self.simulator_samples_per_theta,
                self.n_sim_samples_per_theta,
            )

            # Compute shrinkage loss separately with standard gradients
            shrinkage_loss = 0.0
            if lambda_shrinkage > 0.0:
                # Sample fresh thetas for shrinkage computation
                n_obs = x_obs_real.shape[0]
                K = self.simulator_samples_per_theta

                key_shrink, key = jax.random.split(key)
                keys_theta = jax.random.split(key_shrink, n_obs)

                def sample_thetas_fn(x_single, k):
                    return flow.sample(k, (K,), condition=x_single)

                thetas_for_shrinkage = vmap(sample_thetas_fn)(x_obs_real, keys_theta)
                theta_flat = thetas_for_shrinkage.reshape(-1, thetas_for_shrinkage.shape[-1])

                mean_shift_magnitudes = vmap(correction_model.get_mean_shift_magnitude)(theta_flat)
                shrinkage_loss = lambda_shrinkage * jnp.mean(mean_shift_magnitudes)

            return iwae_loss + shrinkage_loss

        # ========================================================================
        # STANDARD PATH: Original IWAE implementation (no DReG)
        # ========================================================================
        n_obs = x_obs_real.shape[0]

        # Sample theta from posterior for each observation
        key_theta_sample, key = jax.random.split(key)
        keys_theta = jax.random.split(key_theta_sample, n_obs)
        def sample_thetas_for_single_obs(x_obs_single, key_single):
            return flow.sample_and_log_prob(key_single, (self.simulator_samples_per_theta,), condition=x_obs_single)

        # Shape: (n_obs, samples_per_theta, theta_dim)
        thetas_sampled,log_p_posterior = jax.vmap(sample_thetas_for_single_obs)(x_obs_real, keys_theta)
        theta_for_sampling = thetas_sampled[:,:,:]

        prior_logp=jax.vmap(jax.vmap(prior_log))(theta_for_sampling) # evaluate log prior
        
        # Generate keys for simulator sampling: (n_obs, samples_per_theta)
        n_obs_total, samples_per_theta = theta_for_sampling.shape[0], theta_for_sampling.shape[1]
        
        key_sample, key = jax.random.split(key)
        keys_sim_flat = jax.random.split(key_sample, n_obs_total * samples_per_theta)
        keys_sim = keys_sim_flat.reshape(n_obs_total, samples_per_theta)

        # ########### ELBO Computation ###########
        # Sample x_sim from simulator for each theta
        n_samples = self.n_sim_samples_per_theta # Number of simulator samples per theta
        def sample_from_simulator_single(theta_single, key_single):
            # Sample like theta sampling: key, shape, condition
            return simulator_flow.sample(key_single, (n_samples,), condition=theta_single)
        
        # Sample x_sim from simulator: (n_obs, samples_per_theta, n_samples, x_dim)
        x_sim_samples = vmap(vmap(sample_from_simulator_single))(theta_for_sampling, keys_sim)
        n_obs, K_samples = thetas_sampled.shape[0], thetas_sampled.shape[1]
    

        # for single observation: 
        def compute_elbo_single_obs(x_obs_single, theta_samples_single, x_sim_samples_single, log_q_phi_single,log_prior_single):
            #(x_obs_single)=(x_dim)
            # x_sim_samples_single: (n_theta, n_samples, x_dim)
            # log_q_phi_single: (n_theta,)
            # log_prior_single: (n_theta,)
            def compute_terms_single_theta(x_sim_theta_k, theta_single_k):
                # x_sim_theta_k: (n_sim_samples, x_dim)
                # log_q_phi_k: (1)
                # log_prior_single_k: (1)

                theta_single_k = jnp.squeeze(theta_single_k)  # Remove any extra dimensions
                # We need to broadcast theta_single_k to match each x_sim sample
                n_sim_samples = x_sim_theta_k.shape[0]
                # Safely broadcast theta to match x_sim batch size (handle 0D case)
                theta_1d = jnp.atleast_1d(theta_single_k)
                theta_batch = jnp.broadcast_to(theta_1d[None, :], (n_sim_samples, theta_1d.shape[-1]))
                log_p_obs_given_sim = vmap(lambda x_sim, theta: correction_model.log_prob(x_obs_single, x_sim, theta))(x_sim_theta_k, theta_batch)

                elbo_single_obs_single_theta=logsumexp(log_p_obs_given_sim)- jnp.log(x_sim_theta_k.shape[0])
                return elbo_single_obs_single_theta
  
            # Compute for all K theta samples, single here refers to single theta sample
            elbo_single_obs= vmap(compute_terms_single_theta)(x_sim_samples_single, theta_samples_single)
            return logsumexp(elbo_single_obs-log_q_phi_single)+log_prior_single.mean()- jnp.log(theta_samples_single.shape[0])
        
        # Compute L_IWAE for all observations
        iwae_per_obs = vmap(compute_elbo_single_obs)(
            x_obs_real, thetas_sampled, x_sim_samples, log_p_posterior,prior_logp
        )

        # Negative L_IWAE as loss (we want to maximize L_IWAE, so minimize negative L_IWAE)
        iwae_loss = -jnp.mean(iwae_per_obs)

        # Compute shrinkage regularization R_shrink using sampled θ_k ~ q_φ(θ|x_obs)
        shrinkage_loss = 0.0
        if lambda_shrinkage > 0.0:
            # thetas_sampled shape: (n_obs, K, theta_dim)
            # Flatten to (n_obs * K, theta_dim) for batch processing
            theta_flat = thetas_sampled.reshape(-1, thetas_sampled.shape[-1])

            # Compute ||μ_θ(θ_k)||² for each sampled theta
            mean_shift_magnitudes = vmap(correction_model.get_mean_shift_magnitude)(theta_flat)

            # R_shrink = (1/K) Σ_k ||μ_θ(θ_k)||²
            shrinkage_loss = lambda_shrinkage * jnp.mean(mean_shift_magnitudes)

        # Total loss: -L_IWAE + λ_shrinkage * R_shrink
        return iwae_loss + shrinkage_loss

