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






class RVNPLoss:
    """
    RVNP Loss: Robust Variational Neural Posterior loss function.

    Implements the importance-weighted variational objective with shrinkage and entropy regularization:

    L(φ,ψ) = IW-ELBO + λ_shrinkage * ||μ_θ(θ)||² + λ_entropy * H(r_ψ)

    where IW-ELBO is the importance-weighted evidence lower bound computed via _kl_divergence.
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
    ):
        self.lambda_variational = lambda_variational
        self.lambda_kl = lambda_kl
        self.lambda_shrinkage = lambda_shrinkage
        self.lambda_entropy = lambda_entropy
        self.simulator_samples_per_theta = simulator_samples_per_theta
        self.n_sim_samples_per_theta = n_sim_samples_per_theta
        self.prior = prior
        self.empirical_bias = empirical_bias

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
        x_sim: Array,
        x_obs: Array,
        theta: Array,
        key: PRNGKeyArray,
        embedding_stats: dict = None,
    ) -> Float[Array, ""]:
        """Compute RVNP loss: importance-weighted variational objective with regularization.

        Implements the RVNP loss function with three components:

        1. **Importance-Weighted ELBO** (via _kl_divergence)
        2. **Shrinkage Prior** (on mean neural network)
        3. **Entropy Regularization** (for full-rank covariance)

        Mathematical Formulation:

            .. math::

                L(\phi,\psi) = \\text{IW-ELBO} + \lambda_{shrinkage} \cdot \|\mu_\\theta(\\theta)\|^2 + \lambda_{entropy} \cdot H(r_\psi)

            where IW-ELBO is computed via the _kl_divergence method.

            where:
                - :math:`q_\phi(\\theta|\hat{x})`: Posterior flow (amortized inference)
                - :math:`r_\psi(\hat{x}|x,\\theta)`: Correction model :math:`N(\hat{x}; \mu_\psi(x,\\theta), \Sigma_\psi(\\theta))`
                - :math:`p_{sim}(x|\\theta)`: Simulator flow (trained separately)
                - :math:`\\theta`: Parameters of interest
                - :math:`x`: Simulated observation
                - :math:`\hat{x}`: Corrected observation

        Args:
            params_flow: Trainable parameters of posterior flow :math:`q_\phi(\\theta|\hat{x})`
            static_flow: Static (non-trainable) parameters of posterior flow
            params_embedding: Trainable parameters of embedding network :math:`f_\omega(x)`
                             (used for high-dimensional observations)
            static_embedding: Static parameters of embedding network
            params_correction: Trainable parameters of correction model :math:`r_\psi(\hat{x}|x,\\theta)`
            static_correction: Static parameters of correction model
            simulator_flow: Pre-trained simulator flow :math:`p_{sim}(x|\\theta)` (complete model,
                           not split into params/static)
            x_sim: Simulated observations from :math:`p_{sim}(x|\\theta)`, shape (batch_size, obs_dim)
            x_obs: Observed data for inference, shape (n_obs, obs_dim)
            theta: Parameters :math:`\\theta`, shape (batch_size, theta_dim)
            key: JAX random key for stochastic sampling
            embedding_stats: Optional dictionary with embedding statistics:
                - 'mean': Normalization mean
                - 'std': Normalization standard deviation

        Returns:
            Scalar loss value = IW-ELBO + shrinkage + entropy

        Notes:
            - IW-ELBO computed via _kl_divergence (importance-weighted variational objective)
            - Shrinkage only applies to NN correction models (mean neural network)
            - Entropy regularization encourages full-rank covariance
        """

        ###########
        ## pull in the parameters and static parameters
        # Normalizing flow (posterior)
        flow = paramax.unwrap(eqx.combine(params_flow, static_flow))
        correction_model = paramax.unwrap(eqx.combine(params_correction, static_correction))
        # number of simulations and number of actual observations in the batch
        batch_size = x_sim.shape[0]
        n_obs = x_obs.shape[0]
        
        # ======== ======== ======== ======== ======== ======== ========
        # ========              EMBEDDING SECTION               ========
        # ======== ======== ======== ======== ======== ======== ========
        if params_embedding is not None:
            embedding = paramax.unwrap(eqx.combine(params_embedding, static_embedding))
            key_embed_sim, key = jax.random.split(key)
            keys_sim = jax.random.split(key_embed_sim, batch_size)
            x_sim_embedded = jax.lax.stop_gradient(
                vmap(lambda x, k: embedding(x, key=k, inference=True))(x_sim[:, jnp.newaxis, :], keys_sim)
            )
            x_sim_processed = (x_sim_embedded - embedding_stats['mean']) / embedding_stats['std']
            key_embed_obs, key = jax.random.split(key)
            keys_obs = jax.random.split(key_embed_obs, n_obs)
            x_obs_embedded = jax.lax.stop_gradient(
                vmap(lambda x, k: embedding(x, key=k, inference=True))(x_obs[:, jnp.newaxis, :], keys_obs)
            )
            x_obs_processed = (x_obs_embedded - embedding_stats['mean']) / embedding_stats['std']
        else:
            x_sim_processed = x_sim
            x_obs_processed = x_obs
        # ======== ======== ======== ======== ======== ======== ========
        # ========           RVNP LOSS COMPONENTS               ========
        # ======== ======== ======== ======== ======== ======== ========

        # Term 1: Importance-Weighted Variational Loss (IW-ELBO)
        key_variational, key = jax.random.split(key)
        reverse = False  # Use forward KL divergence
        variational_loss = self.lambda_variational * self._kl_divergence(
            correction_model, simulator_flow, flow, reverse, x_obs_processed, key_variational,
            n_samples=self.simulator_samples_per_theta, kl_weight=self.lambda_kl, prior_log=self.prior.log_prob
        )

        # Term 2: Shrinkage Prior (only on mean neural network)
        shrinkage_loss = 0.0
        if self.lambda_shrinkage > 0.0:
            shrinkage_loss = self.lambda_shrinkage * self._compute_shrinkage_prior(correction_model, theta)

        # Term 3: Entropy Regularization (full-rank covariance)
        entropy_loss = 0.0
        if self.lambda_entropy > 0.0:
            entropy_loss = self.lambda_entropy * self._compute_entropy_regularization(correction_model, x_sim_processed)

        # Total RVNP Loss
        total_loss = variational_loss + shrinkage_loss + entropy_loss

        return total_loss



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
    def _kl_divergence(
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

    ) -> Float[Array, ""]:
        """
        Sample from simulator model and use soft attention to match to observations.
        
        Process:
        1. Sample x_sim from simulator_flow using theta parameters  
        2. Compute soft attention weights based on proximity to x_obs
        3. Apply attention-weighted contrastive loss
        
        Args:
            correction_model: Correction model r_ψ(x̂|x_sim)
            simulator_flow: Trained simulator p(x|θ)
            x_obs_real: Real observations (n_obs, dim)
            theta: Parameters from training batch (n_sim, theta_dim)
            key: Random key for sampling
            n_samples: Number of samples per theta (default 100)
            
        Returns:
            Simulator sampling attention-based contrastive loss
        """
        n_obs = x_obs_real.shape[0]
                
        # 1. Get theta values (either from training data or sample from posterior)
        # Sample theta from posterior for each observation
        key_theta_sample, key = jax.random.split(key)
        keys_theta = jax.random.split(key_theta_sample, n_obs)






        stick_the_landing = False
        if(stick_the_landing==True):
            def sample_thetas_for_single_obs(x_obs_single, key_single):
                samples=flow.sample(key_single, (self.simulator_samples_per_theta,), condition=x_obs_single)
                params_flow, static_flow = eqx.partition(
                flow,
                eqx.is_inexact_array,
                is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
                )
                flow_temp = paramax.unwrap(eqx.combine(stop_gradient(params_flow), static_flow))
                log_p = flow_temp.log_prob(samples, condition=x_obs_single)
                return samples, log_p
        else:
            def sample_thetas_for_single_obs(x_obs_single, key_single):
                return flow.sample_and_log_prob(key_single, (self.simulator_samples_per_theta,), condition=x_obs_single)


        # Shape: (n_obs, samples_per_theta, theta_dim)
        thetas_sampled,log_p_posterior = jax.vmap(sample_thetas_for_single_obs)(x_obs_real, keys_theta)
        theta_for_sampling = thetas_sampled[:,:,:]

        # Clip theta values to training bounds if enabled
        #theta_for_sampling = self._clip_theta(theta_for_sampling)

        prior_logp=jax.vmap(jax.vmap(prior_log))(theta_for_sampling)
        # Generate keys for simulator sampling: (n_obs, samples_per_theta)
        n_obs_total, samples_per_theta = theta_for_sampling.shape[0], theta_for_sampling.shape[1]
        
        key_sample, key = jax.random.split(key)
        keys_sim_flat = jax.random.split(key_sample, n_obs_total * samples_per_theta)
        keys_sim = keys_sim_flat.reshape(n_obs_total, samples_per_theta)

        # Use configurable n_sim_samples_per_theta instead of hardcoded value
        n_samples = self.n_sim_samples_per_theta
        def sample_from_simulator_single(theta_single, key_single):
            # Sample like theta sampling: key, shape, condition
            return simulator_flow.sample(key_single, (n_samples,), condition=theta_single)
        
        # Sample for all theta using nested vmap: (n_obs, samples_per_theta, n_samples, x_dim)
        x_sim_samples = vmap(vmap(sample_from_simulator_single))(theta_for_sampling, keys_sim)
        n_obs, K_samples = thetas_sampled.shape[0], thetas_sampled.shape[1]
    

        # for single observation: 
        def compute_elbo_single_obs(x_obs_single, theta_samples_single, x_sim_samples_single, log_q_phi_single,log_prior_single):
            #(x_obs_single)=(x_dim)
            # x_sim_samples_single: (n_theta, n_samples, x_dim)
            # log_q_phi_single: (n_theta,)
            # log_prior_single: (n_theta,)
            def compute_terms_single_theta(x_sim_theta_k, log_q_phi_k, log_prior_single_k, theta_single_k):
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
                if(reverse==True):
                    elbo_single_obs_single_theta = jnp.mean(log_p_obs_given_sim) - log_q_phi_k+log_prior_single_k
                    return jnp.mean(elbo_single_obs_single_theta)
                else:
                    elbo_single_obs_single_theta=logsumexp(log_p_obs_given_sim)- jnp.log(x_sim_theta_k.shape[0])
                    return elbo_single_obs_single_theta



                
            
            # Compute for all K theta samples, single here refers to single theta sample
            elbo_single_obs= vmap(compute_terms_single_theta)(x_sim_samples_single, log_q_phi_single, log_prior_single, theta_samples_single)
            # needs to be averaged over the theta dimension
            if reverse==True:
                return jnp.mean(elbo_single_obs)
            else:
                #this was changed!
                #return logsumexp(elbo_single_obs-log_q_phi_single+log_prior_single)- jnp.log(theta_samples_single.shape[0])
                return logsumexp(elbo_single_obs-log_q_phi_single)+log_prior_single.mean()- jnp.log(theta_samples_single.shape[0])
        
        
        # Compute IW ELBO for all observations
        elbo_per_obs = vmap(compute_elbo_single_obs)(
            x_obs_real, thetas_sampled, x_sim_samples, log_p_posterior,prior_logp
        )
        
        # Return negative IW ELBO as loss (we want to maximize ELBO, so minimize negative ELBO)
        elbo_loss = -jnp.mean(elbo_per_obs)
        return elbo_loss


    @eqx.filter_jit
    def _compute_entropy_regularization(self, correction_model, x_sim: Array) -> Float[Array, ""]:
        """
        Compute entropy regularization term for the correction model.
        
        For SimpleCorrectionModel with diagonal covariance Σ:
        H(p(x_obs|x)) = 0.5 * Σ_i log(2πe * σ_i^2)
        
        For diagonal covariance models:
        H(p(x_obs|x)) = 0.5 * Σ_i log(2πe * σ_i^2)
        
        Args:
            correction_model: The correction model
            x_sim: Simulator samples to evaluate entropy at
            
        Returns:
            Negative entropy (regularization loss)
        """
        if isinstance(correction_model, SimpleCorrectionModel):
            # Full covariance case with Cholesky parameterization
            try:
                # Get the Cholesky factor (same for all inputs in SimpleCorrectionModel)
                # Use dummy theta for SimpleCorrectionModel (ignored internally)
                dummy_theta = jnp.zeros((1, 1))  # Dummy theta, will be ignored
                _, L = correction_model(x_sim[:1], dummy_theta)  # Just need one sample to get L
                
                # Entropy for multivariate Gaussian: H = 0.5 * (dim * log(2πe) + 2*log_det(L))
                dim = L.shape[0]
                log_det_L = jnp.sum(jnp.log(jnp.diag(L)))  # log|Σ| = 2*log|L|
                entropy_per_sample = 0.5 * (dim * jnp.log(2 * jnp.pi * jnp.e) + 2 * log_det_L)
                
                # Return negative entropy for regularization (we want to maximize entropy)
                return -entropy_per_sample
                
            except (jax.errors.LinAlgError, AttributeError):
                # Fallback if there are numerical issues
                return 0.0
                
        elif isinstance(correction_model, GlobalCorrectionModel):
            # GlobalCorrectionModel - similar to SimpleCorrectionModel but no theta dependence
            try:
                # Get covariance matrix directly (no theta needed)
                Sigma = correction_model.get_covariance_matrix()
                
                # Entropy for multivariate Gaussian: H = 0.5 * (dim * log(2πe) + log_det(Σ))
                dim = Sigma.shape[0]
                log_det_Sigma = jnp.log(jnp.linalg.det(Sigma + 1e-6 * jnp.eye(dim)))
                entropy_per_sample = 0.5 * (dim * jnp.log(2 * jnp.pi * jnp.e) + log_det_Sigma)
                
                # Return negative entropy for regularization (we want to maximize entropy)
                return -entropy_per_sample
                
            except (jax.errors.LinAlgError, AttributeError):
                # Fallback if there are numerical issues
                return 0.0
        else:
            # Diagonal covariance case - needs theta for neural correction models
            try:
                # For neural correction models, we need theta. Create dummy theta for entropy computation
                if isinstance(correction_model, (DiagonalNeuralCorrectionModel, HybridCorrectionModel, MuHybridCorrectionModel)):
                    # Use zero theta as placeholder for entropy computation 
                    dummy_theta = jnp.zeros((x_sim.shape[0], correction_model.log_diag_net.in_size if hasattr(correction_model, 'log_diag_net') else correction_model.local_cholesky_net.in_size))
                    _, sigma_batch = correction_model(x_sim, dummy_theta)
                else:
                    _, sigma_batch = correction_model(x_sim)
                # Entropy for diagonal Gaussian: H = 0.5 * Σ_i log(2πe * σ_i^2)
                entropy_batch = 0.5 * jnp.sum(jnp.log(2 * jnp.pi * jnp.e * sigma_batch**2 + 1e-8), axis=-1)
                return -jnp.mean(entropy_batch)
            except:
                return 0.0







