"""
Prior distributions for Simulation-Based Inference (SBI).

This module provides a base abstract class for priors and holder classes
for each specific task where the actual log_prob functions will be implemented.
"""

import jax.numpy as jnp
import jax
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray as KeyArray
import abc
from flowjax.flows import masked_autoregressive_flow
from flowjax.train import fit_to_data
import jax.numpy as jnp

from flowjax.distributions import Normal
import os

class BasePrior(eqx.Module):
    """
    Abstract base class for all prior distributions.
    
    Each task-specific prior should inherit from this class and implement
    the log_prob method with the actual prior logic for that task.
    """
    theta_min: Array
    theta_max: Array
    mean: Array
    std: Array
    
    def __init__(self, key: KeyArray, theta_min: Array, theta_max: Array, mean: Array, std: Array):
        """
        Args:
            key: Random key for initialization (may be unused in some priors)
            theta_min: Minimum values for each theta dimension
            theta_max: Maximum values for each theta dimension  
            mean: Mean for denormalization
            std: Std for denormalization
        """
        self.theta_min = jnp.asarray(theta_min)
        self.theta_max = jnp.asarray(theta_max)
        self.mean = jnp.asarray(mean)
        self.std = jnp.asarray(std)
    
    def _compute_soft_uniform_penalty(self, theta: Array, softness: float = 0.01) -> Array:
        """
        Universal soft uniform penalty implementation.
        
        Creates very sharp boundaries at theta_min/theta_max while maintaining differentiability.
        
        Args:
            theta: Parameter values (..., theta_dim) - normalized
            softness: Controls sharpness of boundaries (smaller = sharper)
            
        Returns:
            log_prob: Soft uniform log probability with exponential penalties outside bounds
        """
        # Denormalize theta back to original scale
        theta_out = theta * self.std[:theta.shape[-1]] + self.mean[:theta.shape[-1]]
        
        log_probs = []
        for i in range(min(theta_out.shape[-1], len(self.theta_min))):
            low, high = self.theta_min[i], self.theta_max[i]
            x = theta_out[..., i]
            
            # Normalize to [-1, 1] based on bounds
            center = (low + high) / 2
            width = (high - low) / 2
            x_norm = (x - center) / width
            
            # Exponentially bad penalties outside [-1, 1] range (numerically stable)
            outside_penalty = jnp.maximum(0.0, jnp.abs(x_norm) - 1.0)
            max_exponent = 50.0  # exp(50) ≈ 5×10^21, still huge but finite
            clamped_exponent = jnp.minimum(outside_penalty**2 / softness**2, max_exponent)
            penalty = jnp.exp(clamped_exponent) - 1.0
            
            # Log probability (uniform inside + penalty outside)
            log_prob_i = -penalty - jnp.log(high - low)
            log_probs.append(log_prob_i)
        
        return sum(log_probs)
    
    @abc.abstractmethod
    def log_prob(self, theta: Array) -> Array:
        """
        Compute log probability density of the prior.
        
        Args:
            theta: Parameter values (..., theta_dim)
            
        Returns:
            log_prob: Log probability density (...,)
        """
        pass


class GaussianPrior(BasePrior):
    """
    Prior for Gaussian/normal distribution tasks.
    
    Combines soft uniform bounds with Gaussian prior.
    """
    
    def __init__(self, key: KeyArray, theta_min: Array, theta_max: Array, mean: Array, std: Array):
        super().__init__(key, theta_min, theta_max, mean, std)
    
    def log_prob(self, theta: Array) -> Array:
        """
        Gaussian prior with soft uniform bounds.
        
        Args:
            theta: Parameter values (..., theta_dim)
            
        Returns:
            log_prob: Log probability density (...,)
        """
        # Soft uniform bounds to prevent extreme values
        uniform_penalty = self._compute_soft_uniform_penalty(theta, softness=0.1)
        
        # Gaussian prior on top of uniform bounds
        theta_denorm = theta * self.std[:theta.shape[-1]] + self.mean[:theta.shape[-1]]
        gaussian_prior = jnp.sum(jax.scipy.stats.norm.logpdf(theta_denorm, 0, 5), axis=-1)
        
        return gaussian_prior#uniform_penalty + gaussian_prior


class PendulumPrior(BasePrior):
    """
    Prior for pendulum physics tasks.
    
    Uses the same form as CS task with sharp cutoffs for pendulum parameters.
    """
    
    def __init__(self, key: KeyArray, theta_min: Array, theta_max: Array, mean: Array, std: Array):
        super().__init__(key, theta_min, theta_max, mean, std)
    
    def log_prob(self, theta: Array) -> Array:
        """
        Soft uniform prior with hard cutoffs for pendulum parameters, same form as CS task.
        
        Uses universal soft uniform penalty implementation from BasePrior with sharp boundaries.
        
        Args:
            theta: Parameter values (..., theta_dim) where theta[..., 0] = omega0, theta[..., 1] = A
            
        Returns:
            log_prob: Log probability density (...,)
        """
        # Use universal soft uniform penalty with sharp cutoffs (same as CS task)
        return self._compute_soft_uniform_penalty(theta, softness=0.01)


class SIRPrior(BasePrior):
    """
    Prior for SIR (Susceptible-Infected-Recovered) epidemiological model.
    
    Uses universal soft uniform penalty.
    """
    
    def __init__(self, key: KeyArray, theta_min: Array, theta_max: Array, mean: Array, std: Array):
        super().__init__(key, theta_min, theta_max, mean, std)
    
    def log_prob(self, theta: Array) -> Array:
        """
        SIR prior with epidemiological constraints.
        
        β ~ Uniform(0, 0.5), γ ~ Uniform(0, 0.5), with constraint γ ≤ β
        
        Args:
            theta: Parameter values (..., theta_dim) where theta[..., 0] = β, theta[..., 1] = γ
            
        Returns:
            log_prob: Log probability density (...,)
        """
        # Individual uniform bounds for β and γ on [0, 0.5]
        uniform_penalty = self._compute_soft_uniform_penalty(theta, softness=0.05)
        
        # Epidemiological constraint: γ ≤ β (recovery rate ≤ infection rate)
        if theta.shape[-1] >= 2:
            # Denormalize parameters to original scale
            theta_denorm = theta * self.std[:theta.shape[-1]] + self.mean[:theta.shape[-1]]
            beta = theta_denorm[..., 0]  # infection rate
            gamma = theta_denorm[..., 1]  # recovery rate
            
            # Exponential penalty when γ > β
            constraint_violation = jnp.maximum(0.0, gamma - beta)
            max_exponent = 50.0  # Prevent numerical overflow
            clamped_exponent = jnp.minimum(constraint_violation**2 / 0.01**2, max_exponent)
            constraint_penalty = -(jnp.exp(clamped_exponent) - 1.0)
            
            return uniform_penalty + constraint_penalty
        else:
            return uniform_penalty


class SpectraPrior(BasePrior):
    """
    Prior for spectral analysis tasks.
    
    Uses universal soft uniform penalty.
    """
    flow: eqx.Module
    def __init__(self, key: KeyArray, theta_min: Array, theta_max: Array, mean: Array, std: Array):
        super().__init__(key, theta_min, theta_max, mean, std)
        key, subkey = jax.random.split(jax.random.key(0))
        self.flow = masked_autoregressive_flow(
            subkey,
            base_dist=Normal(jnp.zeros((3)).astype(jnp.float32)),
        )
        self.flow=eqx.tree_deserialise_leaves('Data/spectra_weights/spectra_prior.eqx', self.flow)
    
    def log_prob(self, theta: Array) -> Array:
        """
        Spectra prior with soft uniform bounds.
        
        Args:
            theta: Parameter values (..., theta_dim)
            
        Returns:
            log_prob: Log probability density (...,)
        """
        #return self.flow.log_prob(theta)+self._compute_soft_uniform_penalty(theta, softness=0.01)
        return self._compute_soft_uniform_penalty(theta, softness=0.01)


class CSTaskPrior(BasePrior):
    """
    Prior for CS (Computer Science/Compressed Sensing) tasks.
    
    Uses universal soft uniform penalty with task-specific bounds.
    """
    
    def __init__(self, key: KeyArray, theta_min: Array, theta_max: Array, mean: Array, std: Array):
        super().__init__(key, theta_min, theta_max, mean, std)
    
    def log_prob(self, theta: Array) -> Array:
        """
        Soft uniform prior with hard cutoffs for CS task parameters.

        Uses universal soft uniform penalty implementation from BasePrior.
        This provides differentiable boundary enforcement for bounded parameter spaces.

        When transform_param_space = False (NLPE mode):
            - Soft penalty enforces bounds via exponential penalties
            - Prior is added outside logsumexp in loss function

        When transform_param_space = True (RVNP mode):
            - Soft penalty still computed (doesn't hurt, still differentiable)
            - Prior is added inside logsumexp with Jacobian adjustment

        Args:
            theta: Parameter values (..., theta_dim)

        Returns:
            log_prob: Log probability density (...,)
        """
        # Use universal soft uniform penalty with sharp cutoffs
        return self._compute_soft_uniform_penalty(theta, softness=0.01)



class GaussianMixturePrior(BasePrior):
    """
    Prior for Gaussian mixture model tasks.
    
    Uses universal soft uniform penalty.
    """
    
    def __init__(self, key: KeyArray, theta_min: Array, theta_max: Array, mean: Array, std: Array):
        super().__init__(key, theta_min, theta_max, mean, std)
    
    def log_prob(self, theta: Array) -> Array:
        """
        Gaussian mixture prior with soft uniform bounds.
        
        Args:
            theta: Parameter values (..., theta_dim)
            
        Returns:
            log_prob: Log probability density (...,)
        """
        return self._compute_soft_uniform_penalty(theta, softness=0.05)


def get_prior_from_config(config, key: KeyArray, mean: Array, std: Array, 
                         theta_min: Array = None, theta_max: Array = None) -> BasePrior:
    """
    Factory function to create prior from config, following the same pattern as correction models.
    
    Args:
        config: Configuration object with model.prior_type and optional model.prior_params
        key: Random key for initialization
        mean: Mean for denormalization
        std: Std for denormalization
        theta_min: Minimum values for each theta dimension (optional, task-specific defaults used)
        theta_max: Maximum values for each theta dimension (optional, task-specific defaults used)
        
    Returns:
        Prior instance of the appropriate type
    """
    prior_type = config.sampling.inference_method

    
    if prior_type == 'gaussian_embedding':
        return GaussianPrior(key=key, theta_min=theta_min, theta_max=theta_max, mean=mean, std=std)
    elif prior_type == 'pendulum_task':
        pendulum_theta_min = jnp.array([0.5, 0.5])
        pendulum_theta_max = jnp.array([3.0, 10.0])
        return PendulumPrior(key=key, theta_min=pendulum_theta_min, theta_max=pendulum_theta_max, mean=mean, std=std)
    elif prior_type == 'SIR_task':
        # Always use hardcoded bounds for SIR task (β, γ both uniform on [0, 0.5])
        sir_theta_min = jnp.array([0.0, 0.0])
        sir_theta_max = jnp.array([0.5, 0.5])
        return SIRPrior(key=key, theta_min=sir_theta_min, theta_max=sir_theta_max, mean=mean, std=std)
    elif prior_type in ['Spectra_task', 'spectra_task']:
        theta_min=jnp.array([-2.62497604,  3.4450733 ,  4.00004014])
        theta_max=jnp.array([0.47249011, 4.1476531 , 5.09998652])
        return SpectraPrior(key=key, theta_min=theta_min, theta_max=theta_max, mean=mean, std=std)
    elif prior_type == 'CS_task':
        # Always use hardcoded bounds for CS task, ignore passed theta_min/theta_max
        cs_theta_min = jnp.array([200.0, 3.0, 10.0])
        cs_theta_max = jnp.array([1500.0, 20.0, 20.0])
        return CSTaskPrior(key=key, theta_min=cs_theta_min, theta_max=cs_theta_max, mean=mean, std=std)
    elif prior_type == 'gaussian_mixture':
        return GaussianMixturePrior(key=key, theta_min=theta_min, theta_max=theta_max, mean=mean, std=std)
    else:
        raise ValueError(f"Unknown prior type: {prior_type}. Available: gaussian_embedding, pendulum_task, sir, spectra, CS_task, gaussian_mixture")