"""
Parameter space transformations for RVNP-SBI.

Implements logit/sigmoid transformations to map bounded priors to unbounded space
for smoother optimization.

When bounded uniform priors create jagged loss landscapes due to hard boundaries,
this module provides differentiable transformations to eliminate boundary penalties.

Example usage:
    >>> transform = ParameterTransform(
    ...     theta_min=jnp.array([0.0, 10.0]),
    ...     theta_max=jnp.array([1.0, 20.0]),
    ...     enabled=True
    ... )
    >>> z = jnp.array([[0.0, 0.0]])  # Unbounded space
    >>> theta = transform.inverse(z)  # Transform to bounded space
    >>> theta
    Array([[0.5, 15.0]], dtype=float32)
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array
from typing import Optional


class ParameterTransform(eqx.Module):
    """
    Logit transformation for bounded parameters.

    Transforms θ ∈ [theta_min, theta_max] ↔ z ∈ (-∞, ∞)

    Forward transform (bounded → unbounded):
        z = log((θ - a)/(b - θ))

    Inverse transform (unbounded → bounded):
        θ = a + (b - a) * sigmoid(z)

    Log-determinant Jacobian (for probability adjustment):
        log|det J| = Σᵢ [log(bᵢ - aᵢ) - softplus(zᵢ) - softplus(-zᵢ)]

    This transformation ensures:
    - Smooth gradients throughout parameter space (no boundary penalties)
    - Proper probability conservation via Jacobian adjustment
    - Backward compatibility via enabled flag

    Attributes:
        theta_min: Lower bounds for each parameter dimension (D,)
        theta_max: Upper bounds for each parameter dimension (D,)
        enabled: Whether transformation is active (bool)
    """
    theta_min: Array = eqx.static_field()
    theta_max: Array = eqx.static_field()
    enabled: bool = eqx.static_field()

    def __init__(self, theta_min: Array, theta_max: Array, enabled: bool = False):
        """
        Initialize parameter transformation.

        Args:
            theta_min: Lower bounds (D,)
            theta_max: Upper bounds (D,)
            enabled: Activate transformation (default: False for backward compat)

        Raises:
            AssertionError: If theta_max <= theta_min for any dimension
        """
        self.theta_min = jnp.asarray(theta_min)
        self.theta_max = jnp.asarray(theta_max)
        self.enabled = enabled

        # Validate bounds
        assert jnp.all(theta_max > theta_min), "theta_max must be > theta_min"

    def forward(self, theta: Array) -> Array:
        """
        Transform bounded θ → unbounded z.

        Applies logit transform: z = log((θ - a)/(b - θ))

        Args:
            theta: Parameters in bounded space (..., D)

        Returns:
            z: Parameters in unbounded space (..., D)

        Note:
            If theta has more dimensions than theta_min/max, extra dimensions
            are passed through unchanged (for extensibility).
        """
        if not self.enabled:
            return theta

        # Extract dimension to transform (handle case where theta has more dims)
        D = self.theta_min.shape[0]
        theta_to_transform = theta[..., :D]

        # Normalize to [0, 1]
        a, b = self.theta_min, self.theta_max
        u = (theta_to_transform - a) / (b - a)

        # Clamp to avoid log(0) or log(negative)
        u = jnp.clip(u, 1e-7, 1 - 1e-7)

        # Logit transform
        z = jnp.log(u / (1 - u))

        # If theta has extra dimensions beyond D, keep them unchanged
        if theta.shape[-1] > D:
            z = jnp.concatenate([z, theta[..., D:]], axis=-1)

        return z

    def inverse(self, z: Array) -> Array:
        """
        Transform unbounded z → bounded θ.

        Applies inverse logit (sigmoid): θ = a + (b - a) * sigmoid(z)

        Args:
            z: Parameters in unbounded space (..., D)

        Returns:
            theta: Parameters in bounded space (..., D)

        Note:
            The sigmoid ensures output is always within [a, b] regardless
            of input z (no clipping needed).
        """
        if not self.enabled:
            return z

        # Extract dimension to transform
        D = self.theta_min.shape[0]
        z_to_transform = z[..., :D]

        # Sigmoid to [0, 1]
        u = jax.nn.sigmoid(z_to_transform)

        # Scale to [a, b]
        a, b = self.theta_min, self.theta_max
        theta = a + (b - a) * u

        # If z has extra dimensions beyond D, keep them unchanged
        if z.shape[-1] > D:
            theta = jnp.concatenate([theta, z[..., D:]], axis=-1)

        return theta

    def log_det_jacobian(self, z: Array) -> Array:
        """
        Compute log-determinant of Jacobian dθ/dz.

        Used to adjust log probabilities when transforming variables:
            log p(θ) = log p_z(z) - log|det J|

        The Jacobian for the inverse transform θ = a + (b-a)*sigmoid(z) is:
            dθ/dz = (b-a) * sigmoid(z) * (1 - sigmoid(z))

        Taking the log:
            log|dθ/dz| = log(b-a) - softplus(z) - softplus(-z)

        Args:
            z: Parameters in unbounded space (..., D)

        Returns:
            log_det: Log determinant summed over dimensions (...,)

        Note:
            Uses softplus for numerical stability. For very large |z|,
            softplus saturates gracefully.
        """
        if not self.enabled:
            return jnp.zeros(z.shape[:-1])

        # Extract dimension to transform
        D = self.theta_min.shape[0]
        z_to_transform = z[..., :D]

        # log|det J| = Σᵢ [log(b - a) - softplus(z) - softplus(-z)]
        a, b = self.theta_min, self.theta_max

        log_scale = jnp.log(b - a)  # (D,)
        log_sigmoid_term = jax.nn.softplus(z_to_transform) + jax.nn.softplus(-z_to_transform)

        # Sum over parameter dimensions
        log_det = jnp.sum(log_scale - log_sigmoid_term, axis=-1)

        return log_det

    def inverse_with_log_det(self, z: Array) -> tuple[Array, Array]:
        """
        Combined inverse transform + Jacobian computation.

        More efficient than calling inverse() and log_det_jacobian() separately
        since sigmoid is computed only once.

        Args:
            z: Parameters in unbounded space (..., D)

        Returns:
            theta: Transformed parameters (..., D)
            log_det: Log determinant (...,)

        Example:
            >>> theta, log_det = transform.inverse_with_log_det(z)
            >>> # Adjust posterior probability:
            >>> log_q_theta = log_q_z - log_det
        """
        theta = self.inverse(z)
        log_det = self.log_det_jacobian(z)
        return theta, log_det


def create_identity_transform(theta_dim: int) -> ParameterTransform:
    """
    Create disabled (identity) transform for backward compatibility.

    When parameter transformation is not enabled, this creates a transform
    that acts as the identity function (no transformation applied).

    Args:
        theta_dim: Parameter dimension

    Returns:
        ParameterTransform with enabled=False (acts as identity)

    Example:
        >>> transform = create_identity_transform(3)
        >>> x = jnp.array([[1.0, 2.0, 3.0]])
        >>> transform.inverse(x)
        Array([[1.0, 2.0, 3.0]], dtype=float32)  # Unchanged
    """
    return ParameterTransform(
        theta_min=jnp.zeros(theta_dim),
        theta_max=jnp.ones(theta_dim),
        enabled=False
    )
