import jax.numpy as jnp
import jax
import equinox as eqx
from jaxtyping import Array
from jaxtyping import PRNGKeyArray as KeyArray



class CorrectionModel(eqx.Module):
    """
    Correction model r_ψ(x̂|x) = N(x̂; μ_ψ(x), σ_ψ²(x)I)
    
    Maps from simulator output x to corrected observation x̂ parameters.
    """
    mean_net: eqx.nn.MLP
    log_std_net: eqx.nn.MLP
    
    def __init__(self, key: KeyArray, input_dim: int, hidden_dim: int = 128, depth: int = 3):
        """
        Args:
            key: Random key for parameter initialization
            input_dim: Dimension of input data (embedding dimension)
            hidden_dim: Hidden layer width
            depth: Number of hidden layers
        """
        key1, key2 = jax.random.split(key)
        
        # Network to predict mean μ_ψ(x)
        self.mean_net = eqx.nn.MLP(
            in_size=input_dim,
            out_size=input_dim,
            width_size=hidden_dim,
            depth=depth,
            activation=jax.nn.tanh,
            key=key1
        )
        
        # Network to predict log standard deviation log(σ_ψ(x))
        self.log_std_net = eqx.nn.MLP(
            in_size=input_dim,
            out_size=input_dim,
            width_size=hidden_dim,
            depth=depth,
            activation=jax.nn.tanh,
            key=key2
        )
    
    def __call__(self, x: Array) -> tuple[Array, Array]:
        """
        Args:
            x: Input data (batch_size, input_dim)
            
        Returns:
            mu: Mean parameters μ_ψ(x) (batch_size, input_dim)
            sigma: Standard deviation parameters σ_ψ(x) (batch_size, input_dim)
        """
        mu = self.mean_net(x)
        log_sigma = self.log_std_net(x)
        sigma = jnp.exp(log_sigma) + 1e-6  # Add small epsilon for numerical stability
        
        return mu, sigma
    
    def log_prob(self, x_hat: Array, x: Array) -> Array:
        """
        Compute log probability log r_ψ(x̂|x)
        
        Args:
            x_hat: Corrected observations (batch_size, dim)
            x: Simulator outputs (batch_size, dim)
            
        Returns:
            log_prob: Log probabilities (batch_size,)
        """
        mu, sigma = self(x)
        
        # Gaussian log probability: log N(x̂; μ, σ²I)
        log_prob = -0.5 * jnp.sum(((x_hat - mu) / sigma)**2, axis=-1)
        log_prob -= 0.5 * jnp.sum(jnp.log(2 * jnp.pi * sigma**2), axis=-1)
        
        return log_prob

    @eqx.filter_jit
    def get_correlation_components(self, theta: Array) -> tuple[Array, Array, Array]:
        """
        Extract correlation components from the fully neural covariance matrix.
        
        Args:
            theta: Parameter vector (theta_dim,)
            
        Returns:
            correlations: Full correlation matrix with tanh-normalized off-diagonals (dim, dim)
            diagonal_vars: Diagonal variance components from neural network (dim,)
            off_diag_correlations: Off-diagonal correlation values, flattened (n_off_diag,)
        """
        # Get full covariance matrix
        Sigma = self.get_covariance_matrix(theta)
        
        # Extract diagonal variances
        diagonal_vars = jnp.diag(Sigma)
        
        # Compute correlation matrix: R[i,j] = Σ[i,j] / sqrt(Σ[i,i] * Σ[j,j])
        sqrt_diag = jnp.sqrt(diagonal_vars)
        correlation_matrix = Sigma / jnp.outer(sqrt_diag, sqrt_diag)
        
        # Apply tanh to off-diagonal elements to ensure [-1, +1] bounds
        # Keep diagonal as 1.0 (perfect self-correlation)
        mask = jnp.eye(self.dim)
        off_diag_mask = 1.0 - mask
        
        normalized_correlations = (
            mask * jnp.ones_like(correlation_matrix) +  # Diagonal = 1.0
            off_diag_mask * jnp.tanh(correlation_matrix)  # Off-diagonal = tanh(correlation)
        )
        
        # Extract off-diagonal correlation values (lower triangular, excluding diagonal)
        tril_indices = jnp.tril_indices(self.dim, k=-1)
        off_diag_correlations = normalized_correlations[tril_indices]
        
        return normalized_correlations, diagonal_vars, off_diag_correlations


class DiagonalNeuralCorrectionModel(eqx.Module):
    """
    Neural correction model r_ψ(x̂|x) = N(x̂; x, Σ_ψ(x))
    
    Where Σ_ψ(x) is a diagonal covariance matrix that depends on input x.
    Uses no mean offset (μ_ψ = x, like SimpleCorrectionModel) but learns 
    input-dependent diagonal covariance.
    """
    log_diag_net: eqx.nn.MLP
    
    def __init__(self, key: KeyArray, theta_dim: int, output_dim: int, hidden_dim: int = 128, depth: int = 3):
        """
        Args:
            key: Random key for parameter initialization
            theta_dim: Dimension of parameter theta (conditioning variable)
            output_dim: Dimension of output data (embedding dimension)
            hidden_dim: Hidden layer width
            depth: Number of hidden layers
        """        
        # Network to predict log diagonal variances log(diag(Σ_ψ(theta)))
        self.log_diag_net = eqx.nn.MLP(
            in_size=theta_dim,
            out_size=output_dim,
            width_size=hidden_dim,
            depth=depth,
            activation=jax.nn.tanh,
            key=key
        )
    
    def __call__(self, x: Array, theta: Array) -> tuple[Array, Array]:
        """
        Args:
            x: Input data (output_dim,) or (batch_size, output_dim)
            theta: Parameter theta (theta_dim,) or (batch_size, theta_dim)
            
        Returns:
            mu: Mean parameters (same as input x, no offset)
            diag_values: Diagonal variance parameters σ_ψ²(theta)
        """
        if x.ndim == 1:
            # Single sample case: (output_dim,)
            mu = x
            log_diag = self.log_diag_net(theta)
            diag_values = jnp.exp(log_diag) + 1e-6
            return mu, diag_values
        else:
            # Batch case: (batch_size, output_dim) - use vmap
            single_call = lambda x_single, theta_single: self.__call__(x_single, theta_single)
            mu_batch, diag_batch = jax.vmap(single_call)(x, theta)
            return mu_batch, diag_batch
    
    def log_prob(self, x_hat: Array, x: Array, theta: Array) -> Array:
        """
        Compute log probability log r_ψ(x̂|x,θ) with diagonal covariance
        
        Args:
            x_hat: Corrected observations (dim,) or (batch_size, dim)
            x: Simulator outputs (dim,) or (batch_size, dim)
            theta: Parameter theta (theta_dim,) or (batch_size, theta_dim)
            
        Returns:
            log_prob: Log probability (scalar) or (batch_size,)
        """
        # Handle both single samples and batches
        if x_hat.ndim == 1:
            x_hat = x_hat[None, :]  # Add batch dimension
            x = x[None, :]
            theta = theta[None, :]
            squeeze_output = True
        else:
            squeeze_output = False
            
        mu, diag_values = self(x, theta)  # mu = x (batched), diag_values = learned variances (batched)
        
        # Diagonal Gaussian log probability: log N(x̂; x, Σ) where Σ = diag(diag_values)
        diff = x_hat - mu  # x_hat - x (batch_size, dim)
        log_prob = -0.5 * jnp.sum((diff**2) / diag_values, axis=-1)  # Mahalanobis distance
        log_prob -= 0.5 * jnp.sum(jnp.log(2 * jnp.pi * diag_values), axis=-1)  # Normalization
        
        if squeeze_output:
            log_prob = log_prob[0]  # Remove batch dimension for single samples
            
        return log_prob

    @eqx.filter_jit
    def get_correlation_components(self, theta: Array) -> tuple[Array, Array, Array]:
        """
        Extract correlation components from the fully neural covariance matrix.
        
        Args:
            theta: Parameter vector (theta_dim,)
            
        Returns:
            correlations: Full correlation matrix with tanh-normalized off-diagonals (dim, dim)
            diagonal_vars: Diagonal variance components from neural network (dim,)
            off_diag_correlations: Off-diagonal correlation values, flattened (n_off_diag,)
        """
        # Get full covariance matrix
        Sigma = self.get_covariance_matrix(theta)
        
        # Extract diagonal variances
        diagonal_vars = jnp.diag(Sigma)
        
        # Compute correlation matrix: R[i,j] = Σ[i,j] / sqrt(Σ[i,i] * Σ[j,j])
        sqrt_diag = jnp.sqrt(diagonal_vars)
        correlation_matrix = Sigma / jnp.outer(sqrt_diag, sqrt_diag)
        
        # Apply tanh to off-diagonal elements to ensure [-1, +1] bounds
        # Keep diagonal as 1.0 (perfect self-correlation)
        mask = jnp.eye(self.dim)
        off_diag_mask = 1.0 - mask
        
        normalized_correlations = (
            mask * jnp.ones_like(correlation_matrix) +  # Diagonal = 1.0
            off_diag_mask * jnp.tanh(correlation_matrix)  # Off-diagonal = tanh(correlation)
        )
        
        # Extract off-diagonal correlation values (lower triangular, excluding diagonal)
        tril_indices = jnp.tril_indices(self.dim, k=-1)
        off_diag_correlations = normalized_correlations[tril_indices]
        
        return normalized_correlations, diagonal_vars, off_diag_correlations



class SimpleCorrectionModel(eqx.Module):
    """
    Simple correction model r_ψ(x̂|x) = N(x̂; x, Σ_ψ)
    
    Uses only learned covariance matrix, no mean offset (μ_ψ = 0 always).
    Uses Cholesky parameterization Σ = L @ L.T for guaranteed positive definiteness.
    """
    L_raw: Array  # Shape: (dim, dim) - raw lower triangular parameters
    dim: int      # Dimension of data
    
    def __init__(self, key: KeyArray, dim: int, initial_covariance: Array = None):
        """
        Args:
            key: Random key for parameter initialization
            dim: Dimension of data
            initial_covariance: Optional (dim,) array of initial diagonal variances
        """
        # Initialize Cholesky factor L
        if initial_covariance is not None:
            # Use data-driven initialization: start with diagonal matrix
            L_init = jnp.diag(initial_covariance)
            # Add small random off-diagonal elements
            L_init = L_init + jnp.tril(jax.random.normal(key, (dim, dim)) * jnp.exp(-4), k=-1)
            self.L_raw = L_init
            print(f"🎯 Data-driven initialization: Using empirical covariance {initial_covariance}")
        else:
            # Initialize as near-identity with small random perturbations
            self.L_raw = jnp.eye(dim) + jnp.tril(jax.random.normal(key, (dim, dim)) * (-4), k=-1)
            print("📋 Standard initialization: Using unit variance")
        
        self.dim = dim
    
    def get_cholesky_factor(self) -> Array:
        """
        Get the Cholesky factor L ensuring lower triangular structure and positive diagonal.
        
        Returns:
            L: Lower triangular Cholesky factor (dim, dim)
        """
        # Ensure lower triangular structure
        L = jnp.tril(self.L_raw)
        
        # Ensure positive diagonal using softplus
        diag_entries = jnp.diag(L)
        positive_diag = jax.nn.softplus(diag_entries) + 1e-6  # Ensure > 0
        L = L.at[jnp.diag_indices(self.dim)].set(positive_diag)
        
        return L
    
    def get_covariance_matrix(self) -> Array:
        """
        Construct covariance matrix using Cholesky parameterization Σ = L @ L.T
        
        Returns:
            Sigma: Covariance matrix (dim, dim) - guaranteed positive definite
        """
        L = self.get_cholesky_factor()
        return L @ L.T
    
    def __call__(self, x: Array, theta: Array = None) -> tuple[Array, Array]:
        """
        Args:
            x: Input data (batch_size, dim)
            theta: Parameter theta (ignored for SimpleCorrectionModel, kept for API consistency)
            
        Returns:
            mu: Mean parameters (same as input x, no offset)
            L: Cholesky factor (dim, dim) where Sigma = L @ L.T
        """
        # Mean is just the input (no learnable offset, μ_ψ = 0)
        mu = x
        
        # Return Cholesky factor directly instead of covariance matrix
        L = self.get_cholesky_factor()
        
        return mu, L
    
    def log_prob(self, x_hat: Array, x: Array, theta: Array = None) -> Array:
        """
        Compute log probability log r_ψ(x̂|x) using direct Cholesky factor
        
        Args:
            x_hat: Corrected observations (dim,) or (batch_size, dim)
            x: Simulator outputs (dim,) or (batch_size, dim)
            theta: Parameter theta (ignored for SimpleCorrectionModel, kept for API consistency)
            
        Returns:
            log_prob: Log probability (scalar) or (batch_size,)
        """
        # Handle both single samples and batches
        if x_hat.ndim == 1:
            x_hat = x_hat[None, :]  # Add batch dimension
            x = x[None, :]
            squeeze_output = True
        else:
            squeeze_output = False
            
        # Mean is just the input (no offset)
        mu = x
        
        # Get Cholesky factor directly (no extra decomposition needed!)
        L = self.get_cholesky_factor()
        
        # Solve L y = (x_hat - mu) for y
        diff = x_hat - mu  # Shape: (batch_size, dim)
        if diff.shape[0] == 1:
            # For single sample case
            diff_vec = diff.reshape(-1)  # Flatten to (dim,)
            y_single = jax.scipy.linalg.solve_triangular(L, diff_vec, lower=True)
            y = y_single[None, :]  # Add batch dimension back: (1, dim)
        else:
            # For batch case
            y = jax.scipy.linalg.solve_triangular(L, diff.T, lower=True).T
        
        # Log probability using Cholesky factor
        log_prob = -0.5 * jnp.sum(y**2, axis=-1)  # -0.5 * (x-μ)^T Σ^{-1} (x-μ)
        log_prob -= jnp.sum(jnp.log(jnp.diag(L)))  # -0.5 * log|Σ| = -sum(log(diag(L)))
        log_prob -= 0.5 * self.dim * jnp.log(2 * jnp.pi)  # Normalization constant
        
        if squeeze_output:
            log_prob = log_prob[0]  # Remove batch dimension for single samples
            
        return log_prob

    @eqx.filter_jit
    def get_correlation_components(self, theta: Array) -> tuple[Array, Array, Array]:
        """
        Extract correlation components from the fully neural covariance matrix.
        
        Args:
            theta: Parameter vector (theta_dim,)
            
        Returns:
            correlations: Full correlation matrix with tanh-normalized off-diagonals (dim, dim)
            diagonal_vars: Diagonal variance components from neural network (dim,)
            off_diag_correlations: Off-diagonal correlation values, flattened (n_off_diag,)
        """
        # Get full covariance matrix
        Sigma = self.get_covariance_matrix(theta)
        
        # Extract diagonal variances
        diagonal_vars = jnp.diag(Sigma)
        
        # Compute correlation matrix: R[i,j] = Σ[i,j] / sqrt(Σ[i,i] * Σ[j,j])
        sqrt_diag = jnp.sqrt(diagonal_vars)
        correlation_matrix = Sigma / jnp.outer(sqrt_diag, sqrt_diag)
        
        # Apply tanh to off-diagonal elements to ensure [-1, +1] bounds
        # Keep diagonal as 1.0 (perfect self-correlation)
        mask = jnp.eye(self.dim)
        off_diag_mask = 1.0 - mask
        
        normalized_correlations = (
            mask * jnp.ones_like(correlation_matrix) +  # Diagonal = 1.0
            off_diag_mask * jnp.tanh(correlation_matrix)  # Off-diagonal = tanh(correlation)
        )
        
        # Extract off-diagonal correlation values (lower triangular, excluding diagonal)
        tril_indices = jnp.tril_indices(self.dim, k=-1)
        off_diag_correlations = normalized_correlations[tril_indices]
        
        return normalized_correlations, diagonal_vars, off_diag_correlations



class HybridCorrectionModel(eqx.Module):
    """
    Cholesky-parameterized hybrid correction model: r_ψ(x̂|x) = N(x̂; x, L_hybrid(x) L_hybrid(x)^T)

    Where L_hybrid(x) = L_global + diag(sqrt(σ_local(x))) is the Cholesky factor.
    
    Key advantages:
    - Guaranteed positive definiteness by construction
    - Numerically stable (no matrix additions that could break PD property)
    - Efficient log-determinant computation: log|Σ| = 2 * sum(log(diag(L)))
    """
    # Global Cholesky factor (lower triangular)
    L_global_raw: Array              # (dim, dim) - raw parameters for global Cholesky factor
    
    # Per-sample Cholesky network (local full covariance structure)
    local_cholesky_net: eqx.nn.MLP
    
    dim: int

    def __init__(self, key: KeyArray, theta_dim: int, output_dim: int, hidden_dim: int = 128):
        key1, key2 = jax.random.split(key)

        # Initialize global Cholesky factor parameters
        # Start with near-identity: diagonal entries ≈ 1, off-diagonal ≈ 0
        self.L_global_raw = jax.random.normal(key1, (output_dim, output_dim)) * 0.05
        # Make diagonal entries start around log(1) = 0 for unit variance
        self.L_global_raw = self.L_global_raw.at[jnp.diag_indices(output_dim)].set(0.0)

        # Local per-sample diagonal network (predicts only diagonal elements)
        n_diagonal_params = output_dim  # Only diagonal elements
        self.local_cholesky_net = eqx.nn.MLP(
            in_size=theta_dim, out_size=n_diagonal_params, width_size=hidden_dim,
            depth=2, activation=jax.nn.tanh, key=key2
        )
        self.dim = output_dim

    def _flat_to_diagonal(self, flat_params: Array) -> Array:
        """Convert flat parameter array to diagonal matrix with positive entries.
        
        Args:
            flat_params: Diagonal parameters (dim,)
            
        Returns:
            L_diag: Diagonal matrix with positive diagonal (dim, dim)
        """
        # Apply softplus to ensure positivity
        positive_diag = jax.nn.softplus(flat_params) + 1e-6
        L_diag = jnp.diag(positive_diag)
        return L_diag
    
    def _get_global_cholesky_factor(self) -> Array:
        """Get the global Cholesky factor L_global (off-diagonal only)"""
        # Ensure lower triangular structure
        L_global = jnp.tril(self.L_global_raw)
        
        # Set diagonal to zero (only off-diagonal elements are global)
        L_global = L_global.at[jnp.diag_indices(self.dim)].set(0.0)
        
        return L_global
    
    @eqx.filter_jit
    def get_global_covariance_matrix(self) -> Array:
        """Get only the global covariance component (for regularization)"""
        L_global = self._get_global_cholesky_factor()
        return L_global @ L_global.T

    def _get_hybrid_cholesky_factor(self, theta: Array) -> Array:
        """Get the hybrid Cholesky factor L_hybrid(theta) = L_global(off-diag) + L_local(diag)"""
        # Get global Cholesky factor (off-diagonal only)
        L_global = self._get_global_cholesky_factor()
        
        # Get local diagonal factor from neural network
        local_diagonal_params = self.local_cholesky_net(theta)
        L_local = self._flat_to_diagonal(local_diagonal_params)
        
        # Combine global off-diagonal and local diagonal
        L_hybrid = L_global + L_local
        
        return L_hybrid
    
    @eqx.filter_jit  
    def get_covariance_matrix(self, theta: Array) -> Array:
        """Build hybrid covariance: Σ = L_hybrid(theta) @ L_hybrid(theta).T"""
        L_hybrid = self._get_hybrid_cholesky_factor(theta)
        return L_hybrid @ L_hybrid.T

    @eqx.filter_jit
    def get_correlation_components(self, theta: Array) -> tuple[Array, Array, Array]:
        """
        Extract correlation components from the hybrid covariance matrix.
        
        Args:
            theta: Parameter vector (theta_dim,)
            
        Returns:
            correlations: Full correlation matrix with tanh-normalized off-diagonals (dim, dim)
            diagonal_vars: Diagonal variance components from neural network (dim,)
            off_diag_correlations: Off-diagonal correlation values, flattened (n_off_diag,)
        """
        # Get full covariance matrix
        Sigma = self.get_covariance_matrix(theta)
        
        # Extract diagonal variances
        diagonal_vars = jnp.diag(Sigma)
        
        # Compute correlation matrix: R[i,j] = Σ[i,j] / sqrt(Σ[i,i] * Σ[j,j])
        sqrt_diag = jnp.sqrt(diagonal_vars)
        correlation_matrix = Sigma / jnp.outer(sqrt_diag, sqrt_diag)
        
        # Apply tanh to off-diagonal elements to ensure [-1, +1] bounds
        # Keep diagonal as 1.0 (perfect self-correlation)
        mask = jnp.eye(self.dim)
        off_diag_mask = 1.0 - mask
        
        normalized_correlations = (
            mask * jnp.ones_like(correlation_matrix) +  # Diagonal = 1.0
            off_diag_mask * jnp.tanh(correlation_matrix)  # Off-diagonal = tanh(correlation)
        )
        
        # Extract off-diagonal correlation values (lower triangular, excluding diagonal)
        tril_indices = jnp.tril_indices(self.dim, k=-1)
        off_diag_correlations = normalized_correlations[tril_indices]
        
        return normalized_correlations, diagonal_vars, off_diag_correlations

    @eqx.filter_jit
    def __call__(self, x: Array, theta: Array) -> tuple[Array, Array]:
        """
        Args:
            x: Input data (dim,) or (batch_size, dim)
            theta: Parameter theta (theta_dim,) or (batch_size, theta_dim)
            
        Returns:
            mu: Mean parameters (same as input x, no offset)
            Sigma: Hybrid covariance matrix (dim, dim) or (batch_size, dim, dim)
        """
        # Handle both single samples and batches
        if x.ndim == 1:
            # Single sample case: (dim,)
            mu = x
            Sigma = self.get_covariance_matrix(theta)
            return mu, Sigma
        else:
            # Batch case: (batch_size, dim) - optimized vmap
            single_cov = jax.vmap(self.get_covariance_matrix)
            Sigma_batch = single_cov(theta)
            mu = x  # Mean is just the input (no offset)
            return mu, Sigma_batch

    @eqx.filter_jit
    def log_prob(self, x_hat: Array, x: Array, theta: Array) -> Array:
        """
        Compute log probability log r_ψ(x̂|x,θ) using direct Cholesky factors
        
        Much more numerically stable - no matrix inversions or extra Cholesky decompositions!
        
        Args:
            x_hat: Corrected observations (dim,) or (batch_size, dim)
            x: Simulator outputs (dim,) or (batch_size, dim)
            theta: Parameter theta (theta_dim,) or (batch_size, theta_dim)
            
        Returns:
            log_prob: Log probability (scalar) or (batch_size,)
        """
        # Handle both single samples and batches
        if x_hat.ndim == 1:
            x_hat = x_hat[None, :]  # Add batch dimension
            x = x[None, :]
            theta = theta[None, :]
            squeeze_output = True
        else:
            squeeze_output = False

        # Mean is just the input x (no offset in hybrid model)
        mu = x
        
        if x.shape[0] == 1:
            # Single sample case - more efficient
            L_hybrid = self._get_hybrid_cholesky_factor(theta[0])
            
            diff = x_hat[0] - mu[0]  # (dim,)
            y = jax.scipy.linalg.solve_triangular(L_hybrid, diff, lower=True)
            
            log_prob = -0.5 * jnp.sum(y**2)
            log_prob -= jnp.sum(jnp.log(jnp.diag(L_hybrid)))  # log det = sum(log(diag(L)))
            log_prob -= 0.5 * self.dim * jnp.log(2 * jnp.pi)
            
        else:
            # Batch case - vectorized computation
            batch_size = x.shape[0]
            
            # Vectorized Cholesky factor computation
            L_hybrid_fn = jax.vmap(self._get_hybrid_cholesky_factor)
            L_batch = L_hybrid_fn(theta)  # (batch_size, dim, dim)
            
            # Vectorized solve
            diff = x_hat - mu  # (batch_size, dim)
            y_batch = jax.vmap(
                lambda L, d: jax.scipy.linalg.solve_triangular(L, d, lower=True)
            )(L_batch, diff)
            
            # Vectorized log probability computation
            log_prob = -0.5 * jnp.sum(y_batch**2, axis=1)  # (batch_size,)
            log_prob -= jnp.sum(jnp.log(jnp.diagonal(L_batch, axis1=1, axis2=2)), axis=1)  # log det
            log_prob -= 0.5 * self.dim * jnp.log(2 * jnp.pi)  # Normalization
        
        if squeeze_output and log_prob.ndim > 0:
            log_prob = log_prob[0]  # Remove batch dimension for single samples
            
        return log_prob

    @eqx.filter_jit
    def get_correlation_components(self, theta: Array) -> tuple[Array, Array, Array]:
        """
        Extract correlation components from the fully neural covariance matrix.
        
        Args:
            theta: Parameter vector (theta_dim,)
            
        Returns:
            correlations: Full correlation matrix with tanh-normalized off-diagonals (dim, dim)
            diagonal_vars: Diagonal variance components from neural network (dim,)
            off_diag_correlations: Off-diagonal correlation values, flattened (n_off_diag,)
        """
        # Get full covariance matrix
        Sigma = self.get_covariance_matrix(theta)
        
        # Extract diagonal variances
        diagonal_vars = jnp.diag(Sigma)
        
        # Compute correlation matrix: R[i,j] = Σ[i,j] / sqrt(Σ[i,i] * Σ[j,j])
        sqrt_diag = jnp.sqrt(diagonal_vars)
        correlation_matrix = Sigma / jnp.outer(sqrt_diag, sqrt_diag)
        
        # Apply tanh to off-diagonal elements to ensure [-1, +1] bounds
        # Keep diagonal as 1.0 (perfect self-correlation)
        mask = jnp.eye(self.dim)
        off_diag_mask = 1.0 - mask
        
        normalized_correlations = (
            mask * jnp.ones_like(correlation_matrix) +  # Diagonal = 1.0
            off_diag_mask * jnp.tanh(correlation_matrix)  # Off-diagonal = tanh(correlation)
        )
        
        # Extract off-diagonal correlation values (lower triangular, excluding diagonal)
        tril_indices = jnp.tril_indices(self.dim, k=-1)
        off_diag_correlations = normalized_correlations[tril_indices]
        
        return normalized_correlations, diagonal_vars, off_diag_correlations


class FullNeuralCorrectionModel(eqx.Module):
    """
    Fully neural correction model: r_ψ(x̂|x,θ) = N(x̂; x, L_neural(θ) L_neural(θ)^T)
    
    Where L_neural(θ) is a full lower triangular Cholesky factor that depends entirely on theta.
    No global/fixed components - everything is learned through the neural network.
    
    Key advantages:
    - Maximum flexibility: all correlations and variances can depend on theta
    - Guaranteed positive definiteness through Cholesky parameterization
    - Fully theta-dependent covariance structure
    """
    cholesky_net: eqx.nn.MLP  # Neural network: theta → all Cholesky parameters
    dim: int
    
    def __init__(self, key: KeyArray, theta_dim: int, output_dim: int, hidden_dim: int = 128, depth: int = 3):
        """
        Args:
            key: Random key for parameter initialization
            theta_dim: Dimension of parameter theta (conditioning variable)
            output_dim: Dimension of output data (embedding dimension)
            hidden_dim: Hidden layer width
            depth: Number of hidden layers
        """
        # Network to predict all lower triangular Cholesky elements
        n_cholesky_params = output_dim * (output_dim + 1) // 2  # Number of lower triangular elements
        self.cholesky_net = eqx.nn.MLP(
            in_size=theta_dim, out_size=n_cholesky_params, width_size=hidden_dim,
            depth=depth, activation=jax.nn.tanh, key=key
        )
        self.dim = output_dim
    
    def _flat_to_lower_triangular(self, flat_params: Array) -> Array:
        """Convert flat parameter array to lower triangular matrix with positive diagonal.
        
        Args:
            flat_params: Flattened parameters (n_cholesky_params,)
            
        Returns:
            L: Lower triangular matrix with positive diagonal (dim, dim)
        """
        L = jnp.zeros((self.dim, self.dim))
        
        # Fill lower triangular part
        idx = 0
        for i in range(self.dim):
            for j in range(i + 1):  # j <= i (lower triangular)
                L = L.at[i, j].set(flat_params[idx])
                idx += 1
        
        # Ensure positive diagonal entries using softplus
        diag_entries = jnp.diag(L)
        positive_diag = jax.nn.softplus(diag_entries) + 1e-6  # Ensure > 0
        L = L.at[jnp.diag_indices(self.dim)].set(positive_diag)
        
        return L
    
    @eqx.filter_jit
    def get_cholesky_factor(self, theta: Array) -> Array:
        """Get the fully neural Cholesky factor L_neural(theta)"""
        flat_params = self.cholesky_net(theta)
        return self._flat_to_lower_triangular(flat_params)
    
    @eqx.filter_jit  
    def get_covariance_matrix(self, theta: Array) -> Array:
        """Build neural covariance: Σ = L_neural(theta) @ L_neural(theta).T"""
        L_neural = self.get_cholesky_factor(theta)
        return L_neural @ L_neural.T

    @eqx.filter_jit
    def __call__(self, x: Array, theta: Array) -> tuple[Array, Array]:
        """
        Args:
            x: Input data (dim,) or (batch_size, dim)
            theta: Parameter theta (theta_dim,) or (batch_size, theta_dim)
            
        Returns:
            mu: Mean parameters (same as input x, no offset)
            Sigma: Neural covariance matrix (dim, dim) or (batch_size, dim, dim)
        """
        # Handle both single samples and batches
        if x.ndim == 1:
            # Single sample case: (dim,)
            mu = x
            Sigma = self.get_covariance_matrix(theta)
            return mu, Sigma
        else:
            # Batch case: (batch_size, dim) - optimized vmap
            single_cov = jax.vmap(self.get_covariance_matrix)
            Sigma_batch = single_cov(theta)
            mu = x  # Mean is just the input (no offset)
            return mu, Sigma_batch

    @eqx.filter_jit
    def log_prob(self, x_hat: Array, x: Array, theta: Array) -> Array:
        """
        Compute log probability log r_ψ(x̂|x,θ) using direct Cholesky factors
        
        Args:
            x_hat: Corrected observations (dim,) or (batch_size, dim)
            x: Simulator outputs (dim,) or (batch_size, dim)
            theta: Parameter theta (theta_dim,) or (batch_size, theta_dim)
            
        Returns:
            log_prob: Log probability (scalar) or (batch_size,)
        """
        # Handle both single samples and batches
        if x_hat.ndim == 1:
            x_hat = x_hat[None, :]  # Add batch dimension
            x = x[None, :]
            theta = theta[None, :]
            squeeze_output = True
        else:
            squeeze_output = False
            
        # Mean is just the input x (no offset in neural model)
        mu = x
        
        if x.shape[0] == 1:
            # Single sample case - more efficient
            L_neural = self.get_cholesky_factor(theta[0])
            diff = (x_hat - mu)[0]  # Remove batch dimension
            y = jax.scipy.linalg.solve_triangular(L_neural, diff, lower=True)
            
            log_prob = -0.5 * jnp.sum(y**2)  # -0.5 * (x-μ)^T Σ^{-1} (x-μ)
            log_prob -= jnp.sum(jnp.log(jnp.diag(L_neural)))  # -0.5 * log|Σ|
            log_prob -= 0.5 * self.dim * jnp.log(2 * jnp.pi)  # Normalization
            log_prob = jnp.array([log_prob])  # Make it batch-like for consistency
        else:
            # Batch case: (batch_size, dim)
            batch_size = x.shape[0]
            
            # Vectorized Cholesky factor computation
            L_neural_fn = jax.vmap(self.get_cholesky_factor)
            L_batch = L_neural_fn(theta)  # (batch_size, dim, dim)
            
            # Vectorized solve
            diff = x_hat - mu  # (batch_size, dim)
            y_batch = jax.vmap(
                lambda L, d: jax.scipy.linalg.solve_triangular(L, d, lower=True)
            )(L_batch, diff)
            
            # Vectorized log probability computation
            log_prob = -0.5 * jnp.sum(y_batch**2, axis=1)  # (batch_size,)
            log_prob -= jnp.sum(jnp.log(jnp.diagonal(L_batch, axis1=1, axis2=2)), axis=1)  # log det
            log_prob -= 0.5 * self.dim * jnp.log(2 * jnp.pi)  # Normalization
        
        if squeeze_output and log_prob.ndim > 0:
            log_prob = log_prob[0]  # Remove batch dimension for single samples
            
        return log_prob

    @eqx.filter_jit
    def get_correlation_components(self, theta: Array) -> tuple[Array, Array, Array]:
        """
        Extract correlation components from the fully neural covariance matrix.
        
        Args:
            theta: Parameter vector (theta_dim,)
            
        Returns:
            correlations: Full correlation matrix with tanh-normalized off-diagonals (dim, dim)
            diagonal_vars: Diagonal variance components from neural network (dim,)
            off_diag_correlations: Off-diagonal correlation values, flattened (n_off_diag,)
        """
        # Get full covariance matrix
        Sigma = self.get_covariance_matrix(theta)
        
        # Extract diagonal variances
        diagonal_vars = jnp.diag(Sigma)
        
        # Compute correlation matrix: R[i,j] = Σ[i,j] / sqrt(Σ[i,i] * Σ[j,j])
        sqrt_diag = jnp.sqrt(diagonal_vars)
        correlation_matrix = Sigma / jnp.outer(sqrt_diag, sqrt_diag)
        
        # Apply tanh to off-diagonal elements to ensure [-1, +1] bounds
        # Keep diagonal as 1.0 (perfect self-correlation)
        mask = jnp.eye(self.dim)
        off_diag_mask = 1.0 - mask
        
        normalized_correlations = (
            mask * jnp.ones_like(correlation_matrix) +  # Diagonal = 1.0
            off_diag_mask * jnp.tanh(correlation_matrix)  # Off-diagonal = tanh(correlation)
        )
        
        # Extract off-diagonal correlation values (lower triangular, excluding diagonal)
        tril_indices = jnp.tril_indices(self.dim, k=-1)
        off_diag_correlations = normalized_correlations[tril_indices]
        
        return normalized_correlations, diagonal_vars, off_diag_correlations


class MuHybridCorrectionModel(eqx.Module):
    """Mu-hybrid correction model with neural mean and covariance.

    The primary RVNP correction model that learns both a parameter-dependent mean shift
    and a hybrid covariance structure to correct for model misspecification.

    Mathematical Formulation:
        The correction distribution is defined as:

        .. math::

            r_\psi(\hat{x}|x,\\theta) = N(\hat{x}; \mu(x,\\theta), \Sigma(\\theta))

    Mean Function:
        .. math::

            \mu(x,\\theta) = x + \mu_{global} + \mu_\\theta(\\theta)

        where:
            - :math:`x`: Original observation (identity component preserves simulator output)
            - :math:`\mu_{global}`: Learnable global bias correction vector (output_dim,)
            - :math:`\mu_\\theta(\\theta)`: Neural network producing parameter-dependent shift

        The mean shift decomposes misspecification into:
            1. **Constant bias** (:math:`\mu_{global}`): Systematic errors independent of :math:`\\theta`
            2. **Parameter-dependent bias** (:math:`\mu_\\theta(\\theta)`): Errors varying with :math:`\\theta`

    Covariance Function:
        .. math::

            \Sigma(\\theta) = L_{hybrid}(\\theta) L_{hybrid}(\\theta)^T

            L_{hybrid}(\\theta) = L_{global} + diag(\sigma_{local}(\\theta))

        where:
            - :math:`L_{global}`: Global lower-triangular Cholesky factor (learned, fixed across :math:`\\theta`)
            - :math:`\sigma_{local}(\\theta)`: Neural network outputs per-dimension scaling factors
            - :math:`diag(\cdot)`: Diagonal matrix constructor

        The hybrid structure combines:
            1. **Global correlations** (:math:`L_{global}`): Captures consistent inter-dimensional relationships
            2. **Local scaling** (:math:`diag(\sigma_{local}(\\theta))`): Adjusts uncertainty per :math:`\\theta`

    Shrinkage Prior:
        The model is regularized via:

        .. math::

            L_{shrinkage} = \lambda_{shrinkage} \cdot E_\\theta[\|\mu_\\theta(\\theta)\|^2]

        This encourages the neural mean shift to remain small when misspecification is minimal,
        preventing overfitting. The global bias :math:`\mu_{global}` is NOT penalized, allowing
        it to capture persistent systematic errors.

    Attributes:
        L_global_raw: Raw parameters for global Cholesky factor (output_dim, output_dim)
        local_cholesky_net: MLP mapping :math:`\\theta` to local diagonal scaling (output_dim,)
        mu_global: Global mean shift vector (output_dim,)
        mean_shift_net: MLP mapping :math:`\\theta` to parameter-dependent mean shift (output_dim,)
        dim: Output dimension

    Key Methods:
        get_mean_shift(theta):
            Returns :math:`\mu_{global} + \mu_\\theta(\\theta)` (total mean shift excluding x)

        get_mean_shift_magnitude(theta):
            Returns :math:`\|\mu_\\theta(\\theta)\|^2` for shrinkage regularization

        get_covariance_matrix(theta):
            Returns :math:`\Sigma(\\theta) = L_{hybrid}(\\theta) L_{hybrid}(\\theta)^T`

        log_prob(x_hat, x, theta):
            Computes :math:`\log r_\psi(\hat{x}|x,\\theta)` (Gaussian log probability)

        sample(key, x, theta):
            Samples :math:`\hat{x} \sim r_\psi(\hat{x}|x,\\theta)`

    Mathematical Notation:
        - :math:`\\theta`: Parameters of interest (theta_dim,)
        - :math:`x`: Original simulated observation (output_dim,)
        - :math:`\hat{x}`: Corrected observation (output_dim,)
        - :math:`\psi`: Correction model parameters {:math:`\mu_{global}`, weights of :math:`\mu_\\theta`,
                       :math:`L_{global}`, weights of :math:`\sigma_{local}`}
        - :math:`\phi`: Posterior flow parameters
        - :math:`q_\phi(\\theta|\hat{x})`: Posterior distribution

    Use Cases:
        **RVNP-mu_hybrid (PRIMARY METHOD)**:
            - Significant model misspecification with parameter-dependent errors
            - Unknown bias structure (both constant and parameter-varying components)
            - When RVNP-simple (diagonal covariance) is insufficient
            - Example: Simulator missing friction term that varies with mass parameter

        **Not Recommended When**:
            - Well-specified simulator (use NPE instead)
            - Minimal misspecification (use RVNP-simple instead)
            - Very limited training data (risk of overfitting)

    Example:
        >>> key = jax.random.PRNGKey(0)
        >>> theta_dim = 3
        >>> output_dim = 10
        >>> model = MuHybridCorrectionModel(key, theta_dim, output_dim, hidden_dim=128)
        >>>
        >>> # Get mean shift for specific theta
        >>> theta = jnp.array([0.5, 1.0, -0.3])
        >>> mean_shift = model.get_mean_shift(theta)
        >>>
        >>> # Compute shrinkage penalty
        >>> magnitude = model.get_mean_shift_magnitude(theta)
        >>>
        >>> # Sample corrected observation
        >>> x = jnp.array([1.0, 2.0, ...])  # Original observation
        >>> x_hat = model.sample(key, x, theta)

    Notes:
        - Cholesky decomposition ensures positive-definite covariance
        - Softplus activation on diagonal entries guarantees positivity
        - Global mean shift captures simulator bias (e.g., wrong physical constant)
        - Neural mean shift captures parameter-specific errors
        - Hybrid covariance balances expressiveness and sample efficiency
        - Training uses SimplifiedPosteriorLoss with shrinkage prior

    See Also:
        - SimpleCorrectionModel: Simpler baseline with fixed diagonal covariance
        - HybridCorrectionModel: Neural covariance only (deprecated, no mean shift)
        - FullNeuralCorrectionModel: Fully neural (experimental, may overfit)
    """
    # Global Cholesky factor (lower triangular) - inherited from hybrid
    L_global_raw: Array              # (dim, dim) - raw parameters for global Cholesky factor
    
    # Per-sample Cholesky network (local full covariance structure) - inherited from hybrid
    local_cholesky_net: eqx.nn.MLP
    
    # New: Global mean shift parameter
    mu_global: Array                 # (dim,) - learnable global bias correction
    
    # New: Theta-dependent mean shift network
    mean_shift_net: eqx.nn.MLP       # Neural network: θ -> μ_θ(θ)
    
    dim: int
    
    def __init__(self, key: KeyArray, theta_dim: int, output_dim: int, hidden_dim: int = 128):
        """
        Args:
            key: Random key for parameter initialization
            theta_dim: Dimension of parameter θ
            output_dim: Dimension of output data x
            hidden_dim: Hidden layer width for networks
        """
        key1, key2, key3, key4 = jax.random.split(key, 4)
        
        # Initialize global Cholesky factor (same as HybridCorrectionModel)
        self.L_global_raw = jax.random.normal(key1, (output_dim, output_dim)) * 0.05
        self.L_global_raw = self.L_global_raw.at[jnp.diag_indices(output_dim)].set(0.0)
        
        # Local per-sample diagonal network (same as HybridCorrectionModel)
        n_diagonal_params = output_dim  # Only diagonal elements
        self.local_cholesky_net = eqx.nn.MLP(
            in_size=theta_dim, out_size=n_diagonal_params, width_size=hidden_dim,
            depth=2, activation=jax.nn.tanh, key=key2
        )
        
        # NEW: Initialize global mean shift parameter (small random values)
        self.mu_global = jax.random.normal(key3, (output_dim,)) * 0.01
        
        # NEW: Theta-dependent mean shift network
        self.mean_shift_net = eqx.nn.MLP(
            in_size=theta_dim, out_size=output_dim, width_size=hidden_dim,
            depth=2, activation=jax.nn.tanh, key=key4
        )
        
        self.dim = output_dim
    
    def _flat_to_diagonal(self, flat_params: Array) -> Array:
        """Convert flat parameter array to diagonal matrix with positive entries."""
        positive_diag = jax.nn.softplus(flat_params) + 1e-6
        L_diag = jnp.diag(positive_diag)
        return L_diag
    
    def _get_global_cholesky_factor(self) -> Array:
        """Get global Cholesky factor from raw parameters."""
        L_global = jnp.tril(self.L_global_raw)  # Ensure lower triangular
        # Make diagonal entries positive via softplus
        diag_mask = jnp.eye(self.dim)
        off_diag_mask = 1.0 - diag_mask
        diag_part = diag_mask * jax.nn.softplus(jnp.diag(self.L_global_raw)) + 1e-6
        off_diag_part = off_diag_mask * L_global
        return jnp.diag(diag_part) + off_diag_part
    
    def _get_hybrid_cholesky_factor(self, theta: Array) -> Array:
        """Get hybrid Cholesky factor L_hybrid(θ) = L_global + diag(sqrt(σ_local(θ)))."""
        L_global = self._get_global_cholesky_factor()
        
        # Get local diagonal scaling from theta
        local_diagonal_params = self.local_cholesky_net(theta)
        L_diagonal = self._flat_to_diagonal(local_diagonal_params)
        
        # Hybrid combination: global structure + local diagonal scaling
        L_hybrid = L_global + L_diagonal
        return L_hybrid
    
    def get_covariance_matrix(self, theta: Array) -> Array:
        """Compute hybrid covariance matrix Σ = L_hybrid L_hybrid^T."""
        L_hybrid = self._get_hybrid_cholesky_factor(theta)
        Sigma = L_hybrid @ L_hybrid.T
        return Sigma
    
    def get_mean_shift(self, theta: Array) -> Array:
        """Compute combined mean shift μ_global + μ_θ(θ)."""
        mu_theta = self.mean_shift_net(theta)
        return self.mu_global + mu_theta

    def get_mean_shift_magnitude(self, theta: Array) -> Array:
        """Compute squared magnitude of neural mean shift for shrinkage prior.

        Returns :math:`\|\mu_\\theta(\\theta)\|^2` where :math:`\mu_\\theta` is the neural network
        component of the mean shift. This quantity is used in the shrinkage regularization term
        of the RVNP loss function.

        Mathematical Formulation:
            The shrinkage loss term is:

            .. math::

                L_{shrinkage} = \lambda_{shrinkage} \cdot E_\\theta[\|\mu_\\theta(\\theta)\|^2]

            where :math:`\lambda_{shrinkage}` is specified in config.model.lambda_shrinkage.

        Regularization Purpose:
            The shrinkage prior encourages the neural mean shift :math:`\mu_\\theta(\\theta)` to
            remain close to zero when model misspecification is minimal. This prevents overfitting
            and ensures the correction model only learns meaningful parameter-dependent biases.

        Key Properties:
            - **Only the neural component** :math:`\mu_\\theta(\\theta)` is penalized
            - **Global bias NOT penalized**: :math:`\mu_{global}` can learn persistent systematic errors
            - **Parameter-dependent**: Penalty varies with :math:`\\theta`, allowing selective correction
            - **Encourages sparsity**: Small :math:`\lambda_{shrinkage}` → more correction allowed
                                     Large :math:`\lambda_{shrinkage}` → forces correction toward identity

        Args:
            theta: Parameter vector, shape (theta_dim,) for single sample
                   or (batch_size, theta_dim) for batch

        Returns:
            Squared L2 norm:
                - Scalar value if theta is 1D (single sample)
                - Array of shape (batch_size,) if theta is 2D (batch)

        Implementation:
            For single theta:
                :math:`\|\mu_\\theta(\\theta)\|^2 = \sum_{i=1}^{d} \mu_{\\theta,i}(\\theta)^2`

            For batch of thetas:
                :math:`[\|\mu_\\theta(\\theta_1)\|^2, \ldots, \|\mu_\\theta(\\theta_B)\|^2]`

        Example:
            >>> model = MuHybridCorrectionModel(key, theta_dim=3, output_dim=10)
            >>> theta = jnp.array([0.5, 1.0, -0.3])
            >>> magnitude = model.get_mean_shift_magnitude(theta)
            >>> # Use in loss: loss += lambda_shrinkage * magnitude

        Notes:
            - Called during SimplifiedPosteriorLoss computation (losses.py:576-613)
            - Computed via vmap over batch in training loop
            - Gradient flows through to mean_shift_net parameters
            - Does NOT include the identity component :math:`x` or global bias :math:`\mu_{global}`
            - Typical values: magnitude ∈ [0.001, 10.0] depending on misspecification severity

        See Also:
            - SimplifiedPosteriorLoss: Uses this method for shrinkage regularization
            - get_mean_shift(): Returns full mean shift including global bias
            - HybridCorrectionModel: No neural mean shift, so no shrinkage on mean
        """
        mu_theta = self.mean_shift_net(theta)

        if mu_theta.ndim == 1:
            # Single sample
            return jnp.sum(mu_theta**2)
        else:
            # Batch
            return jnp.sum(mu_theta**2, axis=-1)
    
    def __call__(self, x: Array, theta: Array) -> tuple[Array, Array]:
        """
        Args:
            x: Input data (dim,) or (batch_size, dim)
            theta: Parameter theta (theta_dim,) or (batch_size, theta_dim)
            
        Returns:
            mu: Mean parameters x + μ_global + μ_θ(θ) (with global and theta-dependent shifts)
            Sigma: Hybrid covariance matrix (dim, dim) or (batch_size, dim, dim)
        """
        # Handle both single samples and batches
        if x.ndim == 1:
            # Single sample case: (dim,)
            mean_shift = self.get_mean_shift(theta)
            mu = x + mean_shift  # Add theta-dependent shift
            Sigma = self.get_covariance_matrix(theta)
            return mu, Sigma
        else:
            # Batch case: (batch_size, dim)
            single_mean_shift = jax.vmap(self.get_mean_shift)
            mean_shifts = single_mean_shift(theta)  # (batch_size, dim)
            mu = x + mean_shifts  # Add theta-dependent shifts
            
            single_cov = jax.vmap(self.get_covariance_matrix)
            Sigma_batch = single_cov(theta)
            return mu, Sigma_batch
    
    def log_prob(self, x_hat: Array, x: Array, theta: Array) -> Array:
        """
        Compute log probability log r_ψ(x̂|x,θ) with global and theta-dependent mean shifts.
        
        Args:
            x_hat: Corrected observations (batch_size, dim) or (dim,)
            x: Simulator outputs (batch_size, dim) or (dim,)
            theta: Parameters (batch_size, theta_dim) or (theta_dim,)
            
        Returns:
            log_prob: Log probabilities (batch_size,) or scalar
        """
        mu, Sigma = self(x, theta)
        
        # Handle both single samples and batches
        squeeze_output = False
        if x_hat.ndim == 1:
            x_hat = x_hat[None, :]  # Add batch dimension
            mu = mu[None, :]
            if Sigma.ndim == 2:
                Sigma = Sigma[None, :, :]
            squeeze_output = True
        
        batch_size = x_hat.shape[0]
        
        if batch_size == 1:
            # Single sample case - direct computation
            L = jnp.linalg.cholesky(Sigma[0] + 1e-6 * jnp.eye(self.dim))
            diff_vec = (x_hat[0] - mu[0]).reshape(-1)
            y_single = jax.scipy.linalg.solve_triangular(L, diff_vec, lower=True)
            
            log_prob = -0.5 * jnp.sum(y_single**2)
            log_prob -= jnp.sum(jnp.log(jnp.diag(L)))
            log_prob -= 0.5 * self.dim * jnp.log(2 * jnp.pi)
            log_prob = jnp.array([log_prob])
        else:
            # Batch case - vectorized computation
            L_batch = jax.vmap(lambda S: jnp.linalg.cholesky(S + 1e-6 * jnp.eye(self.dim)))(Sigma)
            diff = x_hat - mu  # (batch_size, dim)
            
            def solve_single(L_i, diff_i):
                return jax.scipy.linalg.solve_triangular(L_i, diff_i, lower=True)
            
            y_batch = jax.vmap(solve_single)(L_batch, diff)
            
            log_prob = -0.5 * jnp.sum(y_batch**2, axis=-1)
            log_prob -= jnp.sum(jnp.log(jnp.diagonal(L_batch, axis1=-2, axis2=-1)), axis=-1)
            log_prob -= 0.5 * self.dim * jnp.log(2 * jnp.pi)
        
        if squeeze_output:
            log_prob = log_prob[0]
            
        return log_prob
    
    @eqx.filter_jit
    def get_correlation_components(self, theta: Array) -> tuple[Array, Array, Array]:
        """
        Extract correlation components from the hybrid covariance matrix.
        
        Args:
            theta: Parameter vector (theta_dim,)
            
        Returns:
            correlations: Full correlation matrix with tanh-normalized off-diagonals (dim, dim)
            diagonal_vars: Diagonal variance components (dim,)
            off_diag_correlations: Off-diagonal correlation values, flattened (n_off_diag,)
        """
        # Get full covariance matrix
        Sigma = self.get_covariance_matrix(theta)
        
        # Extract diagonal variances
        diagonal_vars = jnp.diag(Sigma)
        
        # Compute correlation matrix: R[i,j] = Σ[i,j] / sqrt(Σ[i,i] * Σ[j,j])
        sqrt_diag = jnp.sqrt(diagonal_vars)
        correlation_matrix = Sigma / jnp.outer(sqrt_diag, sqrt_diag)
        
        # Apply tanh to off-diagonal elements to ensure [-1, +1] bounds
        mask = jnp.eye(self.dim)
        off_diag_mask = 1.0 - mask
        
        normalized_correlations = (
            mask * jnp.ones_like(correlation_matrix) +  # Diagonal = 1.0
            off_diag_mask * jnp.tanh(correlation_matrix)  # Off-diagonal = tanh(correlation)
        )
        
        # Extract off-diagonal correlation values (lower triangular, excluding diagonal)
        tril_indices = jnp.tril_indices(self.dim, k=-1)
        off_diag_correlations = normalized_correlations[tril_indices]
        
        return normalized_correlations, diagonal_vars, off_diag_correlations



class GlobalCorrectionModel(eqx.Module):
    """
    Global correction model: r_ψ(x̂|x) = N(x̂; x + μ_global, L_global L_global^T)
    
    Simple correction model with global (theta-independent) mean shift and covariance.
    - Mean: μ = x + μ_global where μ_global is a learnable global bias
    - Covariance: Σ = L_global L_global^T where L_global is a learnable Cholesky factor
    
    Key features:
    - No theta dependence - parameters are constant across all parameter values
    - Global mean shift handles constant systematic bias
    - Global covariance structure models consistent noise patterns
    - More flexible than SimpleCorrectionModel but simpler than theta-dependent models
    """
    # Global mean shift parameter
    mu_global: Array                 # (dim,) - learnable global bias correction
    
    # Global Cholesky factor (lower triangular)
    L_global_raw: Array              # (dim, dim) - raw parameters for global Cholesky factor
    
    dim: int
    
    def __init__(self, key: KeyArray, output_dim: int):
        """
        Args:
            key: Random key for parameter initialization
            output_dim: Dimension of output data x
        """
        key1, key2 = jax.random.split(key, 2)
        
        # Initialize global mean shift parameter (small random values)
        self.mu_global = jax.random.normal(key1, (output_dim,)) * 0.01
        
        # Initialize global Cholesky factor
        self.L_global_raw = jax.random.normal(key2, (output_dim, output_dim)) * 0.05
        self.L_global_raw = self.L_global_raw.at[jnp.diag_indices(output_dim)].set(-1.0)  # Start with small diagonal
        
        self.dim = output_dim
    
    def _get_global_cholesky_factor(self) -> Array:
        """Get global Cholesky factor from raw parameters."""
        L_global = jnp.tril(self.L_global_raw)  # Ensure lower triangular
        # Make diagonal entries positive via softplus
        diag_mask = jnp.eye(self.dim)
        off_diag_mask = 1.0 - diag_mask
        diag_part = diag_mask * jax.nn.softplus(jnp.diag(self.L_global_raw)) + 1e-6
        off_diag_part = off_diag_mask * L_global
        return jnp.diag(diag_part) + off_diag_part
    
    def get_covariance_matrix(self) -> Array:
        """Compute global covariance matrix Σ = L_global L_global^T."""
        L_global = self._get_global_cholesky_factor()
        Sigma = L_global @ L_global.T
        return Sigma
    
    def __call__(self, x: Array, theta: Array = None) -> tuple[Array, Array]:
        """
        Args:
            x: Input data (dim,) or (batch_size, dim)
            theta: Parameter theta (ignored - for interface compatibility)
            
        Returns:
            mu: Mean parameters x + μ_global (with global shift)
            Sigma: Global covariance matrix (dim, dim) or broadcasted for batch
        """
        # Handle both single samples and batches
        if x.ndim == 1:
            # Single sample case: (dim,)
            mu = x + self.mu_global
            Sigma = self.get_covariance_matrix()
            return mu, Sigma
        else:
            # Batch case: (batch_size, dim)
            batch_size = x.shape[0]
            mu = x + self.mu_global  # Broadcast global mean
            
            # Broadcast covariance matrix for batch
            Sigma = self.get_covariance_matrix()
            Sigma_batch = jnp.broadcast_to(Sigma[None, :, :], (batch_size, self.dim, self.dim))
            return mu, Sigma_batch
    
    def log_prob(self, x_hat: Array, x: Array, theta: Array = None) -> Array:
        """
        Compute log probability log r_ψ(x̂|x) with global mean shift and covariance.
        
        Args:
            x_hat: Corrected observations (batch_size, dim) or (dim,)
            x: Simulator outputs (batch_size, dim) or (dim,)
            theta: Parameters (ignored - for interface compatibility)
            
        Returns:
            log_prob: Log probabilities (batch_size,) or scalar
        """
        mu, Sigma = self(x, theta)
        
        # Handle both single samples and batches
        squeeze_output = False
        if x_hat.ndim == 1:
            x_hat = x_hat[None, :]  # Add batch dimension
            mu = mu[None, :]
            if Sigma.ndim == 2:
                Sigma = Sigma[None, :, :]
            squeeze_output = True
        
        batch_size = x_hat.shape[0]
        
        if batch_size == 1:
            # Single sample case - direct computation
            L = jnp.linalg.cholesky(Sigma[0] + 1e-6 * jnp.eye(self.dim))
            diff_vec = (x_hat[0] - mu[0]).reshape(-1)
            y_single = jax.scipy.linalg.solve_triangular(L, diff_vec, lower=True)
            
            log_prob = -0.5 * jnp.sum(y_single**2)
            log_prob -= jnp.sum(jnp.log(jnp.diag(L)))
            log_prob -= 0.5 * self.dim * jnp.log(2 * jnp.pi)
            log_prob = jnp.array([log_prob])
        else:
            # Batch case - vectorized computation
            L_batch = jax.vmap(lambda S: jnp.linalg.cholesky(S + 1e-6 * jnp.eye(self.dim)))(Sigma)
            diff = x_hat - mu  # (batch_size, dim)
            
            def solve_single(L_i, diff_i):
                return jax.scipy.linalg.solve_triangular(L_i, diff_i, lower=True)
            
            y_batch = jax.vmap(solve_single)(L_batch, diff)
            
            log_prob = -0.5 * jnp.sum(y_batch**2, axis=-1)
            log_prob -= jnp.sum(jnp.log(jnp.diagonal(L_batch, axis1=-2, axis2=-1)), axis=-1)
            log_prob -= 0.5 * self.dim * jnp.log(2 * jnp.pi)
        
        if squeeze_output:
            log_prob = log_prob[0]
            
        return log_prob
