"""Test script to diagnose SimpleCorrectionModel parameter updates."""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from src.models.correction_model import SimpleCorrectionModel

# Initialize model
key = jax.random.PRNGKey(42)
dim = 4
initial_var = 1e-4  # 10^-4 as you want

# Create model with 10^-4 initial variance
initial_covariance = jnp.full(dim, initial_var)
model = SimpleCorrectionModel(key=key, dim=dim, initial_covariance=initial_covariance)

print("=" * 60)
print("SIMPLE CORRECTION MODEL DIAGNOSTIC TEST")
print("=" * 60)

# Check initial covariance
L_init = model.get_cholesky_factor()
Sigma_init = model.get_covariance_matrix()
diag_init = jnp.diag(Sigma_init)

print("\n📊 INITIAL STATE:")
print(f"Target variance: {initial_var:.6f}")
print(f"Actual diagonal variances: {diag_init}")
print(f"L_raw diagonal (raw params): {jnp.diag(model.L_raw)}")
print(f"L diagonal (after softplus): {jnp.diag(L_init)}")
print(f"\nFull covariance matrix:")
print(Sigma_init)

# Check if off-diagonal elements exist
off_diag_mask = 1.0 - jnp.eye(dim)
off_diag_elements = Sigma_init * off_diag_mask
max_off_diag = jnp.max(jnp.abs(off_diag_elements))
print(f"\nMax off-diagonal covariance: {max_off_diag:.8f}")

# Test gradient flow with a simple loss
def simple_loss(model, x_obs, x_sim):
    """Simple loss to test if gradients flow."""
    log_prob = model.log_prob(x_obs, x_sim, theta=None)
    return -log_prob.mean()  # Negative log likelihood

# Create synthetic data
key_data = jax.random.PRNGKey(123)
x_obs = jax.random.normal(key_data, (10, dim))  # 10 observations
x_sim = jax.random.normal(key_data, (10, dim))  # 10 simulated

# Compute initial loss
loss_init = simple_loss(model, x_obs, x_sim)
print(f"\n📉 Initial loss: {loss_init:.6f}")

# Compute gradients
grad_fn = eqx.filter_grad(simple_loss)
grads = grad_fn(model, x_obs, x_sim)

# Check if gradients are non-zero
L_raw_grad = grads.L_raw
diag_grad = jnp.diag(L_raw_grad)
off_diag_grad = L_raw_grad * jnp.tril(jnp.ones_like(L_raw_grad), k=-1)

print("\n🔄 GRADIENT CHECK:")
print(f"L_raw diagonal gradients: {diag_grad}")
print(f"L_raw diagonal grad magnitude: {jnp.linalg.norm(diag_grad):.8f}")
print(f"L_raw off-diagonal grad magnitude: {jnp.linalg.norm(off_diag_grad):.8f}")

if jnp.linalg.norm(diag_grad) < 1e-10:
    print("❌ WARNING: Diagonal gradients are near zero!")
else:
    print("✅ Diagonal gradients are flowing")

if jnp.linalg.norm(off_diag_grad) < 1e-10:
    print("❌ WARNING: Off-diagonal gradients are near zero!")
else:
    print("✅ Off-diagonal gradients are flowing")

# Test a few optimization steps
optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

print("\n🏃 RUNNING 100 OPTIMIZATION STEPS...")
for step in range(100):
    grads = grad_fn(model, x_obs, x_sim)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)

    if step % 20 == 0:
        loss = simple_loss(model, x_obs, x_sim)
        print(f"Step {step:3d}: loss = {loss:.6f}")

# Check final state
L_final = model.get_cholesky_factor()
Sigma_final = model.get_covariance_matrix()
diag_final = jnp.diag(Sigma_final)
off_diag_final = Sigma_final * off_diag_mask

print("\n📊 FINAL STATE (after 100 steps):")
print(f"Diagonal variances: {diag_final}")
print(f"Change from initial: {diag_final - diag_init}")
print(f"L_raw diagonal (raw params): {jnp.diag(model.L_raw)}")
print(f"\nFull covariance matrix:")
print(Sigma_final)
print(f"Max off-diagonal covariance: {jnp.max(jnp.abs(off_diag_final)):.8f}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY:")
print("=" * 60)
diag_change = jnp.linalg.norm(diag_final - diag_init)
cov_change = jnp.linalg.norm(Sigma_final - Sigma_init)

if diag_change < 1e-6:
    print("❌ Model parameters DID NOT update significantly")
    print(f"   Diagonal change: {diag_change:.10f}")
    print(f"   Covariance change: {cov_change:.10f}")
    print("\n🔍 Possible issues:")
    print("   1. Gradients blocked somewhere")
    print("   2. Initialization causing numerical issues")
    print("   3. Learning rate too small")
else:
    print("✅ Model parameters ARE updating!")
    print(f"   Diagonal change: {diag_change:.10f}")
    print(f"   Covariance change: {cov_change:.10f}")
