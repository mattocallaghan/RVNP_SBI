from ml_collections import ConfigDict

from configs.default_flow import get_config as base_get_config
import ml_collections
import jax.numpy as jnp

"""
RANPT configuration for Pendulum task
Uses RANPT model with correction model corruption
"""

def get_config():
    # Start with the base configuration.
    config = base_get_config()
    
    config.training.workspace='output/ranpt_pendulum_task_1000_mu_hybrid'
    config.data.num_simulations = int(100000/(1-config.training.validation_split))      # For custom datasets (if applicable).
    
    config.data.num_iid=1
    config.data.num_tests=1000
    config.data.inference_simulations = int(config.data.num_tests*config.data.num_iid)      # For custom datasets (if applicable).
    config.model.correction_lambda_shrinkage=0.0
    config.data.vector_dim_inference=200 # Pendulum time series data
    config.sampling.sampling_name='ranpt'
    config.sampling.inference_method='pendulum_task'

    ##### training/sampling parameters 
    config.training.train=True 
    config.sampling.inference=not config.training.train

    #################################
    config.training.batch_size = 2**10
    config.training.n_iters = 300           # Total number of training iterations.
    config.optim.lr = 1e-3
    config.optim.optimizer = 'Adam'
    config.optim.beta1 = 0.9
    config.optim.eps = 1e-8
    config.optim.weight_decay = 1e-5
    config.optim.warmup = 1000
    config.optim.grad_clip = 10.0
    config.training.max_patience=100         # Patience for early stopping
    config.training.stop_on_model_mismatch=True  # Stop if saved model has shape mismatch
    config.model.correction_model_name="hybrid"
    config.augmentation_factor=100
    config.model.name = 'ranpt'  # Use RANPT model
    config.model.nn_depth_bnaf = 3
    config.model.activation='tanh'
    config.model.flow_dimension=2  # Pendulum parameters
    config.model.nn_block_dim=32
    config.data.summary_dim=0
    config.model.cond_dim=5  # Pendulum uses embedding_dim=5
    config.model.embedding='IM' # or vae
    config.model.hidden_scale=1
    config.model.embedding_dim=5
    
    #synthetic data parameters
    config.data.dataset = 'pendulum'
    config.data.inference_dataset= 'pendulum_inference'
    config.training.validation_split=0.1
    config.data.data_path = './Data'            # Path to store/load custom dataset.
    config.data.vector_dim=2
    config.data.benchmark=False

    #### Inference
    config.sampling.method = 'nuts'              # Options: 'ode' or 'pc' (predictor-corrector).
    config.sampling.step_size_initial=1
    config.sampling.warmup_steps=300
    config.sampling.samples=1000
    config.sampling.adapt_step_size=True
    config.sampling.adapt_mass_matrix=True
    config.sampling.num_chains=5
    config.sampling.max_tree_depth=10
    config.sampling.acceptance_prob=0.9
    config.sampling.num_steps_train_loss_model=30
    config.sampling.lr_train_loss_model=1e-3
    
    # MAIN VARIATIONAL POSTERIOR CONFIGURATION 
    config.model.correction_type='mu_hybrid'  # 'hybrid', 'diagonal_neural', or 'neural'
    config.model.lambda_variational=1.0  # variational loss weight
    config.model.K_obs_samples=30  # Increased for better sampling diversity
    config.model.lambda_entropy=0.0
    config.model.lambda_shrinkage=0.0
    config.model.lambda_correlation=0.0  # L2 penalty on correlation logits to encourage diagonal covariance
    config.model.use_kl_term=True  # Turn on KL divergence term
    config.model.use_prior_term=False  # Keep prior term off for now
    config.model.lambda_prior=0.0  # Weight for prior term
    config.model.use_variational=True  # Enabled for better data matching
    config.model.variational_temperature=1.0  # temperature for weighting the log p samples
    config.model.use_posterior_theta_sampling=True  # Use posterior samples for theta during training
    config.model.use_iw_elbo=True  # Use importance-weighted ELBO computation (set to True to enable)
    config.model.use_iw_forward_kl=False  # Use importance-weighted Forward KL + Reverse KL (set to True to enable)
    config.model.lambda_kl=1.0  # Weight for KL term
    
    # MMD Configuration
    config.model.use_mmd_divergence=False  # Enable MMD instead of KL divergence
    config.model.lambda_mmd=1.0  # Weight for MMD loss term
    config.model.mmd_n_samples=100  # Number of simulator samples per theta for MMD
    config.model.mmd_n_corrected_samples=50  # Number of corrected samples per x_sim
    config.model.use_median_heuristic=True  # Use median heuristic for RBF bandwidth
    config.model.mmd_sigma=1.0  # Fixed RBF bandwidth (used if use_median_heuristic=False)
    
    # Theta Clipping Configuration
    config.model.clip_theta_to_bounds=True  # Clip sampled theta to training data bounds for stability
    
    # Training loop optimizations
    config.training.validation_frequency=3
    config.training.logging_frequency=5
    
    return config