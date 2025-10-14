

from ml_collections import ConfigDict

from configs.default_flow import get_config as base_get_config
import ml_collections
import jax.numpy as jnp

"""
The main condition is for the scaling by sigma, so we dont condition but we do scale by sigma
"""

def get_config():
    config = base_get_config()
    #################################
    # variable terms
    #################################    
    config.training.batch_size =2**10*2*2
    config.data.num_tests=100
    config.model.name = 'nlpe_rqs_posterior'
    config.training.use_initialization=False  # Enable initialization for better starting point
    config.training.init_epochs=0  # Short initialization
    config.training.warmup_epochs=5  # Epochs for stages 1-4.5 (warmup)
    config.training.final_epochs=0  # Epochs for stage 5 (main training)
    config.training.final_posterior_epochs=100  # Epochs for stage 6 (final posterior tuning)
    config.model.correction_type='simple'  # 'simple', 'diagonal_neural', or 'hybrid'
    config.model.lambda_shrinkage=0.0
    config.model.train_simulator=True       # Stage 2: Train simulator flow p(x_sim|θ)
    config.model.train_embeddings=True       # Enable embedding training
    config.model.lambda_entropy=0.0
    #################################
    # constants
    #################################
    config.training.workspace='output/Spectra_100000'
    config.data.num_simulations = int(125711)      # For custom datasets (if applicable).
    
    config.data.num_iid=1
    config.data.inference_simulations = int(config.data.num_tests*config.data.num_iid)      # For custom datasets (if applicable).
    
    config.data.vector_dim_inference=300 #this is to hold the samples
    config.sampling.sampling_name='nlpe_embedding'
    config.sampling.inference_method='spectra_task'

    ##### training/sampling parameters 
    config.training.train=True 
    config.sampling.inference=not config.training.train

    #################################
    # training parameters
    #################################
    config.training.n_iters = 100          # Total number of training iterations.
    config.optim.lr = 1e-3
    config.optim.optimizer = 'Adam'
    config.optim.beta1 = 0.9
    config.optim.eps = 1e-8
    config.optim.weight_decay = 1e-4
    config.optim.warmup = 1000
    config.optim.grad_clip = 10.0
    config.training.max_patience=100

    #################################
    # model parameters
    #################################
    config.model.nn_depth_bnaf = 5
    config.model.activation='tanh'
    config.model.nn_block_dim=52
    config.data.summary_dim=0
    config.model.cond_dim=3  # Dimension of embedding (what we condition on)
    config.model.embedding='IM' #or vae
    config.model.hidden_scale=1
    config.model.embedding_dim=5
    
    config.model.K_obs_samples=30  # Number of observation samples for correction model

    
    # MAIN VARIATIONAL POSTERIOR CONFIGURATION 
    config.model.lambda_variational=1.0  # variational loss weight
    config.model.use_variational=True  # Enabled for better data matching
    config.model.variational_temperature=1.0  # temperature for weighting the log p samples
    config.model.use_posterior_theta_sampling=True  # Use posterior samples for theta during training
    config.model.use_iw_elbo=True  # Use importance-weighted ELBO computation
    config.model.use_iw_forward_kl=False  # Use importance-weighted Forward KL + Reverse KL
    config.model.lambda_kl=1.0  # Weight for KL term
    config.model.use_kl_term=True  # Turn on KL divergence term
    config.model.use_prior_term=False  # Keep prior term off for now
    config.model.lambda_prior=0.0  # Weight for prior term
    
    # Multi-stage training parameters for embedding tasks
    config.model.use_uniform_attention=False  # Use simulator-based attention weights
    config.model.use_reparam_contrastive=True  # Use reparameterization sampling
    config.model.use_simulator_sampling=True  # Sample from simulator during training
    config.model.simulator_samples_per_theta=32  # Number of samples per theta
    config.model.lambda_consistency=0.0  # Posterior regularization weight
    config.model.lambda_consistency_final=0.0  # Consistency weight for final stage
    
    # Theta Clipping Configuration
    config.model.clip_theta_to_bounds=True  # Clip sampled theta to training data bounds for stability
    
    # MMD Configuration
    config.model.use_mmd_divergence=False  # Enable MMD instead of KL divergence
    config.model.lambda_mmd=1.0  # Weight for MMD loss term
    config.model.mmd_n_samples=100  # Number of simulator samples per theta for MMD
    config.model.mmd_n_corrected_samples=50  # Number of corrected samples per x_sim
    config.model.use_median_heuristic=True  # Use median heuristic for RBF bandwidth
    
    # Additional posterior term parameters
    config.model.use_posterior_term=False  # Turn off first posterior term
    
    
    # Multi-stage training parameters
    config.training.train_embedding_first=True  # Train embeddings before posterior
    config.training.embedding_epochs=200  # Dedicated embedding training epochs
    
    config.model.amortized_training=True  # Enable amortized training across multiple observations
    
    #################################
    # synthetic data parameters
    #################################
    config.data.dataset = 'spectra'
    config.data.inference_dataset= 'spectra_inference'
    config.training.validation_split=0.1
    config.data.data_path = './Data'            # Path to store/load custom dataset.
    config.data.vector_dim=300
    config.data.benchmark=False


    config.model.flow_dimension=3  # Dimension of θ (parameters) - what we're learning posterior for
    
    # === STAGED TRAINING CONFIGURATION ===
    config.model.train_posterior_init=True       # Stage 3: Initialize posterior p_φ(θ|x_obs)
    config.model.train_correction_coarse=True    # Stage 4: Coarse correction training
    config.model.train_posterior_widen=True      # Stage 4.5: Widen posterior using corrected samples
    config.model.train_joint_refinement=True     # Stage 5: Joint posterior + correction refinement
    
    # === FINAL STAGE OPTIONS ===
    config.model.train_joint_alternating=False   # Stage 5: alternating vs joint mode
    config.model.train_final_posterior=True      # Stage 6: Final posterior tuning with fixed correction model
    config.model.load_saved_models_for_stage6=False  # Load saved models and run only Stage 6
    
    # === INITIAL CORRECTION VARIANCE ===
    config.model.initial_correction_variance=0.1  # Positive value to avoid NaN in covariance matrix
    
    # Staged training parameters
    config.model.n_alternates = 2  # Number of iterations per stage in alternating mode
    
    return config