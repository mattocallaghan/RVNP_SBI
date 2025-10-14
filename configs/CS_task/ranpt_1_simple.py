
from ml_collections import ConfigDict

from configs.default_flow import get_config as base_get_config
import ml_collections
import jax.numpy as jnp

"""
NNPE (Noisy Neural Posterior Estimation) configuration for CS task
Uses noisy_neural_posterior model with spike-and-slab corruption for robustness testing
"""

def get_config():
    # Start with the base configuration.
    config = base_get_config()
    
    config.training.workspace='output/nnpe_cs_task_1'
    config.data.num_simulations = int(100000/(1-config.training.validation_split))      # For custom datasets (if applicable).
    
    config.data.num_iid=1
    config.data.num_tests=1
    config.data.inference_simulations = int(config.data.num_tests*config.data.num_iid)      # For custom datasets (if applicable).
    config.model.correction_lambda_shrinkage=0.0
    config.data.vector_dim_inference=4 #this is to hold the samples
    config.sampling.sampling_name='ranpt'
    config.sampling.inference_method='CS_task'

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
    config.model.correction_model_name="simple"
    config.augmentation_factor=100
    config.model.name = 'ranpt'  # Use NNPE model with spike-and-slab corruption
    config.model.nn_depth_bnaf = 5
    config.model.activation='tanh'
    config.model.flow_dimension=3#T/F
    config.model.nn_block_dim=52
    config.data.summary_dim=0
    config.model.cond_dim=4
    


    
    #synthetic data parameters
    config.data.dataset = 'CS'
    config.data.inference_dataset= 'CS_inference'
    config.training.validation_split=0.1
    config.data.data_path = './Data'            # Path to store/load custom dataset.
    config.data.vector_dim=config.model.cond_dim
    config.data.benchmark=False

    #### Inference
    ## tuning paramaerers
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
    config.model.correction_type='simple'  # 'simple', 'diagonal_neural', or 'neural'
    config.model.lambda_variational=1.0  # variational loss weight
    config.model.K_obs_samples=30  # Increased for better sampling diversity
    config.model.lambda_entropy=0.0
    config.model.lambda_shrinkage=0.0
     # Stronger shrinkage prior toward delta function (no misspecification)
    config.model.lambda_correlation=0.0  # L2 penalty on correlation logits to encourage diagonal covariance
    config.model.use_kl_term=True  # Turn on KL divergence term
    config.model.use_prior_term=False  # Keep prior term off for now
    config.model.lambda_prior=0.0  # Weight for prior term
    # use the variational terms
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
    
    # Staged training parameters - 5-stage approach with alternating execution
    # Each stage runs for n_alternates iterations before moving to next stage
    config.model.n_alternates = 2  # Number of iterations per stage in alternating mode (reduced for testing)
    
    # === 5-STAGE TRAINING CONFIGURATION ===
    config.model.train_simulator=False       # Stage 2: Train simulator flow p(x_sim|θ)
    config.model.train_posterior_init=True       # Stage 3: Initialize posterior p_φ(θ|x_obs)
    config.model.train_correction_coarse=True    # Stage 4: Coarse correction training
    config.model.train_posterior_widen=True      # Stage 4.5: Widen posterior using corrected samples
    config.model.train_joint_refinement=True     # Stage 5: Joint posterior + correction refinement
    
    # === FINAL STAGE OPTIONS ===
    # Set to True for alternating correction/posterior updates, False for joint training
    config.model.train_joint_alternating=False   # Stage 5: alternating vs joint mode
    config.model.train_final_posterior=True      # Stage 6: Final posterior tuning with fixed correction model
    config.model.load_saved_models_for_stage6=False  # Load saved models and run only Stage 6
    
    # === ATTENTION MECHANISM OPTIONS ===
    config.model.use_uniform_attention=False      # True=uniform weights, False=simulator-based weights (default)
    config.model.use_reparam_contrastive=True   # True=reparameterization sampling, False=attention-based contrastive
    
    # === SIMULATOR SAMPLING FOR CORRECTION STAGE ===
    # During correction training, sample fresh data from the trained simulator instead of using fixed training data
    # For each theta parameter, generate 25 fresh samples from simulator_flow.sample(theta) (reduced for speed)
    # Use distance-dependent soft attention: w_ij = softmax(-||x_obs_j - x_sim_i||_2 / τ)
    # where τ = contrastive_temperature = 0.01 (sharp attention focusing on nearest samples)
    config.model.use_simulator_sampling=True        # Enable fresh simulator sampling during correction stages
    config.model.simulator_samples_per_theta=32     # Sample 25 times per theta (reduced for speed while maintaining diversity)
    config.model.initial_correction_variance=-0.1
    ###########
    #extra posteror loss parameters
    # Consistency regularization - 
    config.model.lambda_consistency=0.0  # Updated posterior regularization
    config.model.lambda_consistency_final=0.0 # Increased consistency weight for final stage 6
    # the amortiezed term on all the samples
    config.model.use_posterior_term=False  # Turn off first posterior term
    
    # Training parameters -
    config.model.amortized_training=True
    
    # Multi-stage training parameters -  
    config.training.train_embedding_first=False  # CS task doesn't use embeddings
    config.training.use_initialization=False  # Enable initialization for better starting point
    config.training.init_epochs=15  # Short initialization
    config.training.warmup_epochs=5  # Epochs for stages 1-4.5 (warmup)
    config.training.final_epochs=300  # Epochs for stage 5 (main training)
    config.training.final_posterior_epochs=100  # Epochs for stage 6 (final posterior tuning)
    
    # Training loop optimizations
    config.training.validation_frequency=3
    config.training.logging_frequency=5
    
    return config