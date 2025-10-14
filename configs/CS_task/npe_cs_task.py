from ml_collections import ConfigDict

from configs.default_flow import get_config as base_get_config
import ml_collections
import jax.numpy as jnp

"""
NPE (Neural Posterior Estimation) configuration for CS task
Uses standard neural posterior estimation with maximum likelihood training - no corruption or complex staging
"""

def get_config():
    config = base_get_config()
    
    #################################
    # variable terms
    #################################    
    config.training.batch_size = 2**10
    config.data.num_tests=100
    config.data.num_iid=1
    config.data.inference_simulations = int(config.data.num_tests*config.data.num_iid)      # For custom datasets (if applicable).
    config.model.name = 'npe'  # Use standard NPE model
    config.training.use_initialization=False  # NPE uses simple training, no initialization needed
    config.training.init_epochs=0  # No initialization for NPE
    config.training.warmup_epochs=0  # No warmup for NPE
    config.training.final_epochs=0  # No staged training for NPE
    config.training.final_posterior_epochs=0  # No final tuning stage for NPE
    config.model.lambda_shrinkage=0.0  # No shrinkage needed for simple NPE

    #################################
    # constants
    #################################
    config.training.workspace='output/npe_cs_task_100000'
    config.data.num_simulations = int(100000/(1-config.training.validation_split))      # For custom datasets (if applicable).
    
    config.data.vector_dim_inference=4 #this is to hold the samples
    config.sampling.sampling_name='npe'
    config.sampling.inference_method='CS_task'
    ##### training/sampling parameters 
    config.training.train=True 
    config.sampling.inference=not config.training.train
    #################################
    # training parameters
    #################################
    config.training.n_iters = 300           # Total number of training iterations.
    config.optim.lr = 1e-3
    config.optim.optimizer = 'Adam'
    config.optim.beta1 = 0.9
    config.optim.eps = 1e-8
    config.optim.weight_decay = 1e-5
    config.optim.warmup = 1000
    config.optim.grad_clip = 10.0
    config.training.max_patience=100         # Patience for early stopping
    #################################
    # model parameters
    #################################
    config.model.nn_depth_bnaf = 5
    config.model.activation='tanh'
    config.model.flow_dimension=3  # CS task has 3 parameters
    config.model.nn_block_dim=52
    config.data.summary_dim=0
    config.model.cond_dim=4  # CS task condition dimension

    #################################
    # synthetic data parameters
    #################################
    config.data.dataset = 'CS'
    config.data.inference_dataset= 'CS_inference'
    config.training.validation_split=0.1
    config.data.data_path = './Data'            # Path to store/load custom dataset.
    config.data.vector_dim=config.model.cond_dim
    config.data.benchmark=False
    
    #################################
    # Training loop optimizations
    config.training.validation_frequency=3
    config.training.logging_frequency=5
    #################################

    # Simplified NPE configuration - no complex variational or correction terms
    # NPE uses standard maximum likelihood training without corruption or staging
    config.model.use_posterior_term=True  # Standard posterior term for NPE
    config.model.amortized_training=True
    
    # Disable all complex training stages (NPE doesn't need them)
    config.model.train_simulator=False       # NPE doesn't train simulator
    config.model.train_embeddings=True       # NPE can use embeddings if available
    config.model.train_posterior_init=False  # No initialization stage
    config.model.train_correction_coarse=False    # No correction model
    config.model.train_posterior_widen=False      # No widening stage
    config.model.train_joint_refinement=False     # No joint refinement
    config.model.train_joint_alternating=False   # No alternating training
    config.model.train_final_posterior=False      # No final posterior tuning
    config.model.load_saved_models_for_stage6=False  # No stage 6
    config.training.train_embedding_first=False  # Simple training, no separate embedding stage
    
    # Disable variational and correction terms (not needed for standard NPE)
    config.model.lambda_variational=0.0  # No variational loss
    config.model.K_obs_samples=0  # No corruption sampling
    config.model.lambda_entropy=0.0  # No entropy regularization
    config.model.lambda_correlation=0.0  # No correlation penalty
    config.model.use_kl_term=False  # No KL divergence
    config.model.use_prior_term=False  # No prior term
    config.model.lambda_prior=0.0  # No prior weight
    config.model.use_variational=False  # Disable variational terms
    config.model.variational_temperature=1.0
    config.model.use_posterior_theta_sampling=False  # Use training theta, not posterior samples
    config.model.use_iw_elbo=False  # No importance weighting
    config.model.use_iw_forward_kl=False  # No forward KL
    config.model.lambda_kl=0.0  # No KL weight
    
    # Disable MMD (not used in standard NPE)
    config.model.use_mmd_divergence=False
    config.model.lambda_mmd=0.0
    config.model.mmd_n_samples=0
    config.model.mmd_n_corrected_samples=0
    config.model.use_median_heuristic=False
    config.model.mmd_sigma=1.0
    
    # Disable theta clipping (not needed for clean NPE training)
    config.model.clip_theta_to_bounds=False
    
    # Disable simulator sampling and correction parameters
    config.model.use_simulator_sampling=False
    config.model.simulator_samples_per_theta=0
    config.model.initial_correction_variance=0.0
    config.model.lambda_consistency=0.0
    config.model.lambda_consistency_final=0.0
    
    # Disable attention mechanisms (not used in standard NPE)
    config.model.use_uniform_attention=False
    config.model.use_reparam_contrastive=False

    return config