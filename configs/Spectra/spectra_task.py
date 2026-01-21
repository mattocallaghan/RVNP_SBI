

from ml_collections import ConfigDict

from configs.default_flow import get_config as base_get_config
import ml_collections
import jax.numpy as jnp

"""
Spectra Task Base Configuration
Serves as base config for all Spectra experiments (RVNP-NN, RVNP-simple, wellspec)
Inherits from default_flow.py and defines Spectra-specific parameters
"""

def get_config():
    config = base_get_config()

    #################################
    # TASK CONSTANTS (never override)
    #################################
    config.data.dataset = 'spectra'
    config.data.vector_dim = 300  # Raw spectra dimension
    config.data.vector_dim_inference = 300
    config.model.flow_dimension = 3
    config.model.cond_dim = 3  # Dimension of embedding (what we condition on)
    config.data.summary_dim = 0
    config.sampling.inference_method = 'spectra_task'
    config.sampling.sampling_name = 'nlpe_embedding'
    config.data.data_path = './Data'
    config.data.benchmark = False

    # Embedding configuration
    config.model.embedding = 'IM'  # Information Maximization
    config.model.hidden_scale = 1
    config.model.embedding_dim = 5

    #################################
    # TASK-SPECIFIC DEFAULTS (rarely override)
    #################################
    config.training.batch_size = 4096  # Larger batch size for Spectra
    config.training.n_iters = 100  # Shorter training for Spectra
    config.model.nn_depth_bnaf = 5
    config.model.nn_block_dim = 52
    config.model.activation = 'tanh'
    config.training.max_patience = 100
    config.data.num_simulations = 125711  # Unique value for Spectra
    config.training.train_embedding_first = True  # Train embeddings first for Spectra
    config.training.embedding_epochs = 200  # Dedicated embedding training epochs

    #################################
    # OPTIMIZER DEFAULTS
    #################################
    config.optim.lr = 1e-3
    config.optim.optimizer = 'Adam'
    config.optim.beta1 = 0.9
    config.optim.eps = 1e-8
    config.optim.weight_decay = 1e-4  # Different from other tasks
    config.optim.warmup = 1000
    config.optim.grad_clip = 10.0

    #################################
    # EXPERIMENT DEFAULTS (commonly overridden)
    #################################
    config.model.name = 'nlpe_rqs_posterior'
    config.model.correction_type = 'simple'
    config.data.num_tests = 1
    config.data.num_iid = 1
    config.data.inference_simulations = 1
    config.data.inference_dataset = 'spectra_inference'
    config.model.lambda_shrinkage = 0.1  # Shrinkage prior on mean + covariance for NN models
    config.training.workspace = 'output/spectra_tests1_simple'

    #################################
    # TRAINING FLAGS
    #################################
    config.training.train = True
    config.sampling.inference = not config.training.train
    config.training.validation_split = 0.1
    config.training.validation_frequency = 3
    config.training.logging_frequency = 5
    config.training.use_initialization = False
    config.training.init_epochs = 0
    config.training.warmup_epochs = 5
    config.training.final_epochs = 0
    config.training.final_posterior_epochs = 100
    config.model.train_embeddings = True  # Enable embedding training

    #################################
    # RVNP-SPECIFIC PARAMETERS
    #################################
    # Variational posterior configuration
    config.model.lambda_variational = 1.0
    config.model.K_obs_samples = 10  # Default for small datasets
    config.model.lambda_entropy = 0.0
    config.model.lambda_correlation = 0.0
    config.model.use_kl_term = True
    config.model.use_prior_term = False
    config.model.lambda_prior = 0.0
    config.model.use_variational = True
    config.model.variational_temperature = 1.0
    config.model.use_posterior_theta_sampling = True
    config.model.use_iw_elbo = True
    config.model.use_iw_forward_kl = False
    config.model.lambda_kl = 1.0

    # MMD configuration (not used)
    config.model.use_mmd_divergence = False
    config.model.lambda_mmd = 1.0
    config.model.mmd_n_samples = 100
    config.model.mmd_n_corrected_samples = 50
    config.model.use_median_heuristic = True

    # Theta clipping
    config.model.clip_theta_to_bounds = True

    # Multi-stage training
    config.model.n_alternates = 2
    config.model.train_simulator = False
    config.model.train_correction_coarse = True
    config.model.train_posterior_widen = True
    config.model.train_joint_refinement = True
    config.model.train_joint_alternating = False
    config.model.train_final_posterior = True
    config.model.load_saved_models_for_stage6 = False

    # Attention mechanism
    config.model.use_uniform_attention = False
    config.model.use_reparam_contrastive = True

    # Simulator sampling
    config.model.use_simulator_sampling = True
    config.model.simulator_samples_per_theta = 10  # Default for small datasets
    config.model.initial_correction_variance = 0.1  # Positive value for Spectra

    # Extra posterior parameters
    config.model.lambda_consistency = 0.0
    config.model.lambda_consistency_final = 0.0
    config.model.use_posterior_term = False
    config.model.amortized_training = True

    return config
