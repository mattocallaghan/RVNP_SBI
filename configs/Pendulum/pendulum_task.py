

from ml_collections import ConfigDict

from configs.default_flow import get_config as base_get_config
import ml_collections
import jax.numpy as jnp

"""
Pendulum Task Base Configuration
Serves as base config for all Pendulum experiments (RVNP-NN, RVNP-simple, wellspec)
Inherits from default_flow.py and defines Pendulum-specific parameters
"""

def get_config():
    config = base_get_config()

    #################################
    # TASK CONSTANTS (never override)
    #################################
    config.data.dataset = 'pendulum'
    config.data.vector_dim = 2  # Dimension after embedding
    config.data.vector_dim_inference = 200  # Raw time series dimension
    config.model.flow_dimension = 2
    config.model.cond_dim = 2  # Dimension of embedding (what we condition on)
    config.data.summary_dim = 0
    config.sampling.inference_method = 'pendulum_task'
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
    config.training.batch_size = 2**10  # 1024
    config.training.n_iters = 300
    config.model.nn_depth_bnaf = 3  # Shallower network for Pendulum
    config.model.nn_block_dim = 32  # Smaller network for Pendulum
    config.model.activation = 'tanh'
    config.training.max_patience = 40  # Lower patience for Pendulum
    config.data.num_simulations = int(100000/(1-config.training.validation_split))
    config.training.train_embedding_first = False
    config.training.embedding_epochs = 200  # Dedicated embedding training epochs

    #################################
    # OPTIMIZER DEFAULTS
    #################################
    config.optim.lr = 1e-3
    config.optim.optimizer = 'Adam'
    config.optim.beta1 = 0.9
    config.optim.eps = 1e-8
    config.optim.weight_decay = 1e-5
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
    config.data.inference_dataset = 'pendulum_inference'
    config.model.lambda_shrinkage = 0.0
    config.training.workspace = 'output/pendulum_task_tests1_simple'

    #################################
    # TRAINING FLAGS
    #################################
    config.training.train = True
    config.sampling.inference = not config.training.train
    config.training.validation_split = 0.1
    config.training.validation_frequency = 3
    config.training.logging_frequency = 5
    config.training.use_initialization = False
    config.training.init_epochs = 5
    config.training.warmup_epochs = 5
    config.training.final_epochs = 0
    config.training.final_posterior_epochs = 0
    config.model.train_embedding = False
    config.model.train_embeddings = False

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
    config.model.mmd_sigma = 1.0

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
    config.model.initial_correction_variance = -3.0

    # Extra posterior parameters
    config.model.lambda_consistency = 0.0
    config.model.lambda_consistency_final = 0.0
    config.model.use_posterior_term = False
    config.model.amortized_training = True

    return config
