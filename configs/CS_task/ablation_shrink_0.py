
from ml_collections import ConfigDict

from configs.default_flow import get_config as base_get_config
import ml_collections
import jax.numpy as jnp

"""
ABLATION STUDY: Shrinkage Prior - None (Baseline)
CS task with mu_hybrid correction, lambda_shrinkage=0.0
Nobs=100, nn_block_dim=52
"""

def get_config():
    # Start with the base configuration.
    config = base_get_config()

    config.training.workspace='output/ablation_cs_shrink_0_nobs100'
    config.data.num_simulations = int(100000/(1-config.training.validation_split))

    config.data.num_iid=1
    config.data.num_tests=100
    config.data.inference_simulations = int(config.data.num_tests*config.data.num_iid)
    config.model.correction_lambda_shrinkage=0.0
    config.data.vector_dim_inference=4
    config.sampling.sampling_name='ranpt'
    config.sampling.inference_method='CS_task'

    ##### training/sampling parameters
    config.training.train=True
    config.sampling.inference=not config.training.train


    #################################
    config.training.batch_size = 2**10
    config.training.n_iters = 300
    config.optim.lr = 1e-3
    config.optim.optimizer = 'Adam'
    config.optim.beta1 = 0.9
    config.optim.eps = 1e-8
    config.optim.weight_decay = 1e-5
    config.optim.warmup = 1000
    config.optim.grad_clip = 10.0
    config.training.max_patience=100
    config.training.stop_on_model_mismatch=True
    config.model.correction_model_name="mu_hybrid"
    config.augmentation_factor=100
    config.model.name = 'ranpt'
    config.model.nn_depth_bnaf = 5
    config.model.activation='tanh'
    config.model.flow_dimension=3
    config.model.nn_block_dim=52  # ABLATION: Fixed parameter size
    config.data.summary_dim=0
    config.model.cond_dim=4


    #synthetic data parameters
    config.data.dataset = 'CS'
    config.data.inference_dataset= 'CS_inference'
    config.training.validation_split=0.1
    config.data.data_path = './Data'
    config.data.vector_dim=config.model.cond_dim
    config.data.benchmark=False

    #### Inference
    config.sampling.method = 'nuts'
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
    config.model.correction_type='mu_hybrid'
    config.model.lambda_variational=1.0
    config.model.K_obs_samples=30
    config.model.lambda_entropy=0.0
    config.model.lambda_shrinkage=0.0  # ABLATION: No shrinkage (baseline)
    config.model.lambda_correlation=0.0
    config.model.use_kl_term=True
    config.model.use_prior_term=False
    config.model.lambda_prior=0.0
    config.model.use_variational=True
    config.model.variational_temperature=1.0
    config.model.use_posterior_theta_sampling=True
    config.model.use_iw_elbo=True
    config.model.use_iw_forward_kl=False
    config.model.lambda_kl=1.0

    # MMD Configuration
    config.model.use_mmd_divergence=False
    config.model.lambda_mmd=1.0
    config.model.mmd_n_samples=100
    config.model.mmd_n_corrected_samples=50
    config.model.use_median_heuristic=True
    config.model.mmd_sigma=1.0

    # Theta Clipping Configuration
    config.model.clip_theta_to_bounds=True

    # Staged training parameters
    config.model.n_alternates = 2

    # === 5-STAGE TRAINING CONFIGURATION ===
    config.model.train_simulator=False
    config.model.train_posterior_init=True
    config.model.train_correction_coarse=True
    config.model.train_posterior_widen=True
    config.model.train_joint_refinement=True

    # === FINAL STAGE OPTIONS ===
    config.model.train_joint_alternating=False
    config.model.train_final_posterior=True
    config.model.load_saved_models_for_stage6=False

    # === ATTENTION MECHANISM OPTIONS ===
    config.model.use_uniform_attention=False
    config.model.use_reparam_contrastive=True

    # === SIMULATOR SAMPLING FOR CORRECTION STAGE ===
    config.model.use_simulator_sampling=True
    config.model.simulator_samples_per_theta=32
    config.model.initial_correction_variance=-0.1

    # Extra posterior loss parameters
    config.model.lambda_consistency=0.0
    config.model.lambda_consistency_final=0.0
    config.model.use_posterior_term=False

    # Training parameters
    config.model.amortized_training=True

    # Multi-stage training parameters
    config.training.train_embedding_first=False
    config.training.use_initialization=False
    config.training.init_epochs=15
    config.training.warmup_epochs=5
    config.training.final_epochs=300
    config.training.final_posterior_epochs=100

    # Training loop optimizations
    config.training.validation_frequency=3
    config.training.logging_frequency=5

    return config
