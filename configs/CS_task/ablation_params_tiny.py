from configs.CS_task.cs_task import get_config as get_base_config

"""
ABLATION STUDY: Parameter Size - Tiny
CS task, RVNP-NN, nobs=100, nn_block_dim=16
"""

def get_config():
    config = get_base_config()

    # Standard experiment settings (nobs=100, NN)
    config.data.num_tests = 100
    config.data.num_iid = 1
    config.data.inference_simulations = 100
    config.model.correction_type = 'NN'
    config.model.lambda_shrinkage = 0.0
    config.model.K_obs_samples = 30
    config.model.simulator_samples_per_theta = 32
    config.training.final_epochs = 300
    config.training.final_posterior_epochs = 100

    # ABLATION: Parameter size
    config.model.nn_block_dim = 16
    config.training.workspace = 'output/ablation_cs_params_tiny_nobs100'

    return config
