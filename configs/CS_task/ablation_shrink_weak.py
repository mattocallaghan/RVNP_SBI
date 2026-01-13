from configs.CS_task.cs_task import get_config as get_base_config

"""
ABLATION STUDY: Shrinkage Prior - Weak
CS task, RVNP-NN, nobs=100, lambda_shrinkage=0.01
"""

def get_config():
    config = get_base_config()

    # Standard experiment settings (nobs=100, NN)
    config.data.num_tests = 100
    config.data.num_iid = 1
    config.data.inference_simulations = 100
    config.model.correction_type = 'NN'
    config.model.K_obs_samples = 30
    config.model.simulator_samples_per_theta = 32
    config.training.final_epochs = 300
    config.training.final_posterior_epochs = 100

    # ABLATION: Shrinkage prior
    config.model.lambda_shrinkage = 0.01
    config.training.workspace = 'output/ablation_cs_shrink_weak_nobs100'

    return config
