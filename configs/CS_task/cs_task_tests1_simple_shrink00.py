from configs.CS_task.cs_task import get_config as get_base_config

"""
CS Task: RVNP-simple with nobs=1, lambda_shrinkage=0.0
"""

def get_config():
    config = get_base_config()

    # Experiment-specific overrides
    config.data.num_tests = 1
    config.data.num_iid = 1
    config.data.inference_simulations = 1

    config.model.correction_type = 'simple'
    config.model.lambda_shrinkage = 0.0

    # Workspace
    config.training.workspace = 'output/cs_task_tests1_simple'

    # Small dataset parameters
    config.model.K_obs_samples = 10
    config.model.simulator_samples_per_theta = 10
    config.training.final_epochs = 300
    config.training.final_posterior_epochs = 100

    return config
