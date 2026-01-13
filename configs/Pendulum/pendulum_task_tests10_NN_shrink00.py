from configs.Pendulum.pendulum_task import get_config as get_base_config

"""
Pendulum Task: RVNP-NN with nobs=10, lambda_shrinkage=0.0
"""

def get_config():
    config = get_base_config()

    # Experiment-specific overrides
    config.data.num_tests = 10
    config.data.num_iid = 1
    config.data.inference_simulations = 10

    config.model.correction_type = 'NN'
    config.model.lambda_shrinkage = 0.0

    # Workspace
    config.training.workspace = 'output/pendulum_task_tests10_NN'

    # Small dataset parameters
    config.model.K_obs_samples = 10
    config.model.simulator_samples_per_theta = 10
    config.training.final_epochs = 300
    config.training.final_posterior_epochs = 100

    return config
