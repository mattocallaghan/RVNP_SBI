from configs.Pendulum.pendulum_task import get_config as get_base_config

"""
Pendulum Task: RVNP-NN with nobs=100, lambda_shrinkage=0.0
"""

def get_config():
    config = get_base_config()

    # Experiment-specific overrides
    config.data.num_tests = 100
    config.data.num_iid = 1
    config.data.inference_simulations = 100

    config.model.correction_type = 'NN'
    config.model.lambda_shrinkage = 0.0

    # Workspace
    config.training.workspace = 'output/pendulum_task_tests100_NN'

    # Medium/Large dataset parameters
    config.model.K_obs_samples = 30
    config.model.simulator_samples_per_theta = 32
    config.training.final_epochs = 300
    config.training.final_posterior_epochs = 100

    return config
