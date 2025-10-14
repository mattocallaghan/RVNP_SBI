from configs.Pendulum.pendulum_task import get_config as get_base_config

"""
Pendulum Task configuration variant: tests1000_simple_shrink00
Inherits from base pendulum_task.py and overrides specific variable terms
"""

def get_config():
    # Get base configuration from pendulum_task.py
    config = get_base_config()
    
    #################################
    # Variable terms (override only these)
    #################################    
    config.data.num_tests = 1000
    config.model.correction_type = 'simple'  # 'simple', 'diagonal_neural', or 'hybrid'
    config.model.lambda_shrinkage = 0.0
    config.data.num_iid=1
    config.data.num_tests = 1000
    config.data.inference_simulations = int(config.data.num_tests*config.data.num_iid)
    config.training.batch_size = 2**11 
    config.training.final_epochs=500  # Epochs for stage 5 (main training)
    config.training.final_posterior_epochs=0  # Epochs for stage 6 (final posterior tuning)
    config.model.train_simulator=False
    config.model.train_embeddings=False
    # Update workspace to reflect the specific configuration
    config.training.workspace = 'output/pendulum_task_tests1000_simple_shrink00'
    
    return config