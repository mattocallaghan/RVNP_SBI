from configs.gaussian_embedding.gaussian_task import get_config as get_base_config

"""
Gaussian_Embedding Task configuration variant: tests10_simple_shrink00
Inherits from base gaussian_task.py and overrides specific variable terms
"""

def get_config():
    # Get base configuration from gaussian_task.py
    config = get_base_config()
    
    #################################
    # Variable terms (override only these)
    #################################    
    config.data.num_iid=1
    config.data.num_tests = 10
    config.data.inference_simulations = int(config.data.num_tests*config.data.num_iid)
    config.model.correction_type = 'simple'  # 'simple', 'diagonal_neural', or 'hybrid'
    config.model.lambda_shrinkage = 0.0
    config.training.batch_size = 2**10 
    config.training.final_epochs=300  # Epochs for stage 5 (main training)
    config.training.final_posterior_epochs=0  # Epochs for stage 6 (final posterior tuning)
    # Update workspace to reflect the specific configuration
    config.training.workspace = 'output/gaussian_task_tests10_simple_shrink00'
    config.model.train_simulator=False
    return config