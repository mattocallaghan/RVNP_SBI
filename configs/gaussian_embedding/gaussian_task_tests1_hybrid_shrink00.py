from configs.gaussian_embedding.gaussian_task import get_config as get_base_config

"""
Gaussian_Embedding Task configuration variant: tests1_hybrid_shrink00
Inherits from base gaussian_task.py and overrides specific variable terms
"""

def get_config():
    # Get base configuration from gaussian_task.py
    config = get_base_config()
    
    #################################
    # Variable terms (override only these)
    #################################    
    
    config.model.correction_type = 'hybrid'  # 'simple', 'diagonal_neural', or 'hybrid'
    config.model.lambda_shrinkage = 0.0

    config.data.num_iid=1
    config.data.num_tests = 1
    config.data.inference_simulations = int(config.data.num_tests*config.data.num_iid)
    config.training.batch_size = 2**10 
    config.training.final_epochs=300  # Epochs for stage 5 (main training)
    config.training.final_posterior_epochs=0  # Epochs for stage 6 (final posterior tuning)
    config.model.train_simulator=False

    # Update workspace to reflect the specific configuration
    config.training.workspace = 'output/gaussian_task_tests1_hybrid_shrink00'

    return config