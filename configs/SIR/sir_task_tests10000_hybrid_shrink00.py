from configs.SIR.sir_task import get_config as get_base_config

"""
SIR Task configuration variant: tests1000_simple_shrink00
Inherits from base sir_task.py and overrides specific variable terms
"""

def get_config():
    # Get base configuration from sir_task.py
    config = get_base_config()
    
    #################################
    # Variable terms (override only these)
    #################################    
    config.data.num_tests = 10000
    config.model.correction_type = 'hybrid'  # 'simple', 'diagonal_neural', or 'hybrid'
    config.model.lambda_shrinkage = 0.0
    config.data.num_iid=1
    config.data.inference_simulations = int(config.data.num_tests*config.data.num_iid)      # For custom datasets (if applicable).

    # Update workspace to reflect the specific configuration
    config.training.workspace = 'output/sir_task_tests1000_simple_shrink00'
    config.training.batch_size = 2**10
    config.training.final_epochs=500  # Epochs for stage 5 (main training)
    config.training.final_posterior_epochs=0  # Epochs for stage 6 (final posterior tuning)
    config.model.train_simulator=False   
    return config