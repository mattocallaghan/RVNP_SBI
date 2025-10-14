from configs.CS_task.cs_task import get_config as get_base_config

"""
CS Task configuration variant: tests100_hybrid_shrink0.0
Inherits from base cs_task.py and overrides specific variable terms
"""

def get_config():
    # Get base configuration from cs_task.py
    config = get_base_config()
    
    #################################
    # Variable terms (override only these)
    #################################    
    config.data.num_tests=100
    config.data.num_iid=1
    config.data.inference_simulations = int(config.data.num_tests*config.data.num_iid) 
    config.model.correction_type = 'hybrid'  # 'simple', 'diagonal_neural', or 'hybrid'
    config.model.lambda_shrinkage = 0.0
    
    # Update workspace to reflect the specific configuration
    config.training.workspace = 'output/cs_task_tests100_hybrid_shrink00'
    config.training.batch_size = 2**10 
    config.training.final_epochs=300  # Epochs for stage 5 (main training)
    config.training.final_posterior_epochs=0  # Epochs for stage 6 (final posterior tuning)
    config.model.train_simulator=False       # Stage 2: Train simulator flow p(x_sim|θ)    
    return config