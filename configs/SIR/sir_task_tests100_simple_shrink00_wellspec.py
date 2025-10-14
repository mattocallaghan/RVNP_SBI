from configs.SIR.sir_task import get_config as get_base_config

"""
SIR Task configuration variant: tests100_simple_shrink0.0_wellspec
Well-specified evaluation using training data as inference set
Inherits from base sir_task.py and overrides specific variable terms
"""

def get_config():
    # Get base configuration from sir_task.py
    config = get_base_config()
    
    #################################
    # Variable terms (override only these)
    #################################    
    config.data.num_tests=100
    config.data.num_iid=1
    config.data.inference_simulations = int(config.data.num_tests*config.data.num_iid) 
    config.model.correction_type = 'simple'  # 'simple', 'diagonal_neural', or 'hybrid'
    config.model.lambda_shrinkage = 0.0
    config.model.wellspec = True  # Flag for well-specified evaluation to distinguish model names
    
    # Use training data as inference set for well-specified evaluation
    config.data.inference_dataset = 'SIR'  # Use training data instead of 'SIR_inference'
    
    # Update workspace to reflect the specific configuration
    config.training.workspace = 'output/sir_task_tests100_simple_shrink00_wellspec'
    config.training.batch_size = 2**10
    config.training.final_epochs=300  # Epochs for stage 5 (main training)
    config.training.final_posterior_epochs=0  # Epochs for stage 6 (final posterior tuning)
    config.model.train_simulator=False      # Stage 2: Use pre-trained simulator flow p(x_sim|θ)    
    return config