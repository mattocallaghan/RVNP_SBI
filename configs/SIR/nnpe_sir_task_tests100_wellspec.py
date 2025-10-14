from configs.SIR.nnpe_sir_task import get_config as get_base_config

"""
NNPE SIR Task configuration variant: tests100_wellspec
Well-specified evaluation using training data as inference set
Inherits from base nnpe_sir_task.py and overrides only the inference dataset
"""

def get_config():
    # Get base configuration from nnpe_sir_task.py
    config = get_base_config()
    
    # Only change the inference dataset - everything else stays the same
    config.data.inference_dataset = 'SIR'  # Use training data instead of 'SIR_inference'
    config.model.wellspec = True  # Flag for well-specified evaluation to distinguish model names
    
    # Update workspace to reflect wellspec configuration
    config.training.workspace = 'output/nnpe_sir_task_tests100_wellspec'
    
    return config