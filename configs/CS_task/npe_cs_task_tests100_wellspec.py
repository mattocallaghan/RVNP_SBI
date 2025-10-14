from configs.CS_task.npe_cs_task import get_config as get_base_config

"""
NPE CS Task configuration variant: tests100_wellspec
Well-specified evaluation using training data as inference set
Inherits from base npe_cs_task.py and overrides only the inference dataset
"""

def get_config():
    # Get base configuration from npe_cs_task.py
    config = get_base_config()
    
    # Only change the inference dataset - everything else stays the same
    config.data.inference_dataset = 'CS'  # Use training data instead of 'CS_inference'
    config.model.wellspec = True  # Flag for well-specified evaluation to distinguish model names
    
    # Update workspace to reflect wellspec configuration
    config.training.workspace = 'output/npe_cs_task_tests100_wellspec'
    
    return config