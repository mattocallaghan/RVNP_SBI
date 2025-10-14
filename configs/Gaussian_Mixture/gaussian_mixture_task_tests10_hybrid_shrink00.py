from configs.Gaussian_Mixture.gaussian_mixture_task import get_config as get_base_config

"""
Gaussian_Mixture Task configuration variant: tests10_hybrid_shrink00
Inherits from base gaussian_mixture_task.py and overrides specific variable terms
"""

def get_config():
    # Get base configuration from gaussian_mixture_task.py
    config = get_base_config()
    
    #################################
    # Variable terms (override only these)
    #################################    
    config.data.num_tests = 10
    config.model.correction_type = 'hybrid'  # 'simple', 'diagonal_neural', or 'hybrid'
    config.model.lambda_shrinkage = 0.0
    
    # Update workspace to reflect the specific configuration
    config.training.workspace = 'output/gaussian_mixture_task_tests10_hybrid_shrink00'
    
    return config