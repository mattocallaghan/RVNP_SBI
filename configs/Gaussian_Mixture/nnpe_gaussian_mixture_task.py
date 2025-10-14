from configs.Gaussian_Mixture.gaussian_mixture_task import get_config as get_base_config

"""
NNPE Gaussian_Mixture Task configuration - uses noisy neural posterior estimation
"""

def get_config():
    config = get_base_config()
    
    # Override model name for NNPE
    config.model.name = 'noisy_neural_posterior'
    
    # Update workspace to reflect NNPE variant
    config.training.workspace = 'output/nnpe_gaussian_mixture_task'
    
    return config