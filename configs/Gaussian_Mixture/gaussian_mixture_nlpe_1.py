


from ml_collections import ConfigDict

from configs.Gaussian_Mixture.gaussian_mixture import get_config as base_get_config
import ml_collections
import jax.numpy as jnp

"""
The main condition is for the scaling by sigma, so we dont condition but we do scale by sigma
"""

def get_config():
    # Start with the base configuration.
    config = base_get_config()
    config.data.num_iid=1
    config.data.num_tests=1
    config.data.inference_simulations = int(config.data.num_tests*config.data.num_iid)      # For custom datasets (if applicable).
    config.sampling.sampling_name='nlpe_iid='+str(config.data.num_iid)
    config.sampling.inference_method='Gaussian_Mixture'

    ##### sampling parameters 
    config.training.train=False
    config.sampling.inference=not config.training.train
    config.data.inference_dataset= 'gaussian_mixture_inference'
    config.training.validation_split=0.1
    #### Inference
       # Options: 'ode' or 'pc' (predictor-corrector).
    config.sampling.method = 'nuts'              # Options: 'ode' or 'pc' (predictor-corrector).
    config.sampling.step_size_initial=0.1
    config.sampling.warmup_steps=1000
    config.sampling.samples=1000
    config.sampling.adapt_step_size=True
    config.sampling.adapt_mass_matrix=True
    config.sampling.num_chains=1
    config.sampling.max_tree_depth=10
    config.sampling.acceptance_prob=0.8
    


    
    return config