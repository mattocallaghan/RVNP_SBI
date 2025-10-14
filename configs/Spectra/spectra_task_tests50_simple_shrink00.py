from configs.Spectra.spectra_task import get_config as get_base_config

"""
Spectra Task configuration variant: tests50_simple_shrink00
Inherits from base spectra_task.py and overrides specific variable terms
"""

def get_config():
    # Get base configuration from spectra_task.py
    config = get_base_config()
    
    #################################
    # Variable terms (override only these)
    #################################    
    config.model.correction_type = 'simple'  # 'simple', 'diagonal_neural', or 'hybrid'
    config.model.lambda_shrinkage = 0.0
    
    # Update workspace to reflect the specific configuration
    config.training.workspace = 'output/spectra_task_tests50_simple_shrink00'
    #################################    
    config.data.num_tests=50
    config.data.num_iid=1
    config.data.inference_simulations = int(config.data.num_tests*config.data.num_iid)      # For custom datasets (if applicable).

    config.model.correction_type = 'simple'  # 'simple', 'diagonal_neural', or 'hybrid'
    config.model.lambda_shrinkage = 0.0
    
    # Update workspace to reflect the specific configuration

    config.training.batch_size = 2**12
    config.training.final_epochs=500  # Epochs for stage 5 (main training)
    config.training.final_posterior_epochs=0  # Epochs for stage 6 (final posterior tuning)
    config.model.train_simulator=False       # Stage 2: Train simulator flow p(x_sim|θ)    
    config.model.train_embeddings=False       # Stage 2: Train simulator flow p(x_sim|θ)    



    return config