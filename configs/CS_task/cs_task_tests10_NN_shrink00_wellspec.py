from configs.CS_task.cs_task import get_config as get_base_config

"""
CS Task configuration variant: tests10_hybrid_shrink0.0_wellspec
Well-specified evaluation using training data as inference set
Inherits from base cs_task.py and overrides specific variable terms
"""

def get_config():
    # Get base configuration from cs_task.py
    config = get_base_config()
    
    #################################
    # Variable terms (override only these)
    #################################    
    config.data.num_tests=10
    config.data.num_iid=1
    config.data.inference_simulations = int(config.data.num_tests*config.data.num_iid) 
    config.model.correction_type = 'NN'  # 'simple', 'diagonal_neural', or 'hybrid'
    config.model.lambda_shrinkage = 0.0
    config.model.wellspec = True  # Flag for well-specified evaluation to distinguish model names
    
    # Use training data as inference set for well-specified evaluation
    config.data.inference_dataset = 'CS'  # Use training data instead of 'CS_inference'
    
    # Update workspace to reflect the specific configuration
    config.training.workspace = 'output/cs_task_tests10_hybrid_shrink00_wellspec'
    config.training.batch_size = 2**10 
    config.training.warmup_epochs=5  # Epochs for stages 1-4.5 (warmup)
    config.training.final_epochs=0  # Epochs for stage 5 (main training)
    config.training.final_posterior_epochs=0  # Epochs for stage 6 (final posterior tuning)
    config.model.train_simulator=False       # Stage 2: Train simulator flow p(x_sim|θ)    
    return config