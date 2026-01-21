from configs.CS_task.cs_task import get_config as get_base_config

"""
CS Task configuration variant: tests100_hybrid_shrink0.0_transform
Inherits from base cs_task.py and enables parameter space transformation
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
    config.model.correction_type = 'simple'  # 'simple', 'diagonal_neural', or 'hybrid'
    config.model.lambda_shrinkage = 0.1

    # NEW: Enable parameter space transformation
    config.model.transform_param_space = True

    # Update workspace to reflect the specific configuration
    config.training.workspace = 'output/cs_task_tests100_hybrid_shrink00_transform'
    config.training.batch_size = 2**10
    config.training.final_epochs=10000  # Epochs for stage 5 (main training)
    config.training.final_posterior_epochs=0  # Epochs for stage 6 (final posterior tuning)
    config.model.train_simulator=False       # Stage 2: Train simulator flow p(x_sim|θ)

    config.optim.lr = 1e-4
    config.optim.optimizer = 'Adam'
    config.optim.beta1 = 0.9 #originally 0.9
    config.optim.eps = 1e-8
    config.optim.weight_decay = 1e-5 
    config.optim.warmup = 100
    config.optim.grad_clip = float('inf')  # Disabled - no gradient clipping


    # Observation sampling
    config.model.K_obs_samples = 5  # Increased for better sampling diversity (matches NLPE)
    # Simulator sampling
    config.model.use_simulator_sampling = True
    config.model.simulator_samples_per_theta = 5  # Default for small datasets


    config.optim.use_lr_scheduler = True
    config.optim.scheduler_initial_lr = 1e-3     # Start at high LR
    config.optim.scheduler_patience = 500         # 20 epochs without improvement
    config.optim.scheduler_factor = 0.1          # Divide by 10 each reduction
    config.optim.scheduler_min_lr = 1e-5         # Stop at 1e-4
    return config
