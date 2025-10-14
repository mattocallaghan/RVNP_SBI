
"""Training and evaluation for RVNP. """

import jax
import datasets
import normalizing_flow
import os
import jax
import os
os.environ["JAX_PLATFORM_NAME"] = "gpu"
def train_flow(config):
  """Runs the training pipeline.
  Args:
    config: Configuration to use.
  """
  rng = jax.random.key(config.seed)
  
  # Check that num_tests exists
  if not hasattr(config.data, 'num_tests'):
    raise ValueError("config.data.num_tests must be defined")
  
  num_tests = config.data.num_tests
  
  # Determine experiment parameters based on num_tests
  if num_tests >= 100:
    # Normal case: run single experiment
    num_experiments = 1
    points_per_experiment = num_tests
    start_point_offset = 0
  elif num_tests == 1:
    # Special case: run 50 experiments with 1 point each
    num_experiments = 50
    points_per_experiment = 1
    start_point_offset = 0
  elif num_tests == 10:
    # Special case: run 10 experiments with 10 points each  
    num_experiments = 10
    points_per_experiment = 10
    start_point_offset = 0
  elif num_tests == 50:
    # Special case: run 2 experiments with 50 points each
    num_experiments = 2
    points_per_experiment = 50
    start_point_offset = 0
  else:
    raise ValueError(f"Unsupported num_tests value: {num_tests}. Supported values: 1, 10, 50, or >=100")
  
  print(f"Running {num_experiments} experiments with {points_per_experiment} points each")
  
  # Build shared training data (same across all experiments)
  train_ds, eval_ds, mean, std = datasets.get_dataset(config, load_data=True, return_summary=True)
  
  # Store original config values
  original_num_tests = config.data.num_tests
  original_inference_simulations = config.data.inference_simulations
  
  # For small num_tests, load full dataset (50+ points) and partition it
  if num_tests < 100:
    # Temporarily set config to load enough points for all experiments
    if num_tests == 50:
      total_points_needed = 100  # Need 100 points for 2 experiments of 50 each
    else:
      total_points_needed = 100  # Default for other cases
    config.data.num_tests = total_points_needed  
    config.data.inference_simulations = total_points_needed
    
  # Generate the full inference dataset
  full_inference_ds, full_inference_ds_eval = datasets.get_inference_dataset(config, True)
  
  # Run experiments
  for experiment_idx in range(num_experiments):
    print(f"\n=== Experiment {experiment_idx + 1}/{num_experiments} ===")
    
    # Update config for this experiment
    config.data.num_tests = points_per_experiment
    config.data.inference_simulations = points_per_experiment * config.data.num_iid
    
    # Pass experiment index to the flow constructor instead of modifying config
    
    # Calculate the starting point for this experiment
    # For num_tests=1: experiments use points 0,1,2,...,49
    # For num_tests=10: experiments use points 0-9, 10-19, 20-29, 30-39, 40-49
    current_start_point = start_point_offset + (experiment_idx * points_per_experiment)
    
    # Create experiment-specific random key
    experiment_rng = jax.random.fold_in(rng, experiment_idx)
    
    # Partition the inference data for this experiment
    if num_tests < 100:
      # Extract the subset of points for this experiment
      start_idx = current_start_point
      end_idx = current_start_point + points_per_experiment
      
      # Slice the full dataset to get the points for this experiment
      import jax.numpy as jnp
      inference_data_subset = jnp.array(full_inference_ds_eval)[start_idx:end_idx]
      inference_ds_eval = inference_data_subset
      inference_ds = inference_data_subset  # Use same data for train and eval in inference
    else:
      # For large num_tests, use the full dataset
      inference_ds, inference_ds_eval = full_inference_ds, full_inference_ds_eval
    
    # Create and train flow for this experiment
    experiment_idx_for_filename = experiment_idx if num_experiments > 1 else None
    flow_class = normalizing_flow.get_flow(config.model.name)(config, experiment_rng, experiment_idx=experiment_idx_for_filename)
    flow = flow_class.build(train_data=train_ds, eval_data=eval_ds, 
                           inference_data=inference_ds_eval, mean=mean, std=std)
    
    print(f"Completed experiment {experiment_idx + 1} using points {current_start_point} to {current_start_point + points_per_experiment - 1}")
  
  # Restore original config values
  config.data.num_tests = original_num_tests
  config.data.inference_simulations = original_inference_simulations
  
  print(f"\nCompleted all {num_experiments} experiments")

