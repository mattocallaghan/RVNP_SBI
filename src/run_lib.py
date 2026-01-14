"""Training and evaluation for RVNP."""

import jax
import jax.numpy as jnp
import os
from . import datasets
from . import normalizing_flow
from .training_utils import ExperimentConfig, TrainingLogger, validate_config, format_experiment_name
os.environ["JAX_PLATFORM_NAME"] = "gpu"


def load_shared_training_data(config, logger):
    """
    Load training and validation datasets shared across all experiments.

    Args:
        config: Configuration object
        logger: TrainingLogger instance

    Returns:
        Dictionary containing train_ds, eval_ds, mean, std
    """
    with logger.timer("Loading shared training data"):
        train_ds, eval_ds, mean, std = datasets.get_dataset(
            config, load_data=True, return_summary=True
        )

    logger.logger.info(
        f"Loaded simulation data: {len(train_ds)} train batches, "
        f"{len(eval_ds)} eval batches"
    )

    return {
        'train_ds': train_ds,
        'eval_ds': eval_ds,
        'mean': mean,
        'std': std
    }


def load_inference_data(config, exp_config, logger):
    """
    Load raw (unbatched) inference dataset for all experiments.

    Returns raw array because normalizing_flow.py does its own batching internally.

    Args:
        config: Configuration object
        exp_config: ExperimentConfig instance
        logger: TrainingLogger instance

    Returns:
        Raw inference array (unbatched)
    """
    with logger.timer("Loading inference data"):
        # Load raw data directly using load_custom_dataset (before batching)
        inference_dataset_name = config.data.inference_dataset

        # load_custom_dataset returns (train_ds, eval_ds) split
        # For inference=True, p=1, so all data goes to eval_ds
        _, raw_inference_data = datasets.load_custom_dataset(
            inference_dataset_name,
            config,
            num_simulations=exp_config.total_points_needed,
            load_data=True,
            return_summary=False,
            inference=True
        )

    logger.logger.info(
        f"Loaded test observations: {exp_config.total_points_needed} total observations for "
        f"{exp_config.num_experiments} experiments (raw unbatched array)"
    )

    return jnp.array(raw_inference_data)


def partition_inference_data_for_experiment(raw_inference_data, exp_idx, exp_config):
    """
    Extract the subset of inference data for a specific experiment.

    Args:
        raw_inference_data: Raw (unbatched) inference array
        exp_idx: Experiment index (0-based)
        exp_config: ExperimentConfig instance

    Returns:
        Inference data subset for this experiment (raw array slice)
    """
    start_idx = exp_idx * exp_config.points_per_experiment
    end_idx = start_idx + exp_config.points_per_experiment

    # Extract subset - simple array slicing since data is unbatched
    inference_data_subset = raw_inference_data[start_idx:end_idx]

    return inference_data_subset


def run_single_experiment(config, data_dict, inference_data_eval, experiment_rng,
                          exp_idx, exp_config, logger):
    """
    Execute training for a single experiment.

    Args:
        config: Configuration object
        data_dict: Dictionary with train_ds, eval_ds, mean, std
        inference_data_eval: Inference data for this experiment
        experiment_rng: Random key for this experiment
        exp_idx: Experiment index (0-based)
        exp_config: ExperimentConfig instance
        logger: TrainingLogger instance

    Returns:
        Trained flow instance
    """
    # Update config for this experiment
    config.data.num_tests = exp_config.points_per_experiment
    config.data.inference_simulations = exp_config.points_per_experiment * config.data.num_iid

    # Determine experiment filename suffix (None for single experiment case)
    experiment_idx_for_filename = exp_idx if exp_config.num_experiments > 1 else None

    # Calculate point range for logging
    start_point = exp_idx * exp_config.points_per_experiment
    end_point = start_point + exp_config.points_per_experiment - 1

    logger.logger.info(
        f"Training experiment {exp_idx + 1}/{exp_config.num_experiments} "
        f"using points {start_point} to {end_point}"
    )

    # Create and build flow
    with logger.timer(f"Experiment {exp_idx + 1} - Flow initialization"):
        flow_class = normalizing_flow.get_flow(config.model.name)(
            config, experiment_rng, experiment_idx=experiment_idx_for_filename
        )

    with logger.timer(f"Experiment {exp_idx + 1} - Training"):
        flow = flow_class.build(
            train_data=data_dict['train_ds'],
            eval_data=data_dict['eval_ds'],
            inference_data=inference_data_eval,
            mean=data_dict['mean'],
            std=data_dict['std']
        )

    logger.logger.info(
        f"Completed experiment {exp_idx + 1}/{exp_config.num_experiments}"
    )

    return flow


def train_flow(config):
    """
    Main training pipeline for RVNP.

    Orchestrates multi-experiment training with proper data partitioning,
    logging, and timing.

    Args:
        config: Configuration object with all training parameters
    """
    # Validate configuration
    validate_config(config)

    # Setup logging
    experiment_name = format_experiment_name(config)
    logger = TrainingLogger(experiment_name)

    logger.logger.info("=" * 80)
    logger.logger.info("RVNP Training Pipeline")
    logger.logger.info("=" * 80)
    logger.logger.info(f"Model: {config.model.name}")
    logger.logger.info(f"Dataset: {config.data.dataset}")
    logger.logger.info(f"Training simulations (θ, x): {config.data.num_simulations:,}")
    logger.logger.info(f"Test observations: {config.data.num_tests}")
    logger.logger.info(f"Correction type: {config.model.correction_type}")
    logger.logger.info("=" * 80)

    # Initialize random key
    rng = jax.random.key(config.seed)

    # Determine experiment partitioning strategy
    exp_config = ExperimentConfig.from_num_tests(config.data.num_tests)
    logger.logger.info(
        f"Experiment strategy: {exp_config.num_experiments} experiments × "
        f"{exp_config.points_per_experiment} observations = "
        f"{exp_config.total_points_needed} total observations"
    )

    # Load shared training data (same across all experiments)
    data_dict = load_shared_training_data(config, logger)

    # Load raw inference data (unbatched)
    raw_inference_data = load_inference_data(config, exp_config, logger)

    # Run experiments
    logger.logger.info("")
    logger.logger.info("Starting experiment training...")
    logger.logger.info("")

    for exp_idx in range(exp_config.num_experiments):
        logger.logger.info("-" * 80)
        logger.logger.info(f"EXPERIMENT {exp_idx + 1}/{exp_config.num_experiments}")
        logger.logger.info("-" * 80)

        # Create experiment-specific random key
        experiment_rng = jax.random.fold_in(rng, exp_idx)

        # Partition inference data for this experiment (slice raw array)
        inference_data_eval = partition_inference_data_for_experiment(
            raw_inference_data, exp_idx, exp_config
        )

        # Run single experiment
        flow = run_single_experiment(
            config=config,
            data_dict=data_dict,
            inference_data_eval=inference_data_eval,
            experiment_rng=experiment_rng,
            exp_idx=exp_idx,
            exp_config=exp_config,
            logger=logger
        )

        logger.logger.info("")

    # Final summary
    logger.logger.info("=" * 80)
    logger.logger.info(f"Completed all {exp_config.num_experiments} experiments")
    logger.logger.info("=" * 80)
    logger.log_summary()
