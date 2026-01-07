"""Training utilities for RVNP-SBI: logging, timing, and experiment management."""

import logging
import time
import os
from contextlib import contextmanager
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    """Configuration for partitioning evaluation points across multiple experiments.

    Determines how many independent experiments to run and how many test points
    each should use. Used to manage computational resources and memory for large
    evaluation runs.

    Attributes:
        num_experiments: Number of independent experiment runs
        points_per_experiment: Test points per individual experiment
        total_points_needed: Total test points across all experiments

    Strategy:
        Balances number of experiments with points per experiment to achieve
        statistically meaningful results while managing computational cost:

        - num_tests >= 100: 1 experiment (single large run)
        - num_tests == 50: 2 experiments × 50 points = 100 total
        - num_tests == 10: 10 experiments × 10 points = 100 total
        - num_tests == 1: 50 experiments × 1 point = 50 total

    Example:
        >>> config = ExperimentConfig.from_num_tests(num_tests=10)
        >>> print(config.num_experiments)  # 10
        >>> print(config.points_per_experiment)  # 10
        >>> print(config.total_points_needed)  # 100
    """
    num_experiments: int
    points_per_experiment: int
    total_points_needed: int

    @classmethod
    def from_num_tests(cls, num_tests: int) -> 'ExperimentConfig':
        """
        Determine experiment partitioning strategy based on num_tests.

        Strategy:
        - num_tests >= 100: Single experiment with all points
        - num_tests == 50: 2 experiments × 50 points each (100 points total)
        - num_tests == 10: 10 experiments × 10 points each (100 points total)
        - num_tests == 1: 50 experiments × 1 point each (50 points total)

        Args:
            num_tests: Number of test points per experiment

        Returns:
            ExperimentConfig with partitioning parameters
        """
        if num_tests >= 100:
            return cls(
                num_experiments=1,
                points_per_experiment=num_tests,
                total_points_needed=num_tests
            )
        elif num_tests == 50:
            return cls(
                num_experiments=2,
                points_per_experiment=50,
                total_points_needed=100
            )
        elif num_tests == 10:
            return cls(
                num_experiments=10,
                points_per_experiment=10,
                total_points_needed=100
            )
        elif num_tests == 1:
            return cls(
                num_experiments=50,
                points_per_experiment=1,
                total_points_needed=50
            )
        else:
            raise ValueError(
                f"Unsupported num_tests value: {num_tests}. "
                f"Supported values: 1, 10, 50, or >=100"
            )


class TrainingLogger:
    """Structured logging for RVNP training with timing support."""

    def __init__(self, experiment_name: str, log_dir: str = "logs"):
        """
        Initialize training logger.

        Args:
            experiment_name: Name for this training run
            log_dir: Directory to save log files
        """
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.epoch_times = []
        self.stage_times = {}
        self._setup_logger()

    def _setup_logger(self):
        """Configure logging with file and console handlers."""
        os.makedirs(self.log_dir, exist_ok=True)
        log_file = os.path.join(self.log_dir, f"{self.experiment_name}.log")

        # Create logger
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.INFO)

        # Remove existing handlers to avoid duplicates
        self.logger.handlers = []

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Prevent duplicate log messages from propagating to root/absl logger
        self.logger.propagate = False

    @contextmanager
    def timer(self, name: str, log_on_exit: bool = True):
        """
        Context manager for timing code blocks.

        Args:
            name: Name of the timed operation
            log_on_exit: Whether to log when exiting context

        Yields:
            Dictionary that will contain elapsed time

        Example:
            with logger.timer("Data loading") as timer_info:
                load_data()
            # Automatically logs: "Data loading completed in X.XXs"
        """
        start_time = time.time()
        timer_info = {'elapsed': 0.0}

        try:
            yield timer_info
        finally:
            elapsed = time.time() - start_time
            timer_info['elapsed'] = elapsed
            self.stage_times[name] = elapsed

            if log_on_exit:
                self.logger.info(f"{name} completed in {elapsed:.2f}s")

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float,
                  metrics: Optional[Dict[str, float]] = None):
        """
        Log epoch training metrics.

        Args:
            epoch: Epoch number
            train_loss: Training loss value
            val_loss: Validation loss value
            metrics: Optional dictionary of additional metrics
        """
        msg = f"Epoch {epoch:04d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"

        if metrics:
            metric_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            msg += f" | {metric_str}"

        self.logger.info(msg)

    def log_stage(self, stage_name: str, **kwargs):
        """
        Log training stage information.

        Args:
            stage_name: Name of the training stage
            **kwargs: Additional information to log
        """
        info_str = " | ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.info(f"=== {stage_name} === | {info_str}")

    def log_summary(self):
        """Log summary of all timed operations."""
        if not self.stage_times:
            return

        self.logger.info("=" * 60)
        self.logger.info("Training Summary - Timing Breakdown")
        self.logger.info("=" * 60)

        total_time = sum(self.stage_times.values())

        for stage_name, elapsed in sorted(self.stage_times.items()):
            percentage = (elapsed / total_time * 100) if total_time > 0 else 0
            self.logger.info(f"{stage_name:40s}: {elapsed:8.2f}s ({percentage:5.1f}%)")

        self.logger.info("-" * 60)
        self.logger.info(f"{'Total Time':40s}: {total_time:8.2f}s")
        self.logger.info("=" * 60)


def validate_config(config: Any) -> bool:
    """
    Validate RVNP configuration before training.

    Args:
        config: Configuration object

    Returns:
        True if validation passes

    Raises:
        ValueError: If validation fails with detailed error message
    """
    errors = []

    # Required fields with expected types
    required_fields = [
        ('seed', int, None),
        ('data.dataset', str, None),
        ('data.num_simulations', int, lambda x: x > 0),
        ('data.num_tests', int, lambda x: x > 0),
        ('data.inference_simulations', int, lambda x: x > 0),
        ('data.num_iid', int, lambda x: x > 0),
        ('data.data_path', str, None),
        ('model.name', str, None),
        ('model.correction_type', str, None),
        ('model.flow_dimension', int, lambda x: x > 0),
        ('model.cond_dim', int, lambda x: x > 0),
        ('training.batch_size', int, lambda x: x > 0),
        ('training.n_iters', int, lambda x: x > 0),
        ('optim.lr', (int, float), lambda x: x > 0),
    ]

    # Check required fields
    for field_path, expected_type, validator in required_fields:
        try:
            obj = config
            for part in field_path.split('.'):
                obj = getattr(obj, part)

            # Type check
            if not isinstance(obj, expected_type):
                type_name = expected_type.__name__ if not isinstance(expected_type, tuple) else ' or '.join(t.__name__ for t in expected_type)
                errors.append(
                    f"{field_path} must be {type_name}, got {type(obj).__name__}"
                )

            # Value validation
            if validator and not validator(obj):
                errors.append(f"{field_path} failed validation: value={obj}")

        except AttributeError:
            errors.append(f"Missing required field: {field_path}")

    # Additional validation checks (only run if fields exist)
    # Check correction type validity (must be one of the allowed types)
    try:
        valid_corrections = ['simple', 'diagonal_neural', 'hybrid', 'neural', 'mu_hybrid', 'global', 'full_neural']
        if config.model.correction_type not in valid_corrections:
            errors.append(
                f"model.correction_type must be one of {valid_corrections}, "
                f"got '{config.model.correction_type}'"
            )
    except AttributeError:
        pass  # Already caught by required fields check

    # Check num_tests validity (must be compatible with ExperimentConfig)
    try:
        ExperimentConfig.from_num_tests(config.data.num_tests)
    except ValueError as e:
        errors.append(str(e))
    except AttributeError:
        pass  # Already caught by required fields check

    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)

    return True


def format_experiment_name(config: Any, experiment_idx: Optional[int] = None) -> str:
    """
    Create a standardized experiment name from config.

    Args:
        config: Configuration object
        experiment_idx: Optional experiment index

    Returns:
        Formatted experiment name string
    """
    parts = [
        config.model.name,
        config.data.dataset,
        f"n{config.data.num_simulations}"
    ]

    if hasattr(config.data, 'num_tests'):
        parts.append(f"tests{config.data.num_tests}")

    if hasattr(config.model, 'correction_type'):
        parts.append(config.model.correction_type)

    if experiment_idx is not None:
        parts.append(f"exp{experiment_idx}")

    return "_".join(parts)
