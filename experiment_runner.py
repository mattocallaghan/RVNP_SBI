"""
Experiment Runner for ICML 2026 Publication
Systematically runs all experiments across tasks, Nobs values, and methods.
"""

import os
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('experiment_runner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentSpec:
    """Specification for a single experiment."""
    task: str
    nobs: int
    method: str
    correction_type: str  # 'simple', 'hybrid', 'global', 'none'
    config_path: str
    output_dir: str

    def to_dict(self):
        return asdict(self)


class ExperimentRunner:
    """Orchestrates running all publication experiments."""

    # Define experimental matrix from paper
    TASKS = ['CS', 'SIR', 'Pendulum', 'Spectra']
    NOBS_VALUES = [1, 10, 100, 1000, 10000]

    # Method configurations
    # (method_name, model_name, correction_type, description)
    METHODS = [
        ('RVNP', 'ranpt', 'simple', 'RVNP with diagonal covariance'),
        ('RVNP-T', 'ranpt', 'hybrid', 'RVNP-Tuned with neural covariance'),
        ('RVNP-G', 'ranpt', 'global', 'RVNP with global correction'),
        ('NPE', 'npe', 'none', 'Standard Neural Posterior Estimation'),
        ('NNPE', 'ranpt', 'simple', 'Noisy Neural Posterior Estimation'),
    ]

    def __init__(self, base_output_dir: str = 'experiments_output',
                 results_dir: str = 'experiment_results'):
        """
        Initialize experiment runner.

        Args:
            base_output_dir: Directory for model outputs and checkpoints
            results_dir: Directory for collected results and metrics
        """
        self.base_output_dir = Path(base_output_dir)
        self.results_dir = Path(results_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)

        # Track experiment status
        self.status_file = self.results_dir / 'experiment_status.json'
        self.status = self._load_status()

    def _load_status(self) -> Dict:
        """Load experiment status from file."""
        if self.status_file.exists():
            with open(self.status_file, 'r') as f:
                return json.load(f)
        return {'completed': [], 'failed': [], 'running': None}

    def _save_status(self):
        """Save experiment status to file."""
        with open(self.status_file, 'w') as f:
            json.dump(self.status, f, indent=2)

    def generate_experiment_matrix(self) -> List[ExperimentSpec]:
        """
        Generate full experimental matrix for publication.

        Returns:
            List of ExperimentSpec objects
        """
        experiments = []

        for task in self.TASKS:
            # Determine valid Nobs for this task
            if task == 'Spectra':
                # Spectra uses real data - only certain Nobs values may be available
                valid_nobs = [1, 10, 100]  # Adjust based on actual data availability
            else:
                valid_nobs = self.NOBS_VALUES

            for nobs in valid_nobs:
                for method_name, model_name, correction_type, description in self.METHODS:
                    # Generate config path
                    config_path = self._get_config_path(task, nobs, method_name, correction_type)

                    # Generate output directory
                    output_dir = self.base_output_dir / f"{task}_{method_name}_nobs{nobs}"

                    experiments.append(ExperimentSpec(
                        task=task,
                        nobs=nobs,
                        method=method_name,
                        correction_type=correction_type,
                        config_path=config_path,
                        output_dir=str(output_dir)
                    ))

        return experiments

    def _get_config_path(self, task: str, nobs: int, method: str,
                         correction_type: str) -> str:
        """
        Get configuration file path for an experiment.

        Args:
            task: Task name (CS, SIR, Pendulum, Spectra)
            nobs: Number of observations
            method: Method name (RVNP, NPE, NNPE, etc.)
            correction_type: Correction type (simple, hybrid, global, none)

        Returns:
            Path to config file
        """
        # Map task names to config directories
        task_dir_map = {
            'CS': 'CS_task',
            'SIR': 'SIR',
            'Pendulum': 'Pendulum',
            'Spectra': 'Spectra'
        }

        task_dir = task_dir_map.get(task, task)

        # Build config filename based on method
        if method == 'NPE':
            config_name = f"npe_{task.lower()}_task.py"
        elif method == 'NNPE':
            config_name = f"nnpe_{task.lower()}_task.py"
        else:  # RVNP variants
            # Check if ranpt_{nobs}_{correction_type}.py exists
            config_name = f"ranpt_{nobs}_{correction_type}.py"

        config_path = f"configs/{task_dir}/{config_name}"
        return config_path

    def run_experiment(self, spec: ExperimentSpec, dry_run: bool = False) -> bool:
        """
        Run a single experiment.

        Args:
            spec: Experiment specification
            dry_run: If True, only print command without executing

        Returns:
            True if successful, False otherwise
        """
        exp_id = f"{spec.task}_{spec.method}_nobs{spec.nobs}"

        # Check if already completed
        if exp_id in self.status['completed']:
            logger.info(f"Skipping {exp_id} - already completed")
            return True

        logger.info("=" * 80)
        logger.info(f"Running experiment: {exp_id}")
        logger.info(f"Config: {spec.config_path}")
        logger.info(f"Output: {spec.output_dir}")
        logger.info("=" * 80)

        # Build command
        cmd = [
            'python', 'main_train_eval.py',
            f'--config={spec.config_path}',
            '--mode=train'
        ]

        if dry_run:
            logger.info(f"DRY RUN: {' '.join(cmd)}")
            return True

        # Update status
        self.status['running'] = exp_id
        self._save_status()

        # Run experiment
        try:
            start_time = time.time()
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            elapsed = time.time() - start_time

            logger.info(f"✓ Completed {exp_id} in {elapsed:.1f}s")

            # Save experiment info
            self._save_experiment_info(spec, elapsed, success=True)

            # Update status
            self.status['completed'].append(exp_id)
            self.status['running'] = None
            self._save_status()

            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed {exp_id}")
            logger.error(f"Error: {e.stderr}")

            # Save failure info
            self._save_experiment_info(spec, 0, success=False, error=str(e))

            # Update status
            self.status['failed'].append(exp_id)
            self.status['running'] = None
            self._save_status()

            return False

    def _save_experiment_info(self, spec: ExperimentSpec, elapsed: float,
                              success: bool, error: str = None):
        """Save experiment metadata."""
        info = {
            'spec': spec.to_dict(),
            'elapsed_seconds': elapsed,
            'success': success,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        if error:
            info['error'] = error

        exp_id = f"{spec.task}_{spec.method}_nobs{spec.nobs}"
        info_file = self.results_dir / f"{exp_id}_info.json"

        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)

    def run_all_experiments(self, dry_run: bool = False,
                           filter_task: str = None,
                           filter_method: str = None,
                           filter_nobs: int = None):
        """
        Run all experiments in the experimental matrix.

        Args:
            dry_run: If True, only print commands without executing
            filter_task: Optional task filter (e.g., 'CS')
            filter_method: Optional method filter (e.g., 'RVNP')
            filter_nobs: Optional Nobs filter (e.g., 1)
        """
        experiments = self.generate_experiment_matrix()

        # Apply filters
        if filter_task:
            experiments = [e for e in experiments if e.task == filter_task]
        if filter_method:
            experiments = [e for e in experiments if e.method == filter_method]
        if filter_nobs:
            experiments = [e for e in experiments if e.nobs == filter_nobs]

        logger.info(f"Total experiments to run: {len(experiments)}")

        if dry_run:
            logger.info("\n=== DRY RUN MODE - Commands that would be executed ===\n")

        # Run experiments
        successful = 0
        failed = 0

        for i, spec in enumerate(experiments, 1):
            logger.info(f"\n[{i}/{len(experiments)}]")

            success = self.run_experiment(spec, dry_run=dry_run)
            if success:
                successful += 1
            else:
                failed += 1

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("EXPERIMENT RUNNER SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total experiments: {len(experiments)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Results saved to: {self.results_dir}")
        logger.info("=" * 80)

    def print_experiment_matrix(self):
        """Print the full experimental matrix."""
        experiments = self.generate_experiment_matrix()

        print("\n" + "=" * 80)
        print("EXPERIMENTAL MATRIX FOR ICML 2026")
        print("=" * 80)
        print(f"Total experiments: {len(experiments)}\n")

        # Group by task
        tasks = {}
        for exp in experiments:
            if exp.task not in tasks:
                tasks[exp.task] = []
            tasks[exp.task].append(exp)

        for task, task_exps in tasks.items():
            print(f"\n{task} Task ({len(task_exps)} experiments):")
            print("-" * 60)

            # Group by Nobs
            nobs_groups = {}
            for exp in task_exps:
                if exp.nobs not in nobs_groups:
                    nobs_groups[exp.nobs] = []
                nobs_groups[exp.nobs].append(exp)

            for nobs in sorted(nobs_groups.keys()):
                methods = [e.method for e in nobs_groups[nobs]]
                print(f"  Nobs={nobs:5d}: {', '.join(methods)}")

        print("\n" + "=" * 80)


def main():
    """Main entry point for experiment runner."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Run all experiments for ICML 2026 publication'
    )
    parser.add_argument('--dry-run', action='store_true',
                       help='Print commands without executing')
    parser.add_argument('--task', type=str, choices=['CS', 'SIR', 'Pendulum', 'Spectra'],
                       help='Filter by task')
    parser.add_argument('--method', type=str,
                       choices=['RVNP', 'RVNP-T', 'RVNP-G', 'NPE', 'NNPE'],
                       help='Filter by method')
    parser.add_argument('--nobs', type=int, choices=[1, 10, 100, 1000, 10000],
                       help='Filter by number of observations')
    parser.add_argument('--print-matrix', action='store_true',
                       help='Print experimental matrix and exit')
    parser.add_argument('--output-dir', type=str, default='experiments_output',
                       help='Base output directory for experiments')
    parser.add_argument('--results-dir', type=str, default='experiment_results',
                       help='Directory for collected results')

    args = parser.parse_args()

    # Create runner
    runner = ExperimentRunner(
        base_output_dir=args.output_dir,
        results_dir=args.results_dir
    )

    # Print matrix if requested
    if args.print_matrix:
        runner.print_experiment_matrix()
        return

    # Run experiments
    runner.run_all_experiments(
        dry_run=args.dry_run,
        filter_task=args.task,
        filter_method=args.method,
        filter_nobs=args.nobs
    )


if __name__ == '__main__':
    main()
