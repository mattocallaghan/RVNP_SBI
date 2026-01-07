"""
Integrated Pipeline for ICML 2026 Publication
Handles training, evaluation, metrics collection, and plotting in one unified system.
Designed to be resumable for Google Colab with time limits.
"""

import os
import sys
import json
import subprocess
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    task: str
    method: str
    correction_type: str  # 'simple', 'NN', or 'none'
    nobs: int
    config_file: str

    # Tracking
    training_complete: bool = False
    evaluation_complete: bool = False
    evaluation_sir_complete: bool = False

    def get_id(self) -> str:
        """Get unique experiment identifier."""
        return f"{self.task}_{self.method}_{self.correction_type}_nobs{self.nobs}"

    def to_dict(self):
        return asdict(self)


class IntegratedPipeline:
    """Manages the complete experimental pipeline with resumability."""

    # Experimental matrix
    TASKS = ['CS', 'SIR', 'Pendulum', 'Spectra']
    NOBS_VALUES = [1, 10, 100, 1000, 10000]

    # Methods to compare (user wants to focus on these)
    METHODS = [
        ('RVNP-simple', 'ranpt', 'simple'),
        ('RVNP-NN', 'ranpt', 'NN'),
        ('NPE', 'npe', 'none'),
    ]

    def __init__(self,
                 results_dir: str = 'experiment_results',
                 state_file: str = 'pipeline_state.json',
                 wellspec: bool = False):
        """
        Initialize pipeline.

        Args:
            results_dir: Directory for all results
            state_file: File to track pipeline state for resumability
            wellspec: If True, use well-specified configs
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.wellspec = wellspec

        self.state_file = self.results_dir / state_file
        self.state = self._load_state()

        # Metrics database
        self.metrics_db_file = self.results_dir / 'metrics_database.csv'
        self.metrics_db = self._load_metrics_db()

    def _load_state(self) -> Dict:
        """Load pipeline state from file."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            logger.info(f"Loaded pipeline state with {len(state.get('experiments', []))} experiments")
            return state
        return {'experiments': [], 'last_updated': None}

    def _save_state(self):
        """Save pipeline state to file."""
        self.state['last_updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def _load_metrics_db(self) -> pd.DataFrame:
        """Load or create metrics database."""
        if self.metrics_db_file.exists():
            df = pd.read_csv(self.metrics_db_file)
            logger.info(f"Loaded metrics database with {len(df)} entries")
            return df

        # Create empty database
        columns = [
            'task', 'method', 'correction_type', 'nobs',
            'aepc', 'acauc_mean', 'acauc_std', 'lpp_median', 'lpp_std',
            'nrmse_mean', 'nrmse_std', 'ess_mean', 'n_evaluations', 'use_sir', 'timestamp'
        ]
        return pd.DataFrame(columns=columns)

    def _save_metrics_db(self):
        """Save metrics database."""
        self.metrics_db.to_csv(self.metrics_db_file, index=False)
        logger.info(f"Saved metrics database to {self.metrics_db_file}")

    def generate_experiment_list(self) -> List[ExperimentConfig]:
        """Generate list of all experiments to run."""
        experiments = []

        for task in self.TASKS:
            # Determine valid Nobs for this task
            if task == 'Spectra':
                valid_nobs = [1, 10, 100]  # Limited by real data
            else:
                valid_nobs = self.NOBS_VALUES

            for nobs in valid_nobs:
                for method_name, model_name, correction_type in self.METHODS:
                    # Find config file
                    config_file = self._find_config_file(task, model_name, correction_type, nobs)

                    if config_file:
                        exp = ExperimentConfig(
                            task=task,
                            method=method_name,
                            correction_type=correction_type,
                            nobs=nobs,
                            config_file=config_file
                        )
                        experiments.append(exp)
                    else:
                        logger.warning(f"Config not found for {task} {method_name} nobs={nobs}")

        logger.info(f"Generated {len(experiments)} experiments")
        return experiments

    def _find_config_file(self, task: str, model_name: str,
                          correction_type: str, nobs: int) -> Optional[str]:
        """Find config file for an experiment."""
        # Map task names to config directories
        task_dir_map = {
            'CS': 'CS_task',
            'SIR': 'SIR',
            'Pendulum': 'Pendulum',
            'Spectra': 'Spectra'
        }

        task_dir = task_dir_map.get(task, task)

        # Build config filename
        if self.wellspec:
            # Well-specified configs use different naming pattern
            # Pattern: {task}_task_tests{nobs}_{correction_type}_shrink00_wellspec.py
            task_lower = task.lower()
            if model_name == 'npe':
                config_name = f"npe_{task_lower}_task_tests{nobs}_wellspec.py"
            elif model_name == 'ranpt':
                # For RANPT, use 'hybrid' in filename even for mu_hybrid
                # (wellspec configs were created before mu_hybrid naming)
                correction_for_file = 'hybrid' if correction_type == 'NN' else correction_type
                config_name = f"{task_lower}_task_tests{nobs}_{correction_for_file}_shrink00_wellspec.py"
            else:
                return None
        else:
            # Standard configs
            if model_name == 'npe':
                # NPE configs
                config_name = f"npe_{task.lower()}_task.py"
            elif model_name == 'ranpt':
                # RVNP configs
                config_name = f"ranpt_{nobs}_{correction_type}.py"
            else:
                return None

        config_path = f"configs/{task_dir}/{config_name}"

        # Check if file exists
        if os.path.exists(config_path):
            return config_path

        return None

    def _get_or_create_experiment_state(self, exp: ExperimentConfig) -> Dict:
        """Get or create state entry for an experiment."""
        exp_id = exp.get_id()

        # Find existing state
        for exp_state in self.state['experiments']:
            if exp_state['id'] == exp_id:
                return exp_state

        # Create new state
        exp_state = {
            'id': exp_id,
            'task': exp.task,
            'method': exp.method,
            'correction_type': exp.correction_type,
            'nobs': exp.nobs,
            'config_file': exp.config_file,
            'training_complete': False,
            'evaluation_complete': False,
            'evaluation_sir_complete': False,
            'training_time': 0.0,
            'evaluation_time': 0.0,
            'errors': []
        }
        self.state['experiments'].append(exp_state)
        return exp_state

    def run_training(self, exp: ExperimentConfig) -> bool:
        """
        Run training for an experiment.

        Args:
            exp: Experiment configuration

        Returns:
            True if successful, False otherwise
        """
        exp_id = exp.get_id()
        exp_state = self._get_or_create_experiment_state(exp)

        # Check if already trained
        if exp_state['training_complete']:
            logger.info(f"✓ {exp_id} - Training already complete")
            return True

        logger.info(f"Training {exp_id}...")
        logger.info(f"  Config: {exp.config_file}")

        # Run training
        cmd = [
            'python', 'scripts/main_train_eval.py',
            f'--config={exp.config_file}',
            '--mode=train'
        ]

        try:
            start_time = time.time()
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout per experiment
            )
            elapsed = time.time() - start_time

            logger.info(f"✓ {exp_id} - Training completed in {elapsed:.1f}s")

            # Update state
            exp_state['training_complete'] = True
            exp_state['training_time'] = elapsed
            self._save_state()

            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"✗ {exp_id} - Training failed")
            logger.error(f"  Error: {str(e)[:200]}")
            exp_state['errors'].append({
                'stage': 'training',
                'error': str(e)[:500],
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            })
            self._save_state()
            return False

        except subprocess.TimeoutExpired:
            logger.error(f"✗ {exp_id} - Training timed out")
            exp_state['errors'].append({
                'stage': 'training',
                'error': 'Timeout after 2 hours',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            })
            self._save_state()
            return False

    def run_evaluation(self, exp: ExperimentConfig, use_sir: bool = False) -> bool:
        """
        Run evaluation for an experiment.

        Args:
            exp: Experiment configuration
            use_sir: Whether to use Sampling Importance Resampling

        Returns:
            True if successful, False otherwise
        """
        exp_id = exp.get_id()
        exp_state = self._get_or_create_experiment_state(exp)

        # Check if training is complete
        if not exp_state['training_complete']:
            logger.warning(f"⚠ {exp_id} - Cannot evaluate, training not complete")
            return False

        # Check if already evaluated
        eval_key = 'evaluation_sir_complete' if use_sir else 'evaluation_complete'
        if exp_state[eval_key]:
            logger.info(f"✓ {exp_id} - Evaluation{'  +SIR' if use_sir else ''} already complete")
            return True

        sir_suffix = ' +SIR' if use_sir else ''
        logger.info(f"Evaluating {exp_id}{sir_suffix}...")

        # Determine number of evaluation points based on nobs
        if exp.nobs >= 100:
            n_eval_points = 100
        elif exp.nobs >= 10:
            n_eval_points = 50
        else:
            n_eval_points = min(50, 100)  # For nobs=1, evaluate multiple experiments

        # Build command
        cmd = [
            'python', 'scripts/evaluate.py',
            f'--config={exp.config_file}',
            '--n_samples=10000',  # Increased to 10000 for publication quality
            f'--n_eval_points={n_eval_points}',
            f'--save_dir={self.results_dir}'
        ]

        if use_sir:
            cmd.append('--use_sir')

        try:
            start_time = time.time()
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            elapsed = time.time() - start_time

            logger.info(f"✓ {exp_id} - Evaluation{sir_suffix} completed in {elapsed:.1f}s")

            # Load results and add to metrics database
            self._add_evaluation_to_metrics(exp, use_sir)

            # Update state
            exp_state[eval_key] = True
            exp_state['evaluation_time'] = exp_state.get('evaluation_time', 0) + elapsed
            self._save_state()

            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"✗ {exp_id} - Evaluation{sir_suffix} failed")
            logger.error(f"  Error: {str(e)[:200]}")
            exp_state['errors'].append({
                'stage': f'evaluation{"_sir" if use_sir else ""}',
                'error': str(e)[:500],
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            })
            self._save_state()
            return False

        except subprocess.TimeoutExpired:
            logger.error(f"✗ {exp_id} - Evaluation{sir_suffix} timed out")
            exp_state['errors'].append({
                'stage': f'evaluation{"_sir" if use_sir else ""}',
                'error': 'Timeout after 1 hour',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            })
            self._save_state()
            return False

    def _add_evaluation_to_metrics(self, exp: ExperimentConfig, use_sir: bool):
        """Add evaluation results to metrics database."""
        # Find results file
        sir_suffix = '_sir' if use_sir else ''

        # Try different naming patterns
        possible_names = [
            f'{exp.task}_{exp.method.split("-")[0].lower()}_n{exp.nobs}{sir_suffix}_results.csv',
            f'{exp.task}_ranpt_n{exp.nobs}{sir_suffix}_results.csv',
            f'{exp.task}_npe_n{exp.nobs}{sir_suffix}_results.csv',
        ]

        results_df = None
        for fname in possible_names:
            fpath = self.results_dir / fname
            if fpath.exists():
                results_df = pd.read_csv(fpath)
                break

        if results_df is None or len(results_df) == 0:
            logger.warning(f"Could not find results file for {exp.get_id()}")
            return

        # Extract metrics from results
        result_row = results_df.iloc[0]

        # Create metrics row
        metrics_row = {
            'task': exp.task,
            'method': exp.method,
            'correction_type': exp.correction_type,
            'nobs': exp.nobs,
            'aepc': result_row.get('alpha', np.nan),
            'acauc_mean': result_row.get('acauc_mean', np.nan),
            'acauc_std': result_row.get('acauc_std', np.nan),
            'lpp_median': result_row.get('med_log_prob', np.nan),
            'lpp_std': result_row.get('std_log_prob', np.nan),
            'nrmse_mean': result_row.get('mean_nrmse', np.nan),
            'nrmse_std': result_row.get('std_nrmse', np.nan),
            'ess_mean': result_row.get('mean_ess', 0.0),
            'n_evaluations': result_row.get('n_evaluations', 0),
            'use_sir': use_sir,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        # Check if entry exists
        mask = (
            (self.metrics_db['task'] == exp.task) &
            (self.metrics_db['method'] == exp.method) &
            (self.metrics_db['nobs'] == exp.nobs) &
            (self.metrics_db['use_sir'] == use_sir)
        )

        if mask.any():
            # Update existing
            for key, value in metrics_row.items():
                self.metrics_db.loc[mask, key] = value
        else:
            # Add new
            self.metrics_db = pd.concat(
                [self.metrics_db, pd.DataFrame([metrics_row])],
                ignore_index=True
            )

        self._save_metrics_db()
        logger.info(f"Added metrics for {exp.get_id()}{' +SIR' if use_sir else ''}")

    def run_all(self,
                filter_task: Optional[str] = None,
                filter_method: Optional[str] = None,
                filter_nobs: Optional[int] = None,
                use_sir: bool = True,
                skip_training: bool = False,
                skip_evaluation: bool = False):
        """
        Run complete pipeline for all experiments.

        Args:
            filter_task: Optional task filter
            filter_method: Optional method filter
            filter_nobs: Optional nobs filter
            use_sir: Whether to run SIR evaluation
            skip_training: Skip training stage
            skip_evaluation: Skip evaluation stage
        """
        logger.info("=" * 80)
        logger.info("INTEGRATED PIPELINE - ICML 2026")
        logger.info("=" * 80)

        # Generate experiment list
        experiments = self.generate_experiment_list()

        # Apply filters
        if filter_task:
            experiments = [e for e in experiments if e.task == filter_task]
        if filter_method:
            experiments = [e for e in experiments if e.method == filter_method]
        if filter_nobs:
            experiments = [e for e in experiments if e.nobs == filter_nobs]

        logger.info(f"Running {len(experiments)} experiments")
        logger.info(f"  Training: {'skipped' if skip_training else 'enabled'}")
        logger.info(f"  Evaluation: {'skipped' if skip_evaluation else 'enabled'}")
        logger.info(f"  SIR: {'enabled' if use_sir else 'disabled'}")
        logger.info("")

        # Track statistics
        stats = {
            'training_success': 0,
            'training_failed': 0,
            'training_skipped': 0,
            'evaluation_success': 0,
            'evaluation_failed': 0,
            'evaluation_skipped': 0,
            'sir_success': 0,
            'sir_failed': 0,
            'sir_skipped': 0
        }

        for i, exp in enumerate(experiments, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"Experiment {i}/{len(experiments)}: {exp.get_id()}")
            logger.info(f"{'='*80}")

            # Training
            if not skip_training:
                if self.run_training(exp):
                    stats['training_success'] += 1
                else:
                    stats['training_failed'] += 1
                    continue  # Skip evaluation if training failed
            else:
                stats['training_skipped'] += 1

            # Evaluation
            if not skip_evaluation:
                # Regular evaluation
                if self.run_evaluation(exp, use_sir=False):
                    stats['evaluation_success'] += 1
                else:
                    stats['evaluation_failed'] += 1

                # SIR evaluation
                if use_sir:
                    if self.run_evaluation(exp, use_sir=True):
                        stats['sir_success'] += 1
                    else:
                        stats['sir_failed'] += 1
            else:
                stats['evaluation_skipped'] += 1
                stats['sir_skipped'] += 1

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total experiments: {len(experiments)}")
        logger.info(f"\nTraining:")
        logger.info(f"  Success: {stats['training_success']}")
        logger.info(f"  Failed:  {stats['training_failed']}")
        logger.info(f"  Skipped: {stats['training_skipped']}")
        logger.info(f"\nEvaluation:")
        logger.info(f"  Success: {stats['evaluation_success']}")
        logger.info(f"  Failed:  {stats['evaluation_failed']}")
        logger.info(f"  Skipped: {stats['evaluation_skipped']}")
        if use_sir:
            logger.info(f"\nSIR Evaluation:")
            logger.info(f"  Success: {stats['sir_success']}")
            logger.info(f"  Failed:  {stats['sir_failed']}")
            logger.info(f"  Skipped: {stats['sir_skipped']}")
        logger.info(f"\nMetrics database: {self.metrics_db_file}")
        logger.info(f"Pipeline state: {self.state_file}")
        logger.info("=" * 80)

    def print_status(self):
        """Print current pipeline status."""
        print("\n" + "=" * 80)
        print("PIPELINE STATUS")
        print("=" * 80)

        if not self.state['experiments']:
            print("No experiments in pipeline yet.")
            return

        total = len(self.state['experiments'])
        training_complete = sum(1 for e in self.state['experiments'] if e['training_complete'])
        eval_complete = sum(1 for e in self.state['experiments'] if e['evaluation_complete'])
        eval_sir_complete = sum(1 for e in self.state['experiments'] if e['evaluation_sir_complete'])

        print(f"Total experiments: {total}")
        print(f"Training complete: {training_complete}/{total} ({training_complete/total*100:.1f}%)")
        print(f"Evaluation complete: {eval_complete}/{total} ({eval_complete/total*100:.1f}%)")
        print(f"SIR evaluation complete: {eval_sir_complete}/{total} ({eval_sir_complete/total*100:.1f}%)")

        print(f"\nMetrics collected: {len(self.metrics_db)} entries")

        # Breakdown by task
        print("\nProgress by task:")
        for task in self.TASKS:
            task_exps = [e for e in self.state['experiments'] if e['task'] == task]
            if task_exps:
                n = len(task_exps)
                trained = sum(1 for e in task_exps if e['training_complete'])
                evaluated = sum(1 for e in task_exps if e['evaluation_complete'])
                print(f"  {task:10s}: {n} exps, {trained} trained, {evaluated} evaluated")

        # Failed experiments
        failed_exps = [e for e in self.state['experiments'] if e['errors']]
        if failed_exps:
            print(f"\n{len(failed_exps)} experiments with errors:")
            for exp in failed_exps[:10]:  # Show first 10
                print(f"  - {exp['id']}: {len(exp['errors'])} errors")

        print("=" * 80)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Integrated pipeline for ICML 2026 experiments'
    )
    parser.add_argument('--task', type=str, choices=['CS', 'SIR', 'Pendulum', 'Spectra'],
                       help='Filter by task')
    parser.add_argument('--method', type=str,
                       choices=['RVNP-simple', 'RVNP-NN', 'NPE'],
                       help='Filter by method')
    parser.add_argument('--nobs', type=int, choices=[1, 10, 100, 1000, 10000],
                       help='Filter by number of observations')
    parser.add_argument('--no-sir', action='store_true',
                       help='Disable SIR evaluation')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training, only run evaluation')
    parser.add_argument('--skip-evaluation', action='store_true',
                       help='Skip evaluation, only run training')
    parser.add_argument('--status', action='store_true',
                       help='Print pipeline status and exit')
    parser.add_argument('--results-dir', type=str, default='experiment_results',
                       help='Results directory')
    parser.add_argument('--wellspec', action='store_true',
                       help='Use well-specified configs (no model misspecification)')

    args = parser.parse_args()

    # Create pipeline
    pipeline = IntegratedPipeline(
        results_dir=args.results_dir,
        wellspec=args.wellspec
    )

    # Print status if requested
    if args.status:
        pipeline.print_status()
        return

    # Run pipeline
    pipeline.run_all(
        filter_task=args.task,
        filter_method=args.method,
        filter_nobs=args.nobs,
        use_sir=not args.no_sir,
        skip_training=args.skip_training,
        skip_evaluation=args.skip_evaluation
    )


if __name__ == '__main__':
    main()
