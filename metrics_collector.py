"""
Metrics Collection and Aggregation for ICML 2026 Publication

Integrates with evaluate.py to collect, aggregate, and analyze metrics
across all experiments. Prepares data for publication-quality plots.
"""

import os
import json
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and aggregates metrics from all experiments."""

    def __init__(self, results_dir: str = 'experiment_results',
                 metrics_db_file: str = 'metrics_database.csv'):
        """
        Initialize metrics collector.

        Args:
            results_dir: Directory containing experiment results
            metrics_db_file: File to store aggregated metrics database
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        self.metrics_db_path = self.results_dir / metrics_db_file
        self.db = self._load_or_create_db()

    def _load_or_create_db(self) -> pd.DataFrame:
        """Load existing metrics database or create new one."""
        if self.metrics_db_path.exists():
            df = pd.read_csv(self.metrics_db_path)
            logger.info(f"Loaded metrics database with {len(df)} entries")
            return df
        else:
            # Create empty database with expected columns
            columns = [
                'task', 'method', 'correction_type', 'nobs',
                'aepc', 'lpp_median', 'lpp_std', 'aempc', 'nrmse_mean', 'nrmse_std',
                'ess_mean', 'n_evaluations', 'timestamp'
            ]
            df = pd.DataFrame(columns=columns)
            logger.info("Created new metrics database")
            return df

    def run_evaluation_for_experiment(self, config_path: str, task: str,
                                     method: str, nobs: int,
                                     use_sir: bool = False,
                                     n_samples: int = 5000,
                                     n_eval_points: int = 100) -> Optional[Dict]:
        """
        Run evaluation for a single experiment using evaluate.py.

        Args:
            config_path: Path to config file
            task: Task name
            method: Method name
            nobs: Number of observations
            use_sir: Whether to use SIR
            n_samples: Number of samples
            n_eval_points: Number of evaluation points

        Returns:
            Dictionary with metrics or None if failed
        """
        logger.info(f"Running evaluation: {task}_{method}_nobs{nobs}")

        # Build command
        cmd = [
            'python', 'evaluate.py',
            f'--config={config_path}',
            f'--n_samples={n_samples}',
            f'--n_eval_points={n_eval_points}',
            f'--save_dir={self.results_dir}'
        ]
        if use_sir:
            cmd.append('--use_sir')

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            logger.info(f"✓ Evaluation completed for {task}_{method}_nobs{nobs}")

            # Load results
            sir_suffix = '_sir' if use_sir else ''
            results_file = self.results_dir / f'{task}_ranpt_n{nobs}{sir_suffix}_results.csv'

            if results_file.exists():
                df = pd.read_csv(results_file)
                return df.iloc[0].to_dict()
            else:
                logger.warning(f"Results file not found: {results_file}")
                return None

        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Evaluation failed for {task}_{method}_nobs{nobs}")
            logger.error(f"Error: {e.stderr}")
            return None
        except subprocess.TimeoutExpired:
            logger.error(f"✗ Evaluation timed out for {task}_{method}_nobs{nobs}")
            return None

    def add_metrics(self, task: str, method: str, correction_type: str,
                   nobs: int, metrics: Dict):
        """
        Add metrics from an experiment to the database.

        Args:
            task: Task name
            method: Method name
            correction_type: Correction type
            nobs: Number of observations
            metrics: Dictionary of metrics from evaluate.py
        """
        import time

        # Convert evaluate.py metrics to publication metrics
        # AEPC: use alpha (coverage calibration metric)
        aepc = metrics.get('alpha', np.nan)

        # LPP: median log probability
        lpp_median = metrics.get('med_log_prob', np.nan)
        lpp_std = metrics.get('std_log_prob', np.nan)

        # AEMPC: can be computed from marginal coverage (not directly in evaluate.py)
        # For now, we'll compute it separately if needed
        aempc = np.nan  # Placeholder

        # NRMSE
        nrmse_mean = metrics.get('mean_nrmse', np.nan)
        nrmse_std = metrics.get('std_nrmse', np.nan)

        # ESS
        ess_mean = metrics.get('mean_ess', 0.0)

        # Number of evaluations
        n_evaluations = metrics.get('n_evaluations', 0)

        # Create row
        row = {
            'task': task,
            'method': method,
            'correction_type': correction_type,
            'nobs': nobs,
            'aepc': aepc,
            'lpp_median': lpp_median,
            'lpp_std': lpp_std,
            'aempc': aempc,
            'nrmse_mean': nrmse_mean,
            'nrmse_std': nrmse_std,
            'ess_mean': ess_mean,
            'n_evaluations': n_evaluations,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        # Check if entry exists
        mask = (
            (self.db['task'] == task) &
            (self.db['method'] == method) &
            (self.db['nobs'] == nobs)
        )

        if mask.any():
            # Update existing entry
            for key, value in row.items():
                self.db.loc[mask, key] = value
            logger.info(f"Updated metrics for {task}_{method}_nobs{nobs}")
        else:
            # Add new entry
            self.db = pd.concat([self.db, pd.DataFrame([row])], ignore_index=True)
            logger.info(f"Added metrics for {task}_{method}_nobs{nobs}")

        # Save database
        self.save_db()

    def save_db(self):
        """Save metrics database to CSV."""
        self.db.to_csv(self.metrics_db_path, index=False)
        logger.info(f"Metrics database saved to: {self.metrics_db_path}")

    def get_metrics_for_task(self, task: str) -> pd.DataFrame:
        """
        Get all metrics for a specific task.

        Args:
            task: Task name

        Returns:
            DataFrame with metrics for this task
        """
        return self.db[self.db['task'] == task].copy()

    def get_metrics_for_method(self, method: str) -> pd.DataFrame:
        """
        Get all metrics for a specific method.

        Args:
            method: Method name

        Returns:
            DataFrame with metrics for this method
        """
        return self.db[self.db['method'] == method].copy()

    def get_metrics_matrix(self, task: str) -> pd.DataFrame:
        """
        Get metrics matrix for a task (methods × nobs).

        Args:
            task: Task name

        Returns:
            DataFrame with methods as rows, nobs as columns
        """
        task_df = self.get_metrics_for_task(task)

        if task_df.empty:
            logger.warning(f"No metrics found for task: {task}")
            return pd.DataFrame()

        # Pivot for each metric
        matrices = {}
        for metric in ['aepc', 'lpp_median', 'nrmse_mean']:
            pivot = task_df.pivot_table(
                index='method',
                columns='nobs',
                values=metric,
                aggfunc='mean'
            )
            matrices[metric] = pivot

        return matrices

    def compute_summary_statistics(self) -> pd.DataFrame:
        """
        Compute summary statistics across all experiments.

        Returns:
            DataFrame with summary statistics
        """
        if self.db.empty:
            logger.warning("No metrics in database")
            return pd.DataFrame()

        # Group by method
        summary = self.db.groupby('method').agg({
            'aepc': ['mean', 'std', 'min', 'max'],
            'lpp_median': ['mean', 'std', 'min', 'max'],
            'nrmse_mean': ['mean', 'std', 'min', 'max'],
            'n_evaluations': 'sum'
        }).round(4)

        return summary

    def export_for_plotting(self, output_file: str = 'metrics_for_plotting.json'):
        """
        Export metrics in format suitable for plotting scripts.

        Args:
            output_file: Output JSON file
        """
        if self.db.empty:
            logger.warning("No metrics to export")
            return

        # Organize by task
        data = {}
        for task in self.db['task'].unique():
            task_df = self.get_metrics_for_task(task)

            # Organize by method
            task_data = {}
            for method in task_df['method'].unique():
                method_df = task_df[task_df['method'] == method]

                # Sort by nobs
                method_df = method_df.sort_values('nobs')

                task_data[method] = {
                    'nobs': method_df['nobs'].tolist(),
                    'aepc': method_df['aepc'].tolist(),
                    'lpp_median': method_df['lpp_median'].tolist(),
                    'lpp_std': method_df['lpp_std'].tolist(),
                    'nrmse_mean': method_df['nrmse_mean'].tolist(),
                    'nrmse_std': method_df['nrmse_std'].tolist(),
                }

            data[task] = task_data

        # Save to JSON
        output_path = self.results_dir / output_file
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported metrics for plotting to: {output_path}")

    def print_summary(self):
        """Print summary of collected metrics."""
        print("\n" + "=" * 80)
        print("METRICS DATABASE SUMMARY")
        print("=" * 80)

        if self.db.empty:
            print("No metrics collected yet.")
            return

        print(f"Total experiments: {len(self.db)}")
        print(f"\nTasks: {', '.join(self.db['task'].unique())}")
        print(f"Methods: {', '.join(self.db['method'].unique())}")
        print(f"Nobs values: {sorted(self.db['nobs'].unique())}")

        print("\n" + "-" * 80)
        print("Coverage by task:")
        for task in sorted(self.db['task'].unique()):
            task_df = self.db[self.db['task'] == task]
            n_experiments = len(task_df)
            methods = task_df['method'].nunique()
            nobs_vals = task_df['nobs'].nunique()
            print(f"  {task:15s}: {n_experiments:3d} experiments "
                  f"({methods} methods × {nobs_vals} Nobs values)")

        print("\n" + "-" * 80)
        print("Summary statistics by method:")
        print()
        summary = self.compute_summary_statistics()
        print(summary)

        print("=" * 80)


def main():
    """Main entry point for metrics collector."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Collect and aggregate metrics from experiments'
    )
    parser.add_argument('--results-dir', type=str, default='experiment_results',
                       help='Directory containing experiment results')
    parser.add_argument('--summary', action='store_true',
                       help='Print summary of collected metrics')
    parser.add_argument('--export', action='store_true',
                       help='Export metrics for plotting')
    parser.add_argument('--task', type=str,
                       help='Get metrics for specific task')

    args = parser.parse_args()

    # Create collector
    collector = MetricsCollector(results_dir=args.results_dir)

    if args.summary:
        collector.print_summary()

    if args.export:
        collector.export_for_plotting()

    if args.task:
        matrices = collector.get_metrics_matrix(args.task)
        print(f"\n=== Metrics for {args.task} ===\n")
        for metric_name, matrix in matrices.items():
            print(f"\n{metric_name.upper()}:")
            print(matrix)


if __name__ == '__main__':
    main()
