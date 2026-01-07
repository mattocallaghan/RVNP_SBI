"""
Publication-Quality Plotting for ICML 2026

Creates 4-panel plots showing metrics vs. Nobs for all methods,
matching the style of Figures 3-6 in the paper.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import json


class PublicationPlotter:
    """Creates publication-quality plots for ICML paper."""

    # Method display names and colors
    METHOD_STYLES = {
        'RVNP-simple': {'label': 'RVNP (diagonal cov)', 'color': '#2E86AB', 'marker': 'o', 'linestyle': '-'},
        'RVNP-mu_hybrid': {'label': 'RVNP (neural mean+cov)', 'color': '#A23B72', 'marker': 's', 'linestyle': '-'},
        'NPE': {'label': 'NPE', 'color': '#6A994E', 'marker': 'D', 'linestyle': '--'},
    }

    # Task display names
    TASK_NAMES = {
        'CS': 'Cancer/Stromal (CS)',
        'SIR': 'SIR Epidemiological Model',
        'Pendulum': 'Pendulum Dynamics',
        'Spectra': 'Stellar Spectra (Gaia)'
    }

    def __init__(self, metrics_db_path: str = 'experiment_results/metrics_database.csv'):
        """
        Initialize plotter with metrics database.

        Args:
            metrics_db_path: Path to metrics CSV file
        """
        self.metrics_db_path = Path(metrics_db_path)
        if not self.metrics_db_path.exists():
            raise FileNotFoundError(f"Metrics database not found: {metrics_db_path}")

        self.df = pd.read_csv(self.metrics_db_path)
        print(f"Loaded metrics database with {len(self.df)} entries")

        # Setup plotting style
        self._setup_plot_style()

    def _setup_plot_style(self):
        """Setup publication-quality plot style."""
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_context("paper", font_scale=1.2, rc={"lines.linewidth": 2.5})

        # Custom rcParams for publication quality
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'mathtext.fontset': 'cm',
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 9,
            'figure.titlesize': 14,
            'axes.linewidth': 1.2,
            'grid.linewidth': 0.8,
            'lines.linewidth': 2.5,
            'lines.markersize': 8,
            'legend.framealpha': 0.95,
            'legend.edgecolor': '0.8',
        })

    def plot_task_comparison(self, task: str, save_path: Optional[str] = None,
                            use_sir: bool = False):
        """
        Create 4-panel plot for a single task showing all metrics vs. Nobs.

        Args:
            task: Task name (CS, SIR, Pendulum, Spectra)
            save_path: Path to save figure (optional)
            use_sir: Whether to plot SIR results (default: False for non-SIR)
        """
        # Filter data for this task and SIR setting
        task_df = self.df[
            (self.df['task'] == task) &
            (self.df['use_sir'] == use_sir)
        ].copy()

        if task_df.empty:
            print(f"No data found for task: {task}")
            return None

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.patch.set_facecolor('white')

        # Plot each metric
        self._plot_aepc(axes[0, 0], task_df)
        self._plot_lpp(axes[0, 1], task_df)
        self._plot_acauc(axes[1, 0], task_df)
        self._plot_nrmse(axes[1, 1], task_df)

        # Overall title
        task_display = self.TASK_NAMES.get(task, task)
        sir_suffix = ' (with SIR)' if use_sir else ''
        fig.suptitle(f'{task_display}{sir_suffix}', fontsize=16, fontweight='bold', y=0.995)

        # Shared legend
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, -0.02),
                  ncol=5, frameon=True, fancybox=True, shadow=False)

        plt.tight_layout(rect=[0, 0.02, 1, 0.98])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Plot saved to: {save_path}")

        plt.show()
        return fig

    def _plot_aepc(self, ax, task_df):
        """Plot AEPC (Average Expected Posterior Coverage) vs. Nobs."""
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='Ideal')

        for method in sorted(task_df['method'].unique()):
            if method not in self.METHOD_STYLES:
                continue

            method_df = task_df[task_df['method'] == method].sort_values('nobs')
            style = self.METHOD_STYLES[method]

            ax.plot(method_df['nobs'], method_df['aepc'],
                   label=style['label'],
                   color=style['color'],
                   marker=style['marker'],
                   linestyle=style['linestyle'],
                   markersize=8,
                   linewidth=2.5,
                   alpha=0.85)

        ax.set_xlabel(r'$N_{\mathrm{obs}}$ (Number of Observations)', fontweight='bold')
        ax.set_ylabel(r'$\alpha$ (AEPC)', fontweight='bold')
        ax.set_title('Coverage Calibration', fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.axhspan(-0.1, 0.1, alpha=0.1, color='green')  # Ideal range

    def _plot_lpp(self, ax, task_df):
        """Plot LPP (Log Posterior Probability) vs. Nobs."""
        for method in sorted(task_df['method'].unique()):
            if method not in self.METHOD_STYLES:
                continue

            method_df = task_df[task_df['method'] == method].sort_values('nobs')
            style = self.METHOD_STYLES[method]

            # Plot with error bars
            ax.errorbar(method_df['nobs'], method_df['lpp_median'],
                       yerr=method_df['lpp_std'],
                       label=style['label'],
                       color=style['color'],
                       marker=style['marker'],
                       linestyle=style['linestyle'],
                       markersize=8,
                       linewidth=2.5,
                       capsize=4,
                       alpha=0.85)

        ax.set_xlabel(r'$N_{\mathrm{obs}}$ (Number of Observations)', fontweight='bold')
        ax.set_ylabel(r'$\log q(\theta^* \mid x_{\mathrm{obs}})$ (LPP)', fontweight='bold')
        ax.set_title('Log Posterior Probability', fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

    def _plot_acauc(self, ax, task_df):
        """Plot ACAUC (Average Coverage Area Under Curve) vs. Nobs."""
        # Reference line at 1.0 (perfect calibration)
        ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='Ideal')

        for method in sorted(task_df['method'].unique()):
            if method not in self.METHOD_STYLES:
                continue

            method_df = task_df[task_df['method'] == method].sort_values('nobs')
            style = self.METHOD_STYLES[method]

            # Plot with error bars
            ax.errorbar(method_df['nobs'], method_df['acauc_mean'],
                       yerr=method_df['acauc_std'],
                       label=style['label'],
                       color=style['color'],
                       marker=style['marker'],
                       linestyle=style['linestyle'],
                       markersize=8,
                       linewidth=2.5,
                       capsize=4,
                       alpha=0.85)

        ax.set_xlabel(r'$N_{\mathrm{obs}}$ (Number of Observations)', fontweight='bold')
        ax.set_ylabel('ACAUC', fontweight='bold')
        ax.set_title('Coverage Area Under Curve', fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        # Highlight ideal range around 1.0
        ax.axhspan(0.9, 1.1, alpha=0.1, color='green')

    def _plot_nrmse(self, ax, task_df):
        """Plot NRMSE (Normalized Root Mean Squared Error) vs. Nobs."""
        for method in sorted(task_df['method'].unique()):
            if method not in self.METHOD_STYLES:
                continue

            method_df = task_df[task_df['method'] == method].sort_values('nobs')
            style = self.METHOD_STYLES[method]

            # Plot with error bars
            ax.errorbar(method_df['nobs'], method_df['nrmse_mean'],
                       yerr=method_df['nrmse_std'],
                       label=style['label'],
                       color=style['color'],
                       marker=style['marker'],
                       linestyle=style['linestyle'],
                       markersize=8,
                       linewidth=2.5,
                       capsize=4,
                       alpha=0.85)

        ax.set_xlabel(r'$N_{\mathrm{obs}}$ (Number of Observations)', fontweight='bold')
        ax.set_ylabel('NRMSE', fontweight='bold')
        ax.set_title('Parameter Estimation Accuracy', fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

    def plot_all_tasks(self, save_dir: Optional[str] = None, include_sir: bool = True):
        """
        Create comparison plots for all tasks in the database.

        Args:
            save_dir: Directory to save plots (optional)
            include_sir: Whether to also create SIR plots
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True, parents=True)

        tasks = self.df['task'].unique()

        for task in tasks:
            # Regular plots
            print(f"\nGenerating plot for {task}...")
            save_path = None
            if save_dir:
                save_path = save_dir / f'{task}_comparison.png'
            self.plot_task_comparison(task, save_path=save_path, use_sir=False)

            # SIR plots if requested
            if include_sir and self.df['use_sir'].any():
                print(f"\nGenerating SIR plot for {task}...")
                save_path = None
                if save_dir:
                    save_path = save_dir / f'{task}_comparison_sir.png'
                self.plot_task_comparison(task, save_path=save_path, use_sir=True)

    def plot_method_comparison_across_tasks(self, method: str,
                                           save_path: Optional[str] = None):
        """
        Plot a single method's performance across all tasks.

        Args:
            method: Method name (RVNP, NPE, etc.)
            save_path: Path to save figure
        """
        method_df = self.df[self.df['method'] == method].copy()

        if method_df.empty:
            print(f"No data found for method: {method}")
            return None

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.patch.set_facecolor('white')

        # Plot each metric across tasks
        tasks = sorted(method_df['task'].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, len(tasks)))

        for ax_idx, metric in enumerate([('aepc', 'AEPC'),
                                         ('lpp_median', 'LPP'),
                                         ('acauc_mean', 'ACAUC'),
                                         ('nrmse_mean', 'NRMSE')]):
            ax = axes[ax_idx // 2, ax_idx % 2]
            metric_col, metric_name = metric

            for task, color in zip(tasks, colors):
                task_df = method_df[method_df['task'] == task].sort_values('nobs')
                task_display = self.TASK_NAMES.get(task, task)

                ax.plot(task_df['nobs'], task_df[metric_col],
                       label=task_display,
                       color=color,
                       marker='o',
                       markersize=8,
                       linewidth=2.5,
                       alpha=0.85)

            ax.set_xlabel(r'$N_{\mathrm{obs}}$', fontweight='bold')
            ax.set_ylabel(metric_name, fontweight='bold')
            ax.set_title(f'{metric_name} Across Tasks', fontweight='bold')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)

            if ax_idx == 0:  # Add legend to first subplot
                ax.legend(loc='best', frameon=True)

        fig.suptitle(f'{method} Performance Across Tasks',
                    fontsize=16, fontweight='bold', y=0.995)

        plt.tight_layout(rect=[0, 0, 1, 0.98])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Plot saved to: {save_path}")

        plt.show()
        return fig

    def create_summary_table(self, save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Create a summary table showing best methods for each task.

        Args:
            save_path: Path to save table as CSV

        Returns:
            DataFrame with summary
        """
        summary_rows = []

        for task in sorted(self.df['task'].unique()):
            task_df = self.df[self.df['task'] == task]

            # Find best method for each metric
            best_aepc = task_df.loc[task_df['aepc'].abs().idxmin()]
            best_lpp = task_df.loc[task_df['lpp_median'].idxmax()]
            best_nrmse = task_df.loc[task_df['nrmse_mean'].idxmin()]

            summary_rows.append({
                'Task': self.TASK_NAMES.get(task, task),
                'Best AEPC': f"{best_aepc['method']} ({best_aepc['aepc']:.3f})",
                'Best LPP': f"{best_lpp['method']} ({best_lpp['lpp_median']:.2f})",
                'Best NRMSE': f"{best_nrmse['method']} ({best_nrmse['nrmse_mean']:.3f})",
            })

        summary_df = pd.DataFrame(summary_rows)

        if save_path:
            summary_df.to_csv(save_path, index=False)
            print(f"Summary table saved to: {save_path}")

        return summary_df


def main():
    """Main entry point for publication plotting."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Create publication-quality plots from metrics database'
    )
    parser.add_argument('--metrics-db', type=str,
                       default='experiment_results/metrics_database.csv',
                       help='Path to metrics database CSV')
    parser.add_argument('--task', type=str,
                       choices=['CS', 'SIR', 'Pendulum', 'Spectra', 'all'],
                       help='Task to plot (or "all" for all tasks)')
    parser.add_argument('--method', type=str,
                       choices=['RVNP-simple', 'RVNP-mu_hybrid', 'NPE'],
                       help='Plot method comparison across tasks')
    parser.add_argument('--save-dir', type=str, default='publication_plots',
                       help='Directory to save plots')
    parser.add_argument('--summary', action='store_true',
                       help='Create summary table')

    args = parser.parse_args()

    # Create plotter
    try:
        plotter = PublicationPlotter(metrics_db_path=args.metrics_db)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run metrics collection first.")
        return

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Generate plots
    if args.task:
        if args.task == 'all':
            plotter.plot_all_tasks(save_dir=save_dir, include_sir=True)
        else:
            save_path = save_dir / f'{args.task}_comparison.png'
            plotter.plot_task_comparison(args.task, save_path=save_path, use_sir=False)
            # Also create SIR version if data exists
            if plotter.df['use_sir'].any():
                save_path_sir = save_dir / f'{args.task}_comparison_sir.png'
                plotter.plot_task_comparison(args.task, save_path=save_path_sir, use_sir=True)

    if args.method:
        save_path = save_dir / f'{args.method}_across_tasks.png'
        plotter.plot_method_comparison_across_tasks(args.method, save_path=save_path)

    if args.summary:
        save_path = save_dir / 'summary_table.csv'
        summary_df = plotter.create_summary_table(save_path=save_path)
        print("\n" + "=" * 80)
        print("SUMMARY TABLE")
        print("=" * 80)
        print(summary_df.to_string(index=False))
        print("=" * 80)

    if not (args.task or args.method or args.summary):
        print("No action specified. Use --task, --method, or --summary")
        print("Example: python publication_plots.py --task=all")


if __name__ == '__main__':
    main()
