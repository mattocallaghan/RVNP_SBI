"""
Ablation Study Plotting for ICML 2026

Generates 2x2 plots showing:
1. ACAUC vs parameter size (nn_block_dim)
2. NRMSE vs parameter size (nn_block_dim)
3. ACAUC vs shrinkage strength (lambda_shrinkage)
4. NRMSE vs shrinkage strength (lambda_shrinkage)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10


def load_ablation_metrics(metrics_db_path: str) -> pd.DataFrame:
    """Load ablation metrics from CSV."""
    df = pd.read_csv(metrics_db_path)
    print(f"Loaded {len(df)} ablation experiments")
    return df


def extract_ablation_params(df: pd.DataFrame) -> pd.DataFrame:
    """Extract ablation parameters from config names."""
    # Extract parameter size
    param_size_map = {
        'tiny': 16,
        'small': 32,
        'medium': 52,
        'large': 128,
        'huge': 256
    }

    # Extract shrinkage strength
    shrinkage_map = {
        '0': 0.0,
        'weak': 0.01,
        'medium': 0.1,
        'strong': 1.0,
        'verystrong': 10.0
    }

    df = df.copy()

    # Determine experiment type and extract values
    df['experiment_type'] = None
    df['param_size'] = None
    df['shrinkage_value'] = None

    for idx, row in df.iterrows():
        config = row['config']

        if 'params' in config:
            # Parameter size ablation
            df.at[idx, 'experiment_type'] = 'params'
            for key, value in param_size_map.items():
                if key in config:
                    df.at[idx, 'param_size'] = value
                    break

        elif 'shrink' in config:
            # Shrinkage ablation
            df.at[idx, 'experiment_type'] = 'shrinkage'
            for key, value in shrinkage_map.items():
                if f'shrink_{key}' in config:
                    df.at[idx, 'shrinkage_value'] = value
                    break

    return df


def plot_ablation_study(df: pd.DataFrame, save_dir: str):
    """Generate 2x2 ablation study plot."""
    df = extract_ablation_params(df)

    # Separate parameter and shrinkage experiments
    params_df = df[df['experiment_type'] == 'params'].copy()
    shrinkage_df = df[df['experiment_type'] == 'shrinkage'].copy()

    print(f"\nParameter size experiments: {len(params_df)}")
    print(f"Shrinkage prior experiments: {len(shrinkage_df)}")

    # Create 2x2 plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Ablation Study: CS Task (Nobs=100)', fontsize=14, fontweight='bold')

    # Panel 1: ACAUC vs parameter size
    ax = axes[0, 0]
    if not params_df.empty:
        params_sorted = params_df.sort_values('param_size')
        ax.errorbar(
            params_sorted['param_size'],
            params_sorted['acauc_mean'],
            yerr=params_sorted['acauc_std'],
            marker='o',
            markersize=8,
            linewidth=2,
            capsize=5,
            color='#2E86AB',
            label='ACAUC'
        )
        ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='Perfect')
        ax.set_xlabel('Correction Model Parameters (nn_block_dim)', fontweight='bold')
        ax.set_ylabel('ACAUC', fontweight='bold')
        ax.set_title('Coverage vs Model Size', fontweight='bold')
        ax.set_xscale('log', base=2)
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')

    # Panel 2: NRMSE vs parameter size
    ax = axes[0, 1]
    if not params_df.empty:
        params_sorted = params_df.sort_values('param_size')
        ax.errorbar(
            params_sorted['param_size'],
            params_sorted['nrmse_mean'],
            yerr=params_sorted['nrmse_std'],
            marker='s',
            markersize=8,
            linewidth=2,
            capsize=5,
            color='#A23B72',
            label='NRMSE'
        )
        ax.set_xlabel('Correction Model Parameters (nn_block_dim)', fontweight='bold')
        ax.set_ylabel('NRMSE', fontweight='bold')
        ax.set_title('Estimation Error vs Model Size', fontweight='bold')
        ax.set_xscale('log', base=2)
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')

    # Panel 3: ACAUC vs shrinkage strength
    ax = axes[1, 0]
    if not shrinkage_df.empty:
        shrinkage_sorted = shrinkage_df.sort_values('shrinkage_value')
        ax.errorbar(
            shrinkage_sorted['shrinkage_value'],
            shrinkage_sorted['acauc_mean'],
            yerr=shrinkage_sorted['acauc_std'],
            marker='o',
            markersize=8,
            linewidth=2,
            capsize=5,
            color='#2E86AB',
            label='ACAUC'
        )
        ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='Perfect')
        ax.set_xlabel('Shrinkage Prior Strength (λ_shrinkage)', fontweight='bold')
        ax.set_ylabel('ACAUC', fontweight='bold')
        ax.set_title('Coverage vs Shrinkage Prior', fontweight='bold')
        ax.set_xscale('log')
        # Handle zero value by adding small offset
        xlim = ax.get_xlim()
        ax.set_xlim(1e-3, xlim[1])
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')

    # Panel 4: NRMSE vs shrinkage strength
    ax = axes[1, 1]
    if not shrinkage_df.empty:
        shrinkage_sorted = shrinkage_df.sort_values('shrinkage_value')
        ax.errorbar(
            shrinkage_sorted['shrinkage_value'],
            shrinkage_sorted['nrmse_mean'],
            yerr=shrinkage_sorted['nrmse_std'],
            marker='s',
            markersize=8,
            linewidth=2,
            capsize=5,
            color='#A23B72',
            label='NRMSE'
        )
        ax.set_xlabel('Shrinkage Prior Strength (λ_shrinkage)', fontweight='bold')
        ax.set_ylabel('NRMSE', fontweight='bold')
        ax.set_title('Estimation Error vs Shrinkage Prior', fontweight='bold')
        ax.set_xscale('log')
        # Handle zero value
        xlim = ax.get_xlim()
        ax.set_xlim(1e-3, xlim[1])
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')

    plt.tight_layout()

    # Save figure
    save_path = Path(save_dir) / 'ablation_study.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved ablation plot: {save_path}")

    plt.close()


def print_summary_tables(df: pd.DataFrame):
    """Print summary tables for ablation studies."""
    df = extract_ablation_params(df)

    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)

    # Parameter size ablation
    params_df = df[df['experiment_type'] == 'params'].copy()
    if not params_df.empty:
        params_sorted = params_df.sort_values('param_size')
        print("\nParameter Size Ablation (nn_block_dim):")
        print("-" * 80)
        print(f"{'Size':<8} {'ACAUC':>12} {'NRMSE':>12} {'LPP':>12}")
        print("-" * 80)
        for _, row in params_sorted.iterrows():
            size = int(row['param_size'])
            acauc = row['acauc_mean']
            acauc_std = row['acauc_std']
            nrmse = row['nrmse_mean']
            nrmse_std = row['nrmse_std']
            lpp = row['lpp_median']

            print(f"{size:<8} {acauc:>6.3f}±{acauc_std:<.3f} {nrmse:>6.4f}±{nrmse_std:<.4f} {lpp:>10.2f}")

    # Shrinkage prior ablation
    shrinkage_df = df[df['experiment_type'] == 'shrinkage'].copy()
    if not shrinkage_df.empty:
        shrinkage_sorted = shrinkage_df.sort_values('shrinkage_value')
        print("\n\nShrinkage Prior Ablation (λ_shrinkage):")
        print("-" * 80)
        print(f"{'Lambda':<10} {'ACAUC':>12} {'NRMSE':>12} {'LPP':>12}")
        print("-" * 80)
        for _, row in shrinkage_sorted.iterrows():
            lam = row['shrinkage_value']
            acauc = row['acauc_mean']
            acauc_std = row['acauc_std']
            nrmse = row['nrmse_mean']
            nrmse_std = row['nrmse_std']
            lpp = row['lpp_median']

            print(f"{lam:<10.2f} {acauc:>6.3f}±{acauc_std:<.3f} {nrmse:>6.4f}±{nrmse_std:<.4f} {lpp:>10.2f}")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Generate ablation study plots for ICML 2026'
    )
    parser.add_argument('--metrics-db', type=str, required=True,
                       help='Path to ablation metrics CSV file')
    parser.add_argument('--save-dir', type=str, default='publication_plots_ablation',
                       help='Directory to save plots')

    args = parser.parse_args()

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    print("="*80)
    print("ABLATION STUDY PLOTTING")
    print("="*80)
    print(f"Metrics database: {args.metrics_db}")
    print(f"Save directory: {args.save_dir}")
    print("")

    # Load metrics
    df = load_ablation_metrics(args.metrics_db)

    # Print summary tables
    print_summary_tables(df)

    # Generate plots
    print("\nGenerating ablation study plot...")
    plot_ablation_study(df, args.save_dir)

    print("\n" + "="*80)
    print("ABLATION PLOTTING COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
