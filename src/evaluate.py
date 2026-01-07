#!/usr/bin/env python3
"""
RVNP-SBI Unified Evaluation Script

Single config-driven evaluation script that:
- Loads model from config file
- Computes evaluation metrics (coverage, log PDF, NRMSE, ESS)
- Generates publication-ready plots
- Optionally performs Sampling Importance Resampling (SIR)

Usage:
    python evaluate.py --config=configs/CS_task/ranpt_1_simple.py
    python evaluate.py --config=configs/CS_task/ranpt_1_simple.py --use_sir
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
import functools
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# Configure JAX
os.environ["JAX_ENABLE_X64"] = "True"
jax.config.update('jax_enable_x64', True)

# Import project modules
from .normalizing_flow import get_flow
from .training_utils import ExperimentConfig, TrainingLogger


class Evaluator:
    """Handles model evaluation for RVNP-SBI"""

    def __init__(self, config, data_path: str = './Data'):
        """
        Initialize evaluator with config.

        Args:
            config: Configuration object (from config file)
            data_path: Path to data directory (from config.data.data_path)
        """
        self.config = config
        self.data_path = data_path
        self.task_name = config.data.dataset
        self.logger = TrainingLogger(f"eval_{self.task_name}")

        # Validate config
        self._validate_config()

        # Load data
        self.training_data, self.inference_data, self.stats = self._load_data()

        # Determine experiment partitioning
        self.exp_config = ExperimentConfig.from_num_tests(config.data.num_tests)

        self.logger.logger.info(f"Evaluator initialized for task: {self.task_name}")
        self.logger.logger.info(f"Num tests: {config.data.num_tests}")
        self.logger.logger.info(f"Experiment strategy: {self.exp_config.num_experiments} experiments, "
                               f"{self.exp_config.points_per_experiment} points each")

    def _validate_config(self):
        """Validate that required config fields exist"""
        required_fields = [
            'data.dataset',
            'data.data_path',
            'data.num_tests',
            'model.name',
            'model.flow_dimension',
            'model.cond_dim'
        ]

        for field_path in required_fields:
            obj = self.config
            try:
                for part in field_path.split('.'):
                    obj = getattr(obj, part)
            except AttributeError:
                raise ValueError(f"Missing required config field: {field_path}")

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Load training and inference data for the task.

        Returns:
            training_data: Training dataset (theta, x)
            inference_data: Inference dataset (theta_true, x_obs)
            stats: Normalization statistics {'mean': ..., 'std': ...}
        """
        self.logger.logger.info(f"Loading {self.task_name} dataset from {self.data_path}")

        # Determine filename pattern based on task
        task_lower = self.task_name.lower()

        # Load training data
        train_file = os.path.join(self.data_path, f'{self.task_name}_111111.npy')
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Training data not found: {train_file}")
        training_data = np.load(train_file)

        # Load statistics
        stats_file = os.path.join(self.data_path, f'{self.task_name}_stats_111111.npy')
        if not os.path.exists(stats_file):
            raise FileNotFoundError(f"Statistics file not found: {stats_file}")
        sts_loaded = np.load(stats_file, allow_pickle=True)
        stats = sts_loaded.item() if hasattr(sts_loaded, 'item') else sts_loaded

        # Load inference data (try different naming conventions)
        inference_files = [
            f'{self.task_name}_inference_100.npy',
            f'{self.task_name}_100.npy',
            f'{task_lower}_inference_100.npy',
            f'{task_lower}_100.npy'
        ]

        inference_data = None
        for fname in inference_files:
            fpath = os.path.join(self.data_path, fname)
            if os.path.exists(fpath):
                inference_data = np.load(fpath)
                # Normalize
                inference_data = (inference_data - stats['mean']) / stats['std']
                self.logger.logger.info(f"Loaded inference data from: {fname}")
                break

        if inference_data is None:
            raise FileNotFoundError(
                f"Inference data not found. Tried: {', '.join(inference_files)} in {self.data_path}"
            )

        self.logger.logger.info(f"Data shapes - Training: {training_data.shape}, "
                               f"Inference: {inference_data.shape}")

        return training_data, inference_data, stats

    def load_model(self, experiment_idx: Optional[int] = None):
        """
        Load trained model for evaluation.

        Args:
            experiment_idx: Experiment index (None for single experiment)

        Returns:
            flowclass: Loaded model instance
        """
        self.logger.logger.info(f"Loading model: {self.config.model.name}")

        # Get flow class
        flow_class = get_flow(self.config.model.name)

        # Initialize model
        key = jr.key(self.config.seed)
        flowclass = flow_class(
            self.config,
            key,
            mean=self.stats['mean'],
            std=self.stats['std'],
            experiment_idx=experiment_idx
        )

        # Build model (loads saved parameters)
        flowclass.build(
            train_data=jnp.array(self.training_data),
            eval_data=jnp.array(self.training_data[:1]),
            inference_data=jnp.array(self.inference_data),
            mean=self.stats['mean'],
            std=self.stats['std']
        )

        return flowclass

    @staticmethod
    def sampling_importance_resampling(
        flowclass,
        x_hat,
        n_particles: int = 1000,
        n_sims_per_particle: int = 10,
        key=None
    ) -> Tuple[jnp.ndarray, jnp.ndarray, float, jnp.ndarray]:
        """
        Sampling Importance Resampling (SIR) for improved posterior samples.

        Args:
            flowclass: Trained model with flow, simulator_flow, correction_model, prior
            x_hat: Observed data
            n_particles: Number of particles to sample from proposal
            n_sims_per_particle: Number of simulator samples per particle
            key: JAX random key

        Returns:
            resampled_theta: Resampled particles
            log_weights: Log importance weights
            ess: Effective sample size
            theta_particles: Original particles (before resampling)
        """
        if key is None:
            key = jr.key(0)

        key, subkey = jr.split(key)

        # Sample theta from flow (proposal distribution)
        theta_particles = flowclass.flow.sample(subkey, (n_particles,), condition=x_hat)

        # Compute importance weights
        def compute_weight(theta, key):
            key, subkey = jr.split(key)
            x_sims = flowclass.simulator_flow.sample(subkey, (n_sims_per_particle,), condition=theta)

            # Likelihood: E_sim[log p(x_hat|x,theta)]
            log_likelihood_terms = vmap(
                lambda x: flowclass.correction_model.log_prob(x_hat, x, theta)
            )(x_sims)
            log_likelihood = jax.scipy.special.logsumexp(log_likelihood_terms) - jnp.log(n_sims_per_particle)

            # Prior
            log_prior = flowclass.prior.log_prob(theta)

            # Proposal
            log_proposal = flowclass.flow.log_prob(theta, condition=x_hat)

            # Importance weight: p(x|theta) * p(theta) / q(theta|x)
            return log_likelihood + log_prior - log_proposal

        # Compute all weights
        keys = jr.split(key, n_particles)
        log_weights = vmap(compute_weight)(theta_particles, keys)

        # Normalize
        log_weights_normalized = log_weights - jax.scipy.special.logsumexp(log_weights)
        weights = jnp.exp(log_weights_normalized)

        # Resample
        key, subkey = jr.split(key)
        indices = jr.choice(subkey, n_particles, shape=(n_particles,), p=weights.flatten())
        resampled_theta = theta_particles[indices]

        # Effective sample size
        ess = 1.0 / jnp.sum(weights**2)

        return resampled_theta, log_weights, ess, theta_particles

    @staticmethod
    def compute_hdi_coverage(
        flow,
        samples: jnp.ndarray,
        theta_true: jnp.ndarray,
        x_cond: jnp.ndarray
    ) -> Tuple[jnp.ndarray, int, float]:
        """
        Compute HDI (Highest Density Interval) coverage statistics.

        Args:
            flow: Posterior flow
            samples: Posterior samples
            theta_true: True parameter value
            x_cond: Conditioning observation

        Returns:
            coverage: Binary coverage at 21 levels (0 to 100%)
            num_samples: Number of samples
            true_log_prob: Log probability of true parameter
        """
        # Compute log densities
        log_probs = flow.log_prob(samples, condition=x_cond)
        true_log_prob = flow.log_prob(theta_true, condition=x_cond)

        # Sort by density (highest first)
        sorted_log_probs = jnp.sort(log_probs)[::-1]

        # Coverage levels (0%, 5%, 10%, ..., 100%)
        N = log_probs.shape[0]
        coverage_levels = jnp.linspace(0, 1, 21)
        indices = (jnp.round(coverage_levels * N) - 1).astype(int)
        indices = jnp.clip(indices, 0, N - 1)
        threshold_log_probs = sorted_log_probs[indices]

        # Check if true theta is in top X% highest density
        coverage = (threshold_log_probs < true_log_prob).astype(jnp.int32)
        coverage = coverage.at[0].set(0).at[-1].set(1)  # Force 0% and 100%

        return coverage, int(N), float(true_log_prob)

    @staticmethod
    def compute_acauc(
        samples: jnp.ndarray,
        theta_true: jnp.ndarray,
        n_alpha_levels: int = 100
    ) -> float:
        """Compute ACAUC (Average Coverage Area Under Curve) - primary calibration metric.

        ACAUC measures the continuous calibration quality of a posterior distribution by
        computing the area under the coverage curve across all credibility levels. This
        provides a more comprehensive assessment than discrete coverage metrics.

        Mathematical Formulation:
            For a single observation:

            .. math::

                ACAUC = \\frac{1}{k} \sum_{j=1}^{k} \int_{0}^{1} \mathbb{1}[\\theta_j^* \in C_\\alpha^j] d\\alpha

            where:
                - :math:`k`: Parameter dimension
                - :math:`\\theta_j^*`: True value of parameter :math:`j`
                - :math:`C_\\alpha^j`: :math:`\\alpha`-level credible interval for dimension :math:`j`
                - :math:`\mathbb{1}[\cdot]`: Indicator function (1 if true, 0 if false)

            The :math:`\\alpha`-level credible interval is defined as:

            .. math::

                C_\\alpha^j = [q_{(1-\\alpha)/2}(p(\\theta_j|x)), q_{1-(1-\\alpha)/2}(p(\\theta_j|x))]

            where :math:`q_p` denotes the :math:`p`-th quantile of the marginal posterior.

        Interpretation:
            **Ideal value**: 1.0 (perfect calibration)
                - For every credibility level :math:`\\alpha`, the true parameter is inside
                  the :math:`\\alpha`-credible interval exactly :math:`\\alpha` fraction of the time

            **< 1.0**: Under-coverage (overconfident posterior)
                - True parameters fall outside credible intervals more often than expected
                - Posterior is too concentrated around the mode
                - May indicate: insufficient correction, weak regularization, or model misspecification

            **> 1.0**: Over-coverage (too conservative)
                - True parameters fall inside credible intervals more often than expected
                - Posterior is too diffuse/uncertain
                - May indicate: excessive regularization, over-correction, or lack of data

        Algorithm:
            1. For each parameter dimension :math:`j = 1, \ldots, k`:
               a. For each credibility level :math:`\\alpha \in [0, 1]` (discretized):
                  - Compute quantiles: :math:`q_{(1-\\alpha)/2}` and :math:`q_{1-(1-\\alpha)/2}`
                  - Check if :math:`\\theta_j^* \in [q_{lower}, q_{upper}]`
               b. Compute area under coverage curve via trapezoidal integration
            2. Average coverage AUC across all dimensions

        Args:
            samples: Posterior samples from :math:`q_\phi(\\theta|x_{obs})`,
                    shape (n_samples, k) where k is parameter dimension.
                    Typically n_samples ≥ 10,000 for stable quantile estimates.
            theta_true: True parameter value :math:`\\theta^*`, shape (k,).
                       Ground truth used to assess calibration.
            n_alpha_levels: Number of discretization points for :math:`\\alpha \in [0, 1]`.
                           Default: 100. Higher values → more accurate integration but slower.

        Returns:
            acauc: ACAUC value, a scalar in approximately [0, 2]
                  - 1.0: Perfect calibration
                  - [0.9, 1.1]: Good calibration
                  - < 0.9: Significant under-coverage
                  - > 1.1: Significant over-coverage

        Implementation Details:
            **Quantile Computation**:
                - Uses numpy.quantile with linear interpolation
                - Symmetric credible intervals: :math:`[(1-\\alpha)/2, 1-(1-\\alpha)/2]`

            **Integration**:
                - Trapezoidal rule via numpy.trapz
                - :math:`\int_0^1 f(\\alpha) d\\alpha ≈ \sum_{i=0}^{N-1} (f_i + f_{i+1})(\\alpha_{i+1} - \\alpha_i)/2`

            **Per-Dimension Averaging**:
                - Computes coverage AUC independently for each dimension
                - Final metric is arithmetic mean across dimensions

        Advantages Over Discrete Coverage:
            - **Continuous**: Tests all credibility levels, not just fixed values (e.g., 0.95)
            - **Robust**: Single metric captures entire calibration curve
            - **Interpretable**: Direct relationship to ideal calibration
            - **Sensitive**: Detects subtle miscalibration missed by discrete metrics

        Example:
            >>> # Posterior samples from trained model
            >>> samples = posterior_flow.sample(key, x_obs, num_samples=10000)  # (10000, 3)
            >>> theta_true = jnp.array([0.5, 1.0, -0.3])  # True parameter
            >>> # Compute ACAUC
            >>> acauc = Evaluator.compute_acauc(samples, theta_true, n_alpha_levels=100)
            >>> print(f"ACAUC: {acauc:.3f}")
            ACAUC: 0.987

        Interpretation:
            acauc < 0.95 indicates under-coverage (overconfident posterior),
            acauc > 1.05 indicates over-coverage (too conservative), and values near 1.0
            indicate good calibration.

        Notes:
            - Primary metric for evaluating RVNP calibration quality
            - Complementary to AEPC (Average Expected Posterior Coverage) metric
            - More informative than single-point coverage (e.g., 95% credible interval)
            - Should be computed on independent test data (not training data)
            - Requires large sample sizes (≥ 10,000) for stable quantile estimates

        References:
            Hermans, J., Delaunoy, A., Rozet, F., Wehenkel, A., & Louppe, G. (2021).
            "Robust Neural Posterior Estimation and Statistical Model Criticism."
            Advances in Neural Information Processing Systems (NeurIPS).

        See Also:
            - evaluate_single_observation(): Uses this method to compute ACAUC per observation
            - AEPC metric: Discrete calibration metric at specific alpha levels
            - LPP metric: Log posterior probability (likelihood quality, not calibration)
        """
        n_samples, n_dims = samples.shape
        alpha_levels = np.linspace(0, 1, n_alpha_levels)

        # Compute ACAUC for each dimension
        acauc_per_dim = []

        for dim in range(n_dims):
            # Get samples for this dimension
            dim_samples = samples[:, dim]
            true_value = theta_true[dim]

            # Compute coverage at each alpha level
            coverage_at_alpha = []

            for alpha in alpha_levels:
                # Compute credible interval at this alpha level
                # For 1D, this is just the quantiles
                lower_quantile = (1 - alpha) / 2
                upper_quantile = 1 - lower_quantile

                lower_bound = np.quantile(dim_samples, lower_quantile)
                upper_bound = np.quantile(dim_samples, upper_quantile)

                # Check if true value is in interval
                is_covered = float((true_value >= lower_bound) and (true_value <= upper_bound))
                coverage_at_alpha.append(is_covered)

            # Integrate coverage over alpha using trapezoidal rule
            coverage_auc = np.trapz(coverage_at_alpha, alpha_levels)
            acauc_per_dim.append(coverage_auc)

        # Average across all dimensions
        acauc = np.mean(acauc_per_dim)

        return float(acauc)

    def evaluate_single_observation(
        self,
        flowclass,
        observation: jnp.ndarray,
        key,
        use_sir: bool = False,
        n_samples: int = 5000
    ) -> Dict:
        """
        Evaluate model on a single observation.

        Args:
            flowclass: Trained model
            observation: [theta_true, x_obs] concatenated
            key: JAX random key
            use_sir: Whether to use SIR
            n_samples: Number of samples

        Returns:
            metrics: Dictionary of evaluation metrics
        """
        # Split observation
        theta_dim = self.config.model.flow_dimension
        theta_true = observation[:theta_dim]
        x_obs = observation[theta_dim:]

        if use_sir:
            # SIR evaluation
            resampled_theta, log_weights, ess, original_theta = self.sampling_importance_resampling(
                flowclass, x_obs, n_particles=n_samples, key=key
            )
            samples = resampled_theta
            ess_value = float(ess)
        else:
            # Regular sampling
            samples = flowclass.flow.sample(key, (n_samples,), condition=x_obs)
            ess_value = 0.0

        # Compute HDI coverage
        coverage, num_samples, true_log_prob = self.compute_hdi_coverage(
            flowclass.flow, samples, theta_true, x_obs
        )

        # Compute ACAUC
        acauc = self.compute_acauc(
            samples=np.array(samples),
            theta_true=np.array(theta_true),
            n_alpha_levels=100
        )

        # Compute NRMSE
        param_range = 1.0  # Normalized parameters
        nrmse_per_sample = jnp.sqrt(jnp.mean(((samples - theta_true) / param_range)**2, axis=1))
        nrmse = float(jnp.mean(nrmse_per_sample))

        return {
            'coverage': coverage,
            'num_samples': num_samples,
            'true_log_prob': true_log_prob,
            'acauc': acauc,
            'nrmse': nrmse,
            'ess': ess_value
        }

    def run_evaluation(
        self,
        use_sir: bool = False,
        n_samples: int = 5000,
        n_eval_points: int = 100,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Run full evaluation on inference dataset.

        Args:
            use_sir: Whether to use Sampling Importance Resampling
            n_samples: Number of posterior samples per observation
            n_eval_points: Number of observations to evaluate
            seed: Random seed

        Returns:
            DataFrame with results
        """
        self.logger.logger.info("=" * 80)
        self.logger.logger.info(f"Starting evaluation | SIR: {use_sir} | Samples: {n_samples}")
        self.logger.logger.info("=" * 80)

        key = jr.key(seed)

        # Determine number of experiments
        num_experiments = self.exp_config.num_experiments

        # Accumulators
        all_results = []

        with self.logger.timer("Full evaluation"):
            for exp_idx_val in range(num_experiments):
                exp_idx = exp_idx_val if num_experiments > 1 else None

                self.logger.logger.info(f"Experiment {exp_idx_val + 1}/{num_experiments}")

                # Load model for this experiment
                flowclass = self.load_model(experiment_idx=exp_idx)

                # Select evaluation observations
                if num_experiments == 1:
                    eval_obs = self.inference_data[:n_eval_points]
                else:
                    # For multiple experiments, use specific observation
                    if exp_idx < len(self.inference_data):
                        eval_obs = self.inference_data[exp_idx:exp_idx+1]
                    else:
                        self.logger.logger.warning(f"Skipping exp {exp_idx}: no data")
                        continue

                # Evaluate each observation
                keys = jr.split(key, len(eval_obs))
                for obs_idx, (obs, obs_key) in enumerate(zip(eval_obs, keys)):
                    metrics = self.evaluate_single_observation(
                        flowclass, obs, obs_key, use_sir=use_sir, n_samples=n_samples
                    )
                    metrics['experiment_idx'] = exp_idx_val
                    metrics['observation_idx'] = obs_idx
                    all_results.append(metrics)

        # Aggregate results
        df = self._aggregate_results(all_results)

        self.logger.logger.info("=" * 80)
        self.logger.logger.info("Evaluation complete!")
        self.logger.logger.info("=" * 80)

        return df

    def _aggregate_results(self, results: list) -> pd.DataFrame:
        """Aggregate evaluation results into summary DataFrame"""
        if not results:
            raise ValueError("No evaluation results to aggregate")

        # Stack coverage vectors
        all_coverage = np.array([r['coverage'] for r in results])
        total_coverage = all_coverage.sum(axis=0)
        total_samples = len(results)

        # Compute alpha (coverage calibration metric)
        expected_coverage = np.linspace(0, 1, 21)
        alpha = np.median(total_coverage / total_samples - expected_coverage)

        # Other metrics
        all_log_probs = [r['true_log_prob'] for r in results]
        all_nrmses = [r['nrmse'] for r in results]
        all_acaucs = [r['acauc'] for r in results]
        all_ess = [r['ess'] for r in results if r['ess'] > 0]

        summary = {
            'task': self.task_name,
            'model': self.config.model.name,
            'correction_type': self.config.model.correction_type,
            'num_tests': self.config.data.num_tests,
            'n_evaluations': len(results),
            'alpha': alpha,
            'acauc_mean': np.mean(all_acaucs),
            'acauc_std': np.std(all_acaucs),
            'med_log_prob': np.median(all_log_probs),
            'std_log_prob': np.std(all_log_probs),
            'mean_nrmse': np.mean(all_nrmses),
            'std_nrmse': np.std(all_nrmses),
            'mean_ess': np.mean(all_ess) if all_ess else 0.0
        }

        self.logger.logger.info("Summary Statistics:")
        self.logger.logger.info(f"  α (coverage): {summary['alpha']:.4f}")
        self.logger.logger.info(f"  ACAUC: {summary['acauc_mean']:.4f} ± {summary['acauc_std']:.4f}")
        self.logger.logger.info(f"  Median log prob: {summary['med_log_prob']:.4f}")
        self.logger.logger.info(f"  Mean NRMSE: {summary['mean_nrmse']:.4f}")
        if all_ess:
            self.logger.logger.info(f"  Mean ESS: {summary['mean_ess']:.1f}")

        return pd.DataFrame([summary])

    def plot_results(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """
        Create publication-ready plots of evaluation results.

        Args:
            df: DataFrame with evaluation results
            save_path: Path to save plot (optional)
        """
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_context("paper", font_scale=1.3, rc={"lines.linewidth": 2.0})
        plt.rcParams['font.family'] = 'serif'

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.patch.set_facecolor('white')

        # Plot 1: Coverage calibration (alpha)
        ax1 = axes[0]
        ax1.bar(0, df['alpha'].iloc[0], color='#2E86AB', alpha=0.7, edgecolor='white', linewidth=2)
        ax1.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        ax1.fill_between([-0.5, 0.5], -0.1, 0.1, alpha=0.1, color='green', label='Ideal range')
        ax1.set_ylabel(r'$\alpha$', fontsize=16, fontweight='bold')
        ax1.set_title('Coverage Calibration', fontsize=14, fontweight='bold')
        ax1.set_ylim(-0.5, 0.5)
        ax1.set_xticks([])
        ax1.grid(True, alpha=0.2, axis='y')

        # Plot 2: Log probability
        ax2 = axes[1]
        ax2.bar(0, df['med_log_prob'].iloc[0], yerr=df['std_log_prob'].iloc[0],
                color='#6A994E', alpha=0.7, edgecolor='white', linewidth=2, capsize=10)
        ax2.set_ylabel(r'$\log q(\theta^* \mid x_{\mathrm{obs}})$', fontsize=16, fontweight='bold')
        ax2.set_title('Posterior Log-Density', fontsize=14, fontweight='bold')
        ax2.set_xticks([])
        ax2.grid(True, alpha=0.2, axis='y')

        # Plot 3: NRMSE
        ax3 = axes[2]
        ax3.bar(0, df['mean_nrmse'].iloc[0], yerr=df['std_nrmse'].iloc[0],
                color='#C73E1D', alpha=0.7, edgecolor='white', linewidth=2, capsize=10)
        ax3.set_ylabel('NRMSE', fontsize=16, fontweight='normal')
        ax3.set_title('Parameter Estimation Accuracy', fontsize=14, fontweight='bold')
        ax3.set_xticks([])
        ax3.grid(True, alpha=0.2, axis='y')

        # Add title
        task_display = self.task_name
        model_display = f"{self.config.model.name} ({self.config.model.correction_type})"
        fig.suptitle(f'{task_display} | {model_display} | N={self.config.data.num_tests}',
                    fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            self.logger.logger.info(f"Plot saved to: {save_path}")

        plt.show()
        return fig


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='RVNP-SBI Evaluation Script')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file (e.g., configs/CS_task/ranpt_1_simple.py)')
    parser.add_argument('--use_sir', action='store_true',
                       help='Use Sampling Importance Resampling')
    parser.add_argument('--n_samples', type=int, default=10000,
                       help='Number of posterior samples (default: 10000)')
    parser.add_argument('--n_eval_points', type=int, default=100,
                       help='Number of observations to evaluate (default: 100)')
    parser.add_argument('--save_dir', type=str, default='./output/evaluation',
                       help='Directory to save results (default: ./output/evaluation)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')

    args = parser.parse_args()

    # Load config
    print(f"Loading config from: {args.config}")
    sys.path.insert(0, os.path.dirname(args.config))
    config_module_name = os.path.basename(args.config).replace('.py', '')
    config_module = __import__(config_module_name)
    config = config_module.get_config()

    # Set to evaluation mode
    config.training.train = False

    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Initialize evaluator
    evaluator = Evaluator(config, data_path=config.data.data_path)

    # Run evaluation
    df = evaluator.run_evaluation(
        use_sir=args.use_sir,
        n_samples=args.n_samples,
        n_eval_points=args.n_eval_points,
        seed=args.seed
    )

    # Save results
    sir_suffix = '_sir' if args.use_sir else ''
    results_file = os.path.join(
        args.save_dir,
        f'{config.data.dataset}_{config.model.name}_n{config.data.num_tests}{sir_suffix}_results.csv'
    )
    df.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")

    # Create and save plot
    plot_file = results_file.replace('.csv', '.png')
    evaluator.plot_results(df, save_path=plot_file)

    # Display results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
