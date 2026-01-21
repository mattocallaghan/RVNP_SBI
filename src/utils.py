# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Utility code for generating and saving image grids and checkpointing.

   The `save_image` code is copied from
   https://github.com/google/flax/blob/master/examples/vae/utils.py,
   which is a JAX equivalent to the same function in TorchVision
   (https://github.com/pytorch/vision/blob/master/torchvision/utils.py)
"""
import numba
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TypeVar
import flax
import jax
import jax.numpy as jnp
from PIL import Image
import tensorflow as tf
import optax
import os
import numpy as np
#import sbibm
import pandas as pd

from jax import jit, vmap, lax
from jax import random

T = TypeVar("T")


def batch_add(a, b):
  return jax.vmap(lambda a, b: a + b)(a, b)


def batch_mul(a, b):
  return jax.vmap(lambda a, b: a * b)(a, b)


def load_training_state(filepath, state):
  with tf.io.gfile.GFile(filepath, "rb") as f:
    state = flax.serialization.from_bytes(state, f.read())
  return state





def flatten_dict(config):
  """Flatten a hierarchical dict to a simple dict."""
  new_dict = {}
  for key, value in config.items():
    if isinstance(value, dict):
      sub_dict = flatten_dict(value)
      for subkey, subvalue in sub_dict.items():
        new_dict[key + "/" + subkey] = subvalue
    elif isinstance(value, tuple):
      new_dict[key] = str(value)
    else:
      new_dict[key] = value
  return new_dict



def get_optimizer(config, lr_override=None):
    """Returns a flax optimizer object based on `config`.

    Args:
        config: Configuration object containing optimizer settings
        lr_override: Optional learning rate override for dynamic scheduling.
                    If provided, this LR is used instead of config.optim.lr.

    Returns:
        Optax optimizer (chain of transformations)
    """
    # Use override LR if provided, otherwise use config
    effective_lr = lr_override if lr_override is not None else config.optim.lr

    if config.optim.optimizer.lower() == "adam":
        if hasattr(config.optim, "linear_decay_steps"):  # for progressive distillation
            stable_training_schedule = optax.linear_schedule(
                init_value=effective_lr,
                end_value=0.0,
                transition_steps=config.optim.linear_decay_steps,
            )
        else:
            stable_training_schedule = optax.constant_schedule(effective_lr)
        schedule = optax.join_schedules(
            [
                optax.linear_schedule(
                    init_value=0,
                    end_value=effective_lr,
                    transition_steps=config.optim.warmup,
                ),
                stable_training_schedule,
            ],
            [config.optim.warmup],
        )

        if not jnp.isinf(config.optim.grad_clip):
            optimizer = optax.chain(
                optax.clip_by_global_norm(max_norm=config.optim.grad_clip),
                optax.adamw(
                    learning_rate=schedule,
                    b1=config.optim.beta1,
                    eps=config.optim.eps,
                    weight_decay=config.optim.weight_decay,
                ),
            )
        else:
            optimizer = optax.adamw(
                learning_rate=schedule,
                b1=config.optim.beta1,
                eps=config.optim.eps,
                weight_decay=config.optim.weight_decay,
            )

    elif config.optim.optimizer.lower() == "radam":
        beta1 = config.optim.beta1
        beta2 = config.optim.beta2
        eps = config.optim.eps
        weight_decay = config.optim.weight_decay
        lr = config.optim.lr
        optimizer = optax.chain(
            optax.scale_by_radam(b1=beta1, b2=beta2, eps=eps),
            optax.add_decayed_weights(weight_decay, None),
            optax.scale(-lr),
        )
    else:
        raise NotImplementedError(
            f"Optimizer {config.optim.optimizer} not supported yet!"
        )


    return optimizer

  






class Task(ABC):
    """Abstract base class for simulation-based inference tasks.

    Defines the interface for SBI tasks including prior sampling, simulation,
    observation generation, and dataset creation. Each task implements a specific
    simulator and defines how to generate training/inference data.

    Subclasses must implement:
        - sample_prior(): Draw parameters from prior p(θ)
        - simulate(): Run simulator to generate x ~ p(x|θ)
        - generate_observation(): Create observed data x_obs for inference

    Attributes:
        name: Task name (automatically set to class name)
        scales: Dictionary of normalization statistics (mean, std) if scaled

    Example:
        >>> class MyTask(Task):
        ...     def sample_prior(self, key, n):
        ...         return jax.random.uniform(key, (n, 3))
        ...     def simulate(self, key, theta):
        ...         return my_simulator(theta)
        ...     def generate_observation(self, key, misspecified=True):
        ...         theta_true = jnp.array([0.5, 1.0, -0.3])
        ...         x_obs = my_simulator(theta_true)
        ...         return theta_true, x_obs
    """
    @property
    def name(self):
        return type(self).__name__

    @abstractmethod
    def sample_prior(self, key: random.PRNGKey, n: int):
        "Draw n samples from the prior."
        pass

    @abstractmethod
    def simulate(self, key: random.PRNGKey, theta: jnp.ndarray):
        "Carry out simulations."
        pass

    @abstractmethod
    def generate_observation(self, key: random.PRNGKey, misspecified=True):
        "Generate misspecified pseudo-observed data. Returns (theta_true, y)."
        pass

    def generate_dataset(
        self, key: random.PRNGKey, n: int, scale=True, misspecified=True,
    ):
        "Generate optionally scaled dataset with pseudo-observed value and simulations"
        theta_key, x_key, obs_key = random.split(key, 3)
        theta = self.sample_prior(theta_key, n)
        x = self.simulate(x_key, theta)
        x = remove_nans_and_warn(x)

        y_raw= self.simulate(obs_key, theta, summarise=True)
        #y_raw = self.misspecify(y_raw) if misspecified else y_raw
        #y = self.summarise(y_raw)
        y=y_raw

        if scale:
            theta, theta_true, x, y = self.scale(theta, theta_true, x, y)

        data = {
            "theta": theta,
            "theta_true": theta,
            "x": x,
            "y": y,
            "y_raw": y_raw,
        }

        return data

    def scale(self, theta, theta_true, x, y):
        "Center and scale data, using mean and std from theta and x."
        theta_means, theta_stds = theta.mean(axis=0), theta.std(axis=0)
        theta = (theta - theta_means) / theta_stds
        theta_true = (theta_true - theta_means) / theta_stds

        x_means, x_stds = x.mean(axis=0), x.std(axis=0)
        x = (x - x_means) / x_stds
        y = (y - x_means) / x_stds

        self.scales = {
            "theta_mean": theta_means,
            "theta_std": theta_stds,
            "x_mean": x_means,
            "x_std": x_stds,
        }
        return theta, theta_true, x, y

    def in_prior_support(self, theta):
        return jnp.full(theta.shape[0], True)

    def get_true_posterior_samples(self, key: random.PRNGKey, y: jnp.ndarray, n: int):
        raise NotImplementedError(
            "This task does not have a method for sampling the true posterior implemented."
        )


class CS(Task):
    """Compressed Sensing task for spatial cell distribution inference.

    Simulates spatial distribution of cancer cells and compresses observations via
    random projection. The simulator models cell locations in 2D space with dynamics
    that may be misspecified to test robustness of inference methods.

    Task Details:
        - **Parameter dimension**: 3 (θ₁, θ₂, θ₃ control cell distribution)
        - **Observation dimension**: 4 (summary statistics)
        - **Misspecification**: Simulator uses incorrect spatial dynamics
        - **Prior**: Uniform distribution over parameter space
        - **Use case**: Testing RVNP on structured spatial data

    Mathematical Model:
        The true data generating process differs from the simulator, creating
        model misspecification that standard NPE cannot handle but RVNP corrects for.

    Attributes:
        name: "CS" (compressed sensing)
        scales: Normalization statistics if dataset is scaled

    Methods:
        sample_prior(key, n): Sample n parameters from uniform prior
        simulate(key, theta): Generate observations for given parameters
        generate_observation(key, misspecified): Create observed data x_obs

    Example:
        >>> task = CS()
        >>> key = jax.random.PRNGKey(0)
        >>> theta = task.sample_prior(key, n=1000)  # (1000, 3)
        >>> x = task.simulate(key, theta)  # (1000, 4)

    References:
        Adapted from: https://github.com/danielward27/rnpe/blob/main/rnpe/tasks.py
    """
    def __init__(self):
        self.theta_names = [r"$\lambda_c$", r"$\lambda_p$", r"$\lambda_d$"]
        self.x_names = [
            "N Stromal",
            "N Cancer",
            "Mean Min Dist",
            "Max Min Dist",
        ]
        self.cell_rate_lims = {"minval": 200, "maxval": 1500}
        self.parent_rate_lims = {"minval": 3, "maxval": 20}
        self.daughter_rate_lims = {"minval": 10, "maxval": 20}
        self.tractable_posterior = False
    def simulate(
        self,
        key: random.PRNGKey,
        theta: jnp.array,
        summarise: bool = True,
        necrosis: bool = False,
    ):
        theta = np.array(theta)
        np.random.seed(key[1].item())

        x = []
        for row in theta:
            xi = self._simulate(
                cell_rate=row[0],
                cancer_parent_rate=row[1],
                cancer_daughter_rate=row[2],
                necrosis=necrosis,
            )
            if summarise is True:
                xi = self.summarise(*xi)
            x.append(xi)

        if summarise:
            x = np.row_stack(x)

        return x

    def _simulate(
        self,
        cell_rate: float = 1000,
        cancer_parent_rate: float = 5,
        cancer_daughter_rate: float = 30,
        necrosis=False,
    ):
        num_cells = np.random.poisson(cell_rate)
        cells = np.random.uniform(size=(num_cells, 2))

        num_cancer_parents = np.random.poisson(cancer_parent_rate) + 1
        cancer_parents = np.random.uniform(0, 1, size=(num_cancer_parents, 2))

        num_cancer_daughters = (
            np.random.poisson(cancer_daughter_rate, (num_cancer_parents,)) + 1
        )

        dists = dists_between(cells, cancer_parents)  # n_cells by n_parents
        radii = self.n_nearest_dists(dists, num_cancer_daughters)
        is_cancer = (dists <= radii).any(axis=1)

        if necrosis:
            has_necrosis = np.random.binomial(1, p=0.75, size=(num_cancer_parents,))
            has_necrosis = has_necrosis.astype(bool)
            if has_necrosis.sum() > 0:
                bl_array = dists[:, has_necrosis] < (radii[has_necrosis] * 0.8)
                necrotized = np.any(bl_array, axis=1)
                cells, is_cancer = (
                    cells[~necrotized],
                    is_cancer[~necrotized],
                )

        return cells, is_cancer

    def summarise(self, cells, is_cancer, threshold_n_stromal: int = 50):
        """Calculate summary statistics, threshold_n_stromal limits the number
        of stromal cells (trade off for efficiency)."""
        num_cancer = is_cancer.sum()

        if num_cancer == is_cancer.shape[0]:
            print("Warning, no stromal cells. Returning nan for summary statistics.")
            return jnp.full(len(self.x_names), jnp.nan)

        num_stromal = (~is_cancer).sum()
        threshold_num_stromal = min(threshold_n_stromal, num_stromal)
        cancer = cells[is_cancer]
        stromal = cells[~is_cancer][:threshold_num_stromal]

        dists = dists_between(stromal, cancer)

        min_dists = dists.min(axis=1)
        mean_nearest_cancer = min_dists.mean()
        max_nearest_cancer = min_dists.max()

        summaries = [
            num_stromal,
            num_cancer,
            mean_nearest_cancer,
            max_nearest_cancer,
        ]

        return jnp.array(summaries)

    def generate_observation(self, key: random.PRNGKey, misspecified=True,theta=None):
        theta_key, y_key = random.split(key)
        if(theta is not None):
            theta_true = self.sample_prior(theta_key, 1)
            print(theta_true.shape)
            print(theta_true)
            theta_true = theta
        else:
            theta_true = self.sample_prior(theta_key, 1)        
        y_raw = self.simulate(y_key, theta_true, necrosis=misspecified, summarise=False)
        y = self.summarise(*y_raw[0])
        return jnp.squeeze(theta_true), jnp.squeeze(y), y_raw

    def sample_prior(self, key: random.PRNGKey, n: int):
        keys = random.split(key, 3)
        cell_rate = random.uniform(keys[0], (n,), **self.cell_rate_lims)
        cancer_parent_rate = random.uniform(keys[1], (n,), **self.parent_rate_lims)
        cancer_daughter_rate = random.uniform(keys[2], (n,), **self.daughter_rate_lims,)
        return jnp.column_stack([cell_rate, cancer_parent_rate, cancer_daughter_rate])

    def n_nearest_dists(self, dists, n_points):
        "Minimum distance containing n points. n_points to match axis dists in axis 1."
        assert dists.shape[1] == len(n_points)
        d_sorted = np.partition(dists, kth=n_points, axis=0)
        min_d = d_sorted[n_points, np.arange(dists.shape[1])]
        return min_d

    def in_prior_support(self, theta):
        bools = []
        limits = [self.cell_rate_lims, self.parent_rate_lims, self.daughter_rate_lims]
        for col, lims in zip(theta.T, limits):
            in_support = (col > lims["minval"]) & (col < lims["maxval"])
            bools.append(in_support)
        return jnp.column_stack(bools).all(axis=1)


@numba.njit(fastmath=True)
def dists_between(a, b):
    "Returns a.shape[0] by b.shape[0] l2 norms between rows of arrays."
    dists = []
    for ai in a:
        for bi in b:
            dists.append(np.linalg.norm(ai - bi))
    return np.array(dists).reshape(a.shape[0], b.shape[0])


def remove_nans_and_warn(x):
    nan_rows = jnp.any(jnp.isnan(x), axis=1)
    n_nan = nan_rows.sum()
    if n_nan > 0:
        x = x[~nan_rows]
        print(f"Warning {n_nan} simulations contained NAN values have been removed.")
    return x











class SIR(Task):
    """Susceptible-Infected-Recovered epidemiological model task.

    Simulates disease spread dynamics using stochastic differential equations (SDEs)
    implemented in Julia. The simulator may use simplified transmission dynamics
    (misspecified model) to test robustness of inference methods.

    Task Details:
        - **Parameter dimension**: 2 (β = infection rate, γ = recovery rate)
        - **Observation dimension**: 10 (time series of S, I, R compartments)
        - **Misspecification**: Simplified transmission model (multiplier on β)
        - **Prior**: Uniform [0, 0.5]² with constraint β > γ
        - **Use case**: Testing RVNP on dynamical systems with temporal data

    Mathematical Model:
        Standard SIR dynamics:
            dS/dt = -β·S·I/N
            dI/dt = β·S·I/N - γ·I
            dR/dt = γ·I

        Misspecification is introduced via scaling β by misspecify_multiplier,
        creating systematic bias in the training simulator.

    Requirements:
        - Julia installation (1.6+)
        - Julia packages: LinearAlgebra, StochasticDiffEq, NPZ, ArgParse
        - First run may take 1-2 minutes for Julia JIT compilation

    Attributes:
        name: "SIR" (Susceptible-Infected-Recovered)
        julia_env_path: Path to Julia environment
        misspecify_multiplier: Scaling factor for β (creates misspecification)
        scales: Normalization statistics if dataset is scaled

    Methods:
        sample_prior(key, n): Sample n parameters from constrained uniform prior
        simulate(key, theta): Run SDE simulation via Julia
        generate_observation(key, misspecified): Create observed epidemic data

    Example:
        >>> task = SIR(julia_env_path=".", misspecify_multiplier=0.95)
        >>> key = jax.random.PRNGKey(0)
        >>> theta = task.sample_prior(key, n=1000)  # (1000, 2)
        >>> x = task.simulate(key, theta)  # (1000, 10)

    Notes:
        - Constraint β > γ ensures epidemic can spread (R₀ = β/γ > 1)
        - Time series summarized to fixed-length observations
        - Julia backend provides fast SDE integration
    """

    def __init__(self, julia_env_path=".", misspecify_multiplier=0.95):
        self.julia_env_path = julia_env_path
        self.misspecify_multiplier = misspecify_multiplier
        self.x_names = [
            "Mean",  # Mean infections
            "Median",  # Median infections
            "Max",  # Max infections
            "Max Day",  # Day of max infections
            "Half Day",  # Day where half of cumulative total number of infections reached
            "Autocor",  # Autocorrelation lag 1
        ]
        self.theta_names = [r"$\beta$", r"$\gamma$"]
        self.tractable_posterior = False

    def sample_prior(self, key: random.PRNGKey, n: int):
        u1key, u2key = random.split(key)
        x = jnp.sqrt(random.uniform(u1key, (n,))) / 2
        y = random.uniform(u2key, (n,)) * x
        return jnp.column_stack((x, y))

    def simulate(self, key: random.PRNGKey, theta: jnp.ndarray, summarise=True):
        "Simulates 365 days of sir model."
        key = key[1].item()
        # temporarily save parameters to file to be accessed from julia
        jl_script_f = os.path.dirname(os.path.realpath(__file__)) + "/julia/SIR.jl"
        tmp_theta_f = f"_temp_theta_{key}.npz"
        tmp_x_f = f"_temp_x_{key}.npz"
        jnp.savez(tmp_theta_f, theta=theta)

        try:
            # Run julia simulation script, outputing to file
            command = f"""
            julia --project={self.julia_env_path} {jl_script_f} --seed={key}  \
                --theta_path={tmp_theta_f} --output_path={tmp_x_f}
            """
            os.system(command)
            x = jnp.load(tmp_x_f)["x"]
        finally:
            for f in [tmp_theta_f, tmp_x_f]:
                try:
                    os.remove(f)
                except OSError:
                    pass

        if summarise:
            x = self.summarise(x)
        return x

    def summarise(self, x):
        @jax.jit
        @jax.vmap
        def autocorr_lag1(x):
            x1 = x[:-1]
            x2 = x[1:]
            x1_dif = x1 - x1.mean()
            x2_dif = x2 - x2.mean()
            numerator = (x1_dif * x2_dif).sum()
            denominator = jnp.sqrt((x1_dif ** 2).sum() * (x2_dif ** 2).sum())
            return numerator / denominator

        def cumulative_day(x, q):
            "Day when q proportion of total infections was reached."
            prop_i = (jnp.cumsum(x, axis=1).T / jnp.sum(x, axis=1)).T
            return jnp.argmax(prop_i > q, axis=1)

        summaries = [
            jnp.log(x.mean(axis=1)),
            jnp.log(jnp.median(x, axis=1)),
            jnp.log(jnp.max(x, axis=1)),
            jnp.log(jnp.argmax(x, axis=1) + 1),  # +1 incase 0 is max_day
            jnp.log(cumulative_day(x, 0.5)),
            autocorr_lag1(x),
        ]

        summaries = jnp.column_stack(summaries)
        return summaries

    def generate_observation(self, key: random.PRNGKey, misspecified=True,theta=None):
        theta_key, y_key = random.split(key)
        if(theta is not None):
            theta_true = self.sample_prior(theta_key, 1)
            print(theta_true.shape)
            print(theta_true)
            theta_true = theta
        else:
            theta_true = self.sample_prior(theta_key, 1)
        y_raw = self.simulate(y_key, theta_true, summarise=False)
        y_raw = self.misspecify(y_raw) if misspecified else y_raw
        y = self.summarise(y_raw)
        return theta_true[0, :], y[0, :], y_raw

    def misspecify(self, x):
        x = np.array(x)
        x = x.copy()
        sat_idx, sun_idx, mon_idx = [range(i, 365, 7) for i in range(1, 4)]
        sat_new = x[:, sat_idx] * self.misspecify_multiplier
        sun_new = x[:, sun_idx] * self.misspecify_multiplier
        missed_cases = (x[:, sat_idx] - sat_new) + (x[:, sun_idx] - sun_new)
        mon_new = x[:, mon_idx] + missed_cases
        for idx, new in zip([sat_idx, sun_idx, mon_idx], [sat_new, sun_new, mon_new]):
            x[:, idx] = new
        return jnp.array(x)

    def in_prior_support(self, theta):
        a = jnp.all(theta > 0, axis=1) & jnp.all(theta < 0.5, axis=1)
        b = theta[:, 0] > theta[:, 1]
        return a & b