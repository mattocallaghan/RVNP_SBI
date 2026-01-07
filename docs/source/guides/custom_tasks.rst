Adding Custom Tasks
===================

This guide explains how to add a new simulation-based inference task to RVNP-SBI.

Overview
--------

A task consists of:

1. **Simulator**: Generates data :math:`x \sim p(x|\theta)`
2. **Dataset**: Training and inference data
3. **Configuration**: Model architecture and hyperparameters

Step 1: Implement Simulator
----------------------------

Add your simulator function to ``utils.py``::

    class MyTask:
        def __init__(self, key):
            self.key = key

        def sample(self, theta):
            """
            Generate observation x given parameters theta.

            Args:
                theta: Parameters (theta_dim,)

            Returns:
                x: Observation (obs_dim,)
            """
            # Your simulator logic here
            key, subkey = jr.split(self.key)
            x = my_simulator_function(subkey, theta)
            return x

        def prior_sample(self, num_samples):
            """Sample from prior p(theta)."""
            key, subkey = jr.split(self.key)
            theta = my_prior_distribution(subkey, num_samples)
            return theta

Step 2: Add Dataset Loading
----------------------------

Add dataset generation to ``datasets.py``::

    def get_dataset(config):
        ...
        elif config.data.dataset == 'MyTask':
            # Generate training data
            simulator = MyTask(key)
            theta_train = simulator.prior_sample(num_simulations)
            x_train = vmap(simulator.sample)(theta_train)

            return {
                'theta': theta_train,
                'x': x_train,
                ...
            }

Step 3: Create Configuration
-----------------------------

Create ``configs/MyTask/ranpt_100_mu_hybrid.py``::

    from configs.default_flow import get_config as base_get_config

    def get_config():
        config = base_get_config()

        # Dataset
        config.data.dataset = 'MyTask'
        config.data.inference_dataset = 'MyTask_inference'
        config.data.num_simulations = 100000
        config.data.num_iid = 100

        # Model architecture
        config.model.name = 'nlpe_rqs_posterior'
        config.model.correction_type = 'NN'
        config.model.flow_dimension = 3  # theta dimension
        config.model.cond_dim = 10  # x dimension

        # Training
        config.training.n_iters = 300
        config.optim.lr = 1e-3

        return config

High-Dimensional Observations
------------------------------

If your observations are high-dimensional (e.g., images, spectra), you need
an embedding network.

Enable embedding::

    config.model.embedding = 'IM'  # Information Maximizing
    config.model.embedding_dim = 10  # Reduced dimension
    config.data.vector_dim_inference = 1000  # Raw observation dim

The embedding network :math:`f_\omega(x)` learns to compress :math:`x \in \mathbb{R}^{1000}`
to :math:`f_\omega(x) \in \mathbb{R}^{10}` while preserving information about :math:`\theta`.

See the Spectra task for an example.

Model Misspecification
-----------------------

To test RVNP's robustness, create a misspecified simulator:

1. **Training simulator**: Simplified/incorrect model
2. **Inference simulator**: True data-generating process

Example::

    class MyTaskMisspecified:
        def sample(self, theta):
            # Simplified model (missing some physics)
            x = simplified_simulator(theta)
            return x

    class MyTaskTrue:
        def sample(self, theta):
            # True model (complete physics)
            x = true_simulator(theta)
            return x

Use ``MyTaskMisspecified`` for training, ``MyTaskTrue`` for inference.

Step 4: Test
------------

Train your task::

    python main_train_eval.py --config=configs/MyTask/ranpt_100_mu_hybrid.py --mode=train

Evaluate::

    python main_train_eval.py --config=configs/MyTask/ranpt_100_mu_hybrid.py --mode=eval

Example: Two Moons
------------------

Here's a complete minimal example for a 2D toy problem::

    # In utils.py
    class TwoMoons:
        def __init__(self, key):
            self.key = key

        def sample(self, theta):
            # Generate data from two moons distribution
            angle = theta[0]
            radius = theta[1]
            x = jnp.array([
                radius * jnp.cos(angle),
                radius * jnp.sin(angle) + 0.5 * jnp.sign(jnp.cos(angle))
            ])
            return x + 0.1 * jr.normal(self.key, (2,))

        def prior_sample(self, num_samples):
            theta = jr.uniform(self.key, (num_samples, 2), minval=0, maxval=2*jnp.pi)
            return theta

This generates crescent-shaped observations based on angle and radius parameters.

For more examples, see the existing tasks: CS, SIR, Pendulum, Spectra in ``utils.py`` and ``datasets.py``.
