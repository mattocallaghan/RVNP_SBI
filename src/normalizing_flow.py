import jax.numpy as jnp
import abc
import jax.random as jr
from flowjax.distributions import Normal
import equinox as eqx
import optax
import functools
from tqdm import tqdm
from .losses import MaximumLikelihoodLoss,RVNPLoss
from tensorflow.summary import create_file_writer
from .utils import get_optimizer
import os
import jax
import pickle
import numpy as np
import tensorflow as tf
from .activation_functions import get_activation
from flowjax.flows import masked_autoregressive_flow,block_neural_autoregressive_flow,coupling_flow
from flowjax.bijections import RationalQuadraticSpline
import paramax
from .models.priors import get_prior_from_config
from .model_utils import (
    get_ranpt_model_breakdown,
    print_parameter_breakdown,
    save_parameter_breakdown,
    count_parameters,
    create_correction_model,
    create_embedding_models
)



# Constants
_FLOWS = {}

def register_flow(cls=None, *, name=None):
    """A decorator for registering flow classes."""
    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _FLOWS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _FLOWS[local_name] = cls
        return cls
    return _register(cls) if cls is not None else _register

def get_flow(name):
    """Retrieve a registered flow class by name."""
    return _FLOWS[name]

class Normalizing_Flow(abc.ABC):
    """Base class for normalizing flow models."""
    def __init__(self, config, *args, **kwargs):
        
        self.config = config
        self.train_bool = config.training.train
        self.optimizer = get_optimizer(config)
        #self.loss_function = G23_Loss()
        self.loss_function=MaximumLikelihoodLoss()
        self.mean = None
        self.std = None
        self.train_data = None
        self.eval_data = None
        self.prior=None



@register_flow(name='nlpe_rqs_posterior')
class Rational_Quadratic_Spline_w_posterior(Normalizing_Flow):
    """Rational Quadratic Spline Flow with Posterior Misspecification Correction.
    Can work with or without embeddings depending on the dataset:
    - With embeddings: spectra, pendulum (high-dimensional data)
    - Without embeddings: other tasks (low-dimensional data)
    """
    def __init__(self, config,key, data=None, eval_data=None, mean=None, std=None, experiment_idx=None):
        super().__init__(config)
        self.key, self.subkey = jr.split(key)
        self.nn_depth = config.model.nn_depth_bnaf
        self.nn_block_dim = config.model.nn_block_dim
        self.cond_dim = config.model.cond_dim
        self.activation = get_activation(config)
        # Base naming for shared components (embedding, decoder, discriminator, embedding_stats)
        base_shared = f"{config.model.name}_{config.data.dataset}_{config.data.num_simulations}"
        
        # Create separate naming for simulator flow (with wellspec suffix if needed)
        simulator_base_shared = base_shared
        if getattr(config.model, 'wellspec', False):
            simulator_base_shared += "_wellspec"
        
        # Test-specific suffix for posterior and correction models only
        test_suffix = ""
        if hasattr(config.data, 'num_tests'):
            shrinkage_val = getattr(config.model, 'lambda_shrinkage', 0.0)
            test_suffix = f"_tests{config.data.num_tests}_{config.model.correction_type}_shrink{shrinkage_val}"
            
            # Add wellspec suffix for well-specified evaluation models
            if getattr(config.model, 'wellspec', False):
                test_suffix += "_wellspec"
            
            # Add experiment index for multiple experiment scenarios
            if experiment_idx is not None:
                test_suffix += f"_exp{experiment_idx}"
        
        # Create dataset-specific weights folder
        weights_folder = os.path.join(config.data.data_path, f"{config.data.dataset}_weights")
        if not os.path.exists(weights_folder):
            os.makedirs(weights_folder)
            print(f"Created weights folder: {weights_folder}")
        
        # Shared files (no test suffix) - same across all test configurations, saved in weights folder
        self.file_name_embedding = os.path.join(weights_folder, f"embedding_{base_shared}.eqx")
        self.file_name_decoder = os.path.join(weights_folder, f"decoder_{base_shared}.eqx")
        self.file_name_discriminator = os.path.join(weights_folder, f"discriminator_{base_shared}.eqx")
        self.file_name_embedding_stats = os.path.join(weights_folder, f"embedding_stats_{base_shared}.npy")
        self.file_name_simulator = os.path.join(weights_folder, f"simulator_{simulator_base_shared}.eqx")
        
        # Test-specific files (with test suffix) - unique per test configuration, saved in weights folder
        self.file_name = os.path.join(weights_folder, f"{base_shared}{test_suffix}.eqx")  # posterior
        self.file_name_correction = os.path.join(weights_folder, f"correction_{base_shared}{test_suffix}.eqx")


        ### flow dimension is the dimension of theta, dimension of the posterior flow
        self.flow_dimension=config.model.flow_dimension 
        
        # Check if we should use embeddings (for high-dimensional data)
        self.use_embeddings = self.config.data.dataset in ['spectra', 'pendulum']
        
        if self.use_embeddings:
            self.embedding_dim = config.model.embedding_dim
            # Flow condition dimension is the full embedding dimension (no slicing)
            flow_cond_dim = self.embedding_dim
        else:
            # Flow condition dimension is the original data dimension
            flow_cond_dim = config.data.vector_dim  # Raw data dimension
        self.flow_cond_dim=flow_cond_dim
        self.key,subkey=jr.split(self.key)
        
        # Initialize theta bounds (will be computed from training data)
        self.theta_min = None
        self.theta_max = None
        
        ###############################
        ############# Poterior Flow
        ###############################
        self.flow = masked_autoregressive_flow(
            key=subkey,
            cond_dim=flow_cond_dim,  # Conditional on embeddings or raw data
            base_dist=Normal(jnp.zeros(self.flow_dimension), jnp.ones(self.flow_dimension)),
            transformer=RationalQuadraticSpline(knots=10, interval=5),
            flow_layers=self.nn_depth,
            nn_width=self.nn_block_dim,
            invert=False
        )
        ###############################
        ############# Embedding (only if needed)
        ###############################
        self.key, subkey = jr.split(self.key)
        if self.use_embeddings:
            self.embedding, self.discriminator, self.decoder, self.embedding_loss_function = create_embedding_models(
                key=subkey,
                config=self.config,
                embedding_dim=self.embedding_dim,
                cond_dim=self.cond_dim
            )
            self.embedding_type = self.config.model.embedding
        else:
            # No embeddings for low-dimensional data
            self.embedding = None
            self.discriminator = None
            self.decoder = None
            self.embedding_loss_function = None

        ###############################
        ############# Correction Model
        ###############################
        self.key, subkey = jr.split(self.key)
        correction_dim = self.embedding_dim if self.use_embeddings else config.data.vector_dim
        self.correction_model = create_correction_model(
            key=subkey,
            config=self.config,
            correction_dim=correction_dim,
            flow_dimension=self.flow_dimension
        )
        
        
        ###############################
        ############# Simulator Flow p(x|θ)
        ###############################
        self.key, subkey = jr.split(self.key)
        self.simulator_flow = masked_autoregressive_flow(
            key=subkey,
            cond_dim=self.flow_dimension,  # Conditional on θ parameters
            base_dist=Normal(jnp.zeros(flow_cond_dim), jnp.ones(flow_cond_dim)),  # Output x dimension
            transformer=RationalQuadraticSpline(knots=15, interval=10), #the interval is set because of the bounds of the data, maybe later could be worth scaling to 0,1 or changing architecture
            flow_layers=self.nn_depth-1,
            nn_width=self.nn_block_dim,
            invert=False #prioritise sampling
        )
        
        self.optimizer_flow = get_optimizer(self.config)

    ############### initialize the class
    def build(self, train_data=None, eval_data=None,inference_data=None, mean=None, std=None):
        self.key,subkey=jr.split(self.key)
        self.train_data = train_data
        self.eval_data = eval_data
        self.inference_data = inference_data
        assert(mean is not None and std is not None), "Mean and std must be provided for training"
        self.mean, self.std = mean, std
        self.inference_data = (inference_data-self.mean) / self.std #infernece data is saved unnormalized
        

        # Compute theta bounds from training data for clipping
        all_theta = jnp.array(train_data)[..., :self.flow_dimension].reshape(-1, self.flow_dimension)
        self.theta_min = jnp.min(all_theta, axis=0)
        self.theta_max = jnp.max(all_theta, axis=0)

        # Compute empirical bias between real observations and training simulations
        train_x = jnp.array(train_data)[..., self.flow_dimension:]  # x_sim
        inference_x = self.inference_data[..., self.flow_dimension:]  # x_obs
        self.empirical_bias = jnp.mean(inference_x, axis=0) - jnp.mean(train_x.reshape(-1,train_x.shape[-1]), axis=0)

        self.prior=get_prior_from_config(self.config, self.subkey, mean=self.mean, std=self.std, 
                                         theta_min=self.theta_min-2.8, theta_max=self.theta_max+2.8)
        self.key,subkey=jr.split(self.key)

        if(self.config.sampling.inference==True):
            assert mean is not None and std is not None, "Mean and std should not be provided for inference"

        # Check if training mode is enabled
        if self.train_bool:
            assert self.train_data is not None, "Data must be provided for training"
            if os.path.exists(self.file_name):
                print(f"Loading existing model from {self.file_name}")
                try:
                    self.flow = eqx.tree_deserialise_leaves(self.file_name, self.flow)
                    print("Successfully loaded existing flow model")
                except Exception as e:            
                    raise RuntimeError(f"Model shape mismatch detected: {e}")

            if(os.path.exists(self.file_name_embedding)):
                self.embedding = eqx.tree_deserialise_leaves(self.file_name_embedding, self.embedding)
                print(f"Loading existing model from {self.file_name_embedding}")
            if(os.path.exists(self.file_name_decoder)):
                self.decoder = eqx.tree_deserialise_leaves(self.file_name_decoder, self.decoder)
                print(f"Loading existing model from {self.file_name_decoder}")
            if(os.path.exists(self.file_name_discriminator)):
                self.discriminator = eqx.tree_deserialise_leaves(self.file_name_discriminator, self.discriminator)
                print(f"Loading existing model from {self.file_name_discriminator}")
            if(os.path.exists(self.file_name_correction)):
                self.correction_model = eqx.tree_deserialise_leaves(self.file_name_correction, self.correction_model)
                print(f"Loading existing model from {self.file_name_correction}")
            if(os.path.exists(self.file_name_simulator)):
                self.simulator_flow = eqx.tree_deserialise_leaves(self.file_name_simulator, self.simulator_flow)
                print(f"Loading existing model from {self.file_name_simulator}")

            print(f"Training model...")
            # Always use staged training approach
            assert self.train_data is not None, "Data must be provided for training"
            self.key,subkey=jr.split(self.key)
            self.key=self.train(subkey,train_data=self.train_data, inference_data=self.inference_data)
            
            # Save embedding models if used
            if self.use_embeddings:
                eqx.tree_serialise_leaves(self.file_name_embedding, self.embedding)
                eqx.tree_serialise_leaves(self.file_name_decoder, self.decoder)
                eqx.tree_serialise_leaves(self.file_name_discriminator, self.discriminator)
            eqx.tree_serialise_leaves(self.file_name, self.flow)
            eqx.tree_serialise_leaves(self.file_name_correction, self.correction_model)
            eqx.tree_serialise_leaves(self.file_name_simulator, self.simulator_flow)


        else:
            try:
                self.flow = eqx.tree_deserialise_leaves(self.file_name, self.flow)
                if self.use_embeddings:
                    self.embedding = eqx.tree_deserialise_leaves(self.file_name_embedding, self.embedding)
                    self.decoder=eqx.tree_deserialise_leaves(self.file_name_decoder, self.decoder)
                    self.discriminator=eqx.tree_deserialise_leaves(self.file_name_discriminator, self.discriminator)
                self.correction_model=eqx.tree_deserialise_leaves(self.file_name_correction, self.correction_model)
                self.simulator_flow=eqx.tree_deserialise_leaves(self.file_name_simulator, self.simulator_flow)
                
            except FileNotFoundError:
                print(f"Model file not found: {self.file_name}. Training new model...")
                assert self.train_data is not None, "Data must be provided for training"

                self.key,subkey=jr.split(self.key)
                self.key=self.train(subkey,train_data=self.train_data, inference_data=self.inference_data)
                eqx.tree_serialise_leaves(self.file_name, self.flow)
                eqx.tree_serialise_leaves(self.file_name_correction, self.correction_model)
                eqx.tree_serialise_leaves(self.file_name_simulator, self.simulator_flow)
                if self.use_embeddings:
                    eqx.tree_serialise_leaves(self.file_name_embedding, self.embedding)
                    eqx.tree_serialise_leaves(self.file_name_decoder, self.decoder)
                    eqx.tree_serialise_leaves(self.file_name_discriminator, self.discriminator)

    def train(self,key, train_data=None,inference_data=None):
        """Train the model using staged approach."""

        assert inference_data is not None, "Inference Data must be provided for training"
        print("Starting staged training...")
        key, subkey = jr.split(key)
        key = self.train_staged(subkey, train_data, inference_data)
        return key
    
    def train_staged(self, key, train_data, inference_data):
        """Train RVNP model using multi-stage approach.

        Implements the RVNP training algorithm across multiple stages. Each stage focuses
        on different model components to ensure stable and effective learning.

        Training Stages:

            **Stage 1**: Embedding networks (optional, for high-dimensional data)
                - **Data used**: train_data (pre-generated simulations)
                - **Trains**: f_ω(x), discriminator, decoder
                - **Method**: InfoMax (mutual information maximization)
                - **Config**: config.model.train_embeddings

            **Stage 2**: Simulator flow p(x|θ)
                - **Data used**: train_data (pre-generated (θ, x) pairs)
                - **Trains**: Simulator flow p(x|θ)
                - **Method**: Maximum likelihood
                - **Config**: config.model.train_simulator

            **Stage 3**: Joint posterior + correction
                - **Data used**: ONLY inference_data (observed x_obs)
                - **Trains**: Posterior q_φ(θ|x̂) and correction r_ψ(x̂|x,θ)
                - **Method**: RVNP Loss (samples θ ~ q_φ(θ|x_obs) and x_sim ~ p(x|θ) on-the-fly)
                - **NO pre-generated training data used in this stage**

        Args:
            key: JAX random key for reproducibility
            train_data: Pre-generated simulations (θ, x) ~ p(θ)p(x|θ)
                - **Used in**: Stages 1 and 2 only
                - **NOT used in**: Stage 3 (joint training)
            inference_data: Observed data x_obs
                - **Used in**: Stage 3 (joint training)

        Returns:
            Updated JAX random key

        Notes:
            - Stages 1-2 use pre-generated train_data to learn simulator and embeddings
            - Stage 3 uses ONLY inference_data and samples fresh (θ, x_sim) each iteration
            - Config flags: train_embeddings (Stage 1), train_simulator (Stage 2)
            - Epochs: warmup_epochs (Stages 1-2), final_epochs (Stage 3)

        Example:
            >>> key = jax.random.PRNGKey(0)
            >>> model = RANPT(config)
            >>> key = model.train_staged(key, train_data, inference_data)
        """
        print("Starting 3-stage training system...")
        # Stage 1: Embedding (handled separately in train() method)
        if getattr(self.config.model, 'train_embeddings', True):
            print("Stage 1: Training embedding networks...")
            key, subkey = jr.split(key)
            key=self.train_embedding_networks(subkey,train_data,inference_data)
        else:
            print("Stage 1: Skipping embedding training (train_embeddings=False)")
        
        # Stage 2: Train simulator flow p(x|θ) if enabled

        print("Stage 2: Training simulator flow...")
        key, subkey = jr.split(key)
        key, self.simulator_flow = self.train_simulator(subkey, self.simulator_flow, train_data)

        # Stage 3: Joint training
        print("Stage 3: Joint training")
        key, subkey = jr.split(key)
        key,self.flow,self.correction_model = self.train_single_stage(subkey, 'joint',self.flow,self.correction_model, train_data, inference_data)

        print("Training system completed.")
        return key
    
    
    def train_single_stage(self, key, stage_name, flow, correction_model, train_data, inference_data, max_epochs=None):
        """Train posterior and correction jointly (Stage 3).

        This is a wrapper around train_posterior() that implements Stage 3 of RVNP.

        Args:
            key: JAX random key
            stage_name: Must be 'joint' (only joint training supported)
            flow: Posterior flow q_φ(θ|x̂) to train
            correction_model: Correction model r_ψ(x̂|x,θ) to train
            train_data: NOT USED IN STAGE 3 (kept for API compatibility with Stages 1-2)
            inference_data: Observed data x_obs for training
            max_epochs: Max training epochs (uses config.training.final_epochs if None)

        Returns:
            Tuple of (key, trained_flow, trained_correction_model)

        Notes:
            - Stage 3 uses ONLY inference_data (observed x_obs)
            - train_data parameter is ignored (not used in this stage)
            - Samples θ ~ q_φ(θ|x_obs) and x_sim ~ p(x|θ) on-the-fly

        Raises:
            ValueError: If stage_name is not 'joint'
        """
        max_epochs = getattr(self.config.training, 'final_epochs', self.config.training.n_iters)
        train_data_prepared = train_data
        eval_data_prepared = self.eval_data


        if stage_name != "joint":
            raise ValueError(f"Unknown stage name: {stage_name}. Only 'joint' training is supported.")

        # Joint training using RVNPLoss
        print("Training using RVNPLoss (joint training)...")
        print(f"Using only observed data (no pre-generated training data)")
        key, subkey = jr.split(key)
        flow, correction_model, stage_losses = self.train_posterior(
            key=subkey,
            dist=flow,
            correction_model=correction_model,
            embedding_model=self.embedding,
            inference_data=inference_data,
            learning_rate=self.config.optim.lr,
            max_epochs=max_epochs,
            max_patience=self.config.training.max_patience
        )

        if not hasattr(self, 'losses') or self.losses is None:
            self.losses = stage_losses
        key, subkey = jr.split(key)
        return key,flow, correction_model

    def train_simulator(self, key, simulator_flow, train_data):
        """Train the simulator emulator p(x|θ) using maximum likelihood."""
        # Add config check BEFORE file existence check
        if not getattr(self.config.model, 'train_simulator', True):
            print("Simulator training disabled by config (train_simulator=False).")
            return key, simulator_flow
        print("Training simulator emulator p(x|θ) from scratch...")
        # Initialize simulator parameters
        params_simulator, static_simulator = eqx.partition(
            simulator_flow, 
            eqx.is_inexact_array,
            is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
        )
        if(self.embedding is not None):
            embedding = paramax.unwrap(self.embedding)
            p_embedding = lambda x, k: embedding(x, key=k, inference=True)
            
            # Load embedding stats for normalization
            with open(self.file_name_embedding_stats, 'rb') as f:
                embedding_stats = pickle.load(f)
        
        optimizer_simulator = get_optimizer(self.config)
        opt_state_simulator = optimizer_simulator.init(params_simulator)
        
        # Use MaximumLikelihoodLoss for simulator training
        # Apply hard clipping for spectra and pendulum tasks to prevent collapse
        # But remove bounds for wellspec cases
        if getattr(self.config.model, 'wellspec', False):
            loss_fn = MaximumLikelihoodLoss()
            print(f"Wellspec mode: No log probability bound applied for {self.config.data.dataset} task")
        elif self.config.data.dataset in ['spectra', 'pendulum']:
            loss_fn = MaximumLikelihoodLoss(min_loss_bound=-11.0)
            print(f"Applied hard clipping at -11.0 for {self.config.data.dataset} task")
        else:
            loss_fn = MaximumLikelihoodLoss()
        
        # Training state
        best_params_simulator = params_simulator
        best_loss = float('inf')
        patience_counter = 0
        max_patience = getattr(self.config.training, 'max_patience', 50)
        
        losses = {"train": [], "val": []}
        
        # Training loop
        simulator_epochs =getattr(self.config.training, 'simulator_epochs', 200)  # Configurable, default reduced from 200
        loop_epoch = tqdm(range(simulator_epochs), desc="Simulator Training")
        

        
        @eqx.filter_jit
        def step(params_simulator, z_target,theta, opt_state, key_step):
            """Perform a single training step with RVNPLoss."""
 
            
            loss, grads = eqx.filter_value_and_grad(loss_fn)(
                params_simulator, static_simulator, z_target, theta, None
            )
            
            # Update parameters
            updates, opt_state = optimizer_simulator.update(grads, opt_state, params_simulator)
            params_simulator = eqx.apply_updates(params_simulator, updates)
            
            return params_simulator, opt_state, loss



        for epoch in loop_epoch:
            train_losses = []
            key, epoch_key = jr.split(key)
            
            # Training phase
            for batch_data in train_data:
                batch = jax.tree_util.tree_map(lambda x: jnp.array(x), batch_data)
                
                # Extract θ (condition) and x (target) for simulator training
                theta = batch[..., :self.config.model.flow_dimension]  # Parameters (condition)
                x_data = batch[..., self.config.model.flow_dimension:]  # Data (target)

                if(self.embedding is not None):
                    epoch_key, batch_key = jr.split(epoch_key)
                    embed_key, _ = jr.split(batch_key)
                    keys_batch = jax.random.split(embed_key, x_data.shape[0])
                    x_embedded = jax.lax.stop_gradient(
                        jax.vmap(p_embedding)(x_data[:, jnp.newaxis, :], keys_batch)
                    )
                    z_target = jax.lax.stop_gradient(
                        (x_embedded - embedding_stats['mean']) / embedding_stats['std']
                    )
                else:
                    z_target = x_data
                
                # Training step: p(x|θ) using MaximumLikelihoodLoss
                params_simulator, opt_state_simulator, loss = step(
                    params_simulator, z_target, theta, opt_state_simulator, epoch_key
                )
                train_losses.append(loss)
            
            # Compute average training loss
            avg_train_loss = sum(train_losses) / len(train_losses)
            losses["train"].append(avg_train_loss)
            
            # Validation phase
            val_losses = []
            for val_batch_data in self.eval_data:
                val_batch = jax.tree_util.tree_map(lambda x: jnp.array(x), val_batch_data)
                
                # Extract θ (condition) and x (target) for validation
                theta_val = val_batch[..., :self.config.model.flow_dimension]
                x_data_val = val_batch[..., self.config.model.flow_dimension:]

                if(self.embedding is not None):
                    epoch_key, batch_key = jr.split(epoch_key)
                    embed_key, _ = jr.split(batch_key)
                    keys_batch = jax.random.split(embed_key, x_data_val.shape[0])
                    x_embedded = jax.lax.stop_gradient(
                        jax.vmap(p_embedding)(x_data_val[:, jnp.newaxis, :], keys_batch)
                    )
                    
                    # Normalize embeddings with stop_gradient
                    z_target = jax.lax.stop_gradient(
                        (x_embedded - embedding_stats['mean']) / embedding_stats['std']
                    )
                    
                    # Apply embedding indices if specified
                    if hasattr(self, 'embedding_indices') and self.embedding_indices is not None:
                        z_target = z_target[..., self.embedding_indices]
                else:
                    z_target = x_data_val
                
                # Validation step
                val_loss = loss_fn(params_simulator, static_simulator, z_target, theta_val, None)
                val_losses.append(val_loss)
            
            # Compute average validation loss
            avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float('inf')
            losses["val"].append(avg_val_loss)
            
            # Early stopping based on validation loss
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_params_simulator = params_simulator
                patience_counter = 0
            else:
                patience_counter += 1
                
            if epoch % 10 == 0:
                print(f"Simulator Epoch {epoch}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
                
            if patience_counter >= max_patience:
                print("Early stopping for simulator training!")
                break
        
        # Create updated simulator model
        simulator_flow = eqx.combine(best_params_simulator, static_simulator)
        
        # Save simulator model
        # temporary code for testing, save it now but also saved later
        eqx.tree_serialise_leaves(self.file_name_simulator, simulator_flow)
        print(f"Simulator model saved to {self.file_name_simulator}")        
        return key, simulator_flow



    def train_posterior(self, key, dist, correction_model, embedding_model, inference_data, max_epochs, max_patience, learning_rate,
                       return_best=True, show_progress=True):
        """Train posterior q_φ(θ|x̂) and correction r_ψ(x̂|x,θ) jointly using RVNP Loss.

        This is Stage 3 of RVNP training. Uses ONLY observed data (x_obs) - no pre-generated
        simulations. ALL sampling happens inside the loss function (_kl_divergence):

        Training Loop:
        1. Pass x_obs to RVNPLoss
        2. Inside _kl_divergence:
           a. Sample θ ~ q_φ(θ|x_obs) from current posterior
           b. Sample x_sim ~ p(x|θ) from trained simulator (Stage 2)
           c. Compute IW-ELBO with corrected observations x̂ ~ r_ψ(x̂|x_sim, θ)
           d. Compute shrinkage prior: λ * E_θ[||μ_θ(θ)||²] using sampled θ
           e. Return -ELBO + shrinkage
        3. Update both posterior φ and correction ψ via gradient descent

        Args:
            key: JAX random key
            dist: Current posterior flow q_φ(θ|x̂) to train
            correction_model: Current correction model r_ψ(x̂|x,θ) to train
            embedding_model: Pre-trained embedding f_ω(x) (optional, for high-D data)
            inference_data: Observed data x_obs, shape (n_obs, obs_dim)
            max_epochs: Maximum training epochs
            max_patience: Early stopping patience
            learning_rate: Learning rate for optimizer
            return_best: Return best params based on validation loss (default: True)
            show_progress: Show progress bar (default: True)

        Returns:
            Tuple of (trained_flow, trained_correction_model, losses_dict)

        Notes:
            - No pre-generated (θ, x) pairs used - all sampling done inside loss function
            - No sampling in training loop - only x_obs passed to loss
            - Validation is stochastic (uses different random seeds on same x_obs)
            - Memory efficient: no need to store massive training datasets
            - Theoretically sound: θ sampled from q_φ(θ|x_obs) relevant to observed data
        """
        workspace_dir = str(self.config.training.workspace)+'/Tensorboard'
        if not os.path.exists(workspace_dir):
            os.makedirs(workspace_dir)
        summary_writer = create_file_writer(workspace_dir)
        optimizer = get_optimizer(self.config)

        # Initialize RVNPLoss with config parameters
        loss_fn = RVNPLoss(
            lambda_variational=self.config.model.lambda_variational,
            lambda_kl=getattr(self.config.model, 'lambda_kl', 1.0),
            lambda_shrinkage=self.config.model.lambda_shrinkage,
            lambda_entropy=self.config.model.lambda_entropy,
            simulator_samples_per_theta=getattr(self.config.model, 'simulator_samples_per_theta', 100),
            n_sim_samples_per_theta=getattr(self.config.model, 'n_sim_samples_per_theta', 32),
            prior=self.prior,
            empirical_bias=self.empirical_bias,
        )

        # Partition parameters for both flow and correction model
        params_flow, static_flow = eqx.partition(
            dist,
            eqx.is_inexact_array,
            is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
        )
        
        params_correction, static_correction = eqx.partition(
            correction_model,
            eqx.is_inexact_array,
            is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
        )

        if embedding_model is not None:
            with open(self.file_name_embedding_stats, 'rb') as f:
                embedding_stats = pickle.load(f)
            assert embedding_stats is not None, "Embedding stats must be provided for training"
            params_embedding, static_embedding = eqx.partition(
                embedding_model,
                eqx.is_inexact_array,
                is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
            )
        else:
            params_embedding, static_embedding, embedding_stats = None, None, None

        # Prepare observed data (inference_data contains only x_obs, no theta)
        # Shape: (n_obs, obs_dim)
        inference_batch = inference_data.reshape(-1, inference_data.shape[-1])
        # If inference_data contains both theta and x, extract only x
        if inference_batch.shape[-1] > self.config.data.vector_dim:
            x_obs_inference = inference_batch[..., self.flow_dimension:]
        else:
            x_obs_inference = inference_batch

        # Batch observed data
        batch_size_obs = getattr(self.config.training, 'batch_size_obs', 200)
        x_obs_batches = []
        for i in range(0, x_obs_inference.shape[0], batch_size_obs):
            x_obs_batches.append(x_obs_inference[i:i+batch_size_obs])

        @eqx.filter_jit
        def step(params_flow, params_correction, x_obs, opt_state, key_step):
            """Perform a single training step with RVNPLoss.

            All sampling (θ and x_sim) happens inside the loss function.
            """
            # Compute loss and gradients
            def loss_wrapper(pf, pc):
                return loss_fn(
                    params_flow=pf,
                    static_flow=static_flow,
                    params_embedding=params_embedding,
                    static_embedding=static_embedding,
                    params_correction=pc,
                    static_correction=static_correction,
                    simulator_flow=self.simulator_flow,
                    x_obs=x_obs,
                    key=key_step,
                    embedding_stats=embedding_stats
                )

            def combined_loss_wrapper(combined_params):
                pf, pc = combined_params
                return loss_wrapper(pf, pc)

            loss_val, combined_grads = eqx.filter_value_and_grad(combined_loss_wrapper)((params_flow, params_correction))
            flow_grads, correction_grads = combined_grads

            # Joint training: update both posterior and correction
            combined_params = (params_flow, params_correction)
            combined_grads = (flow_grads, correction_grads)
            updates, opt_state = optimizer.update(combined_grads, opt_state, combined_params)
            params_flow, params_correction = eqx.apply_updates(combined_params, updates)

            return params_flow, params_correction, opt_state, loss_val

        best_params_flow = params_flow
        best_params_correction = params_correction
        # Initialize optimizer state for joint training
        opt_state = optimizer.init((params_flow, params_correction))

        key, subkey = jr.split(key)
        losses = {"train": [], "val": []}

        best_val_loss = float('inf')
        patience_counter = 0

        loop_epoch = tqdm(range(max_epochs), disable=not show_progress, desc="RVNP Training")
        for epoch in loop_epoch:
            train_losses = []

            # Iterate over batches of observed data only
            for x_obs_batch in x_obs_batches:
                key, subkey = jr.split(key)

                params_flow, params_correction, opt_state, train_loss = step(
                    params_flow, params_correction, x_obs_batch, opt_state, subkey)
                train_losses.append(train_loss)

            # Compute average training loss
            avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else float('inf')
            losses["train"].append(avg_train_loss)
            
            # Monitor correction model covariance matrix and loss components
            if epoch % 20 == 0:  # Print every 20 epochs to reduce clutter
                if hasattr(self, 'correction_model') and self.correction_model is not None:
                    correction_combined = eqx.combine(params_correction, static_correction)
                    from .models.correction_model import HybridCorrectionModel, FullNeuralCorrectionModel, MuHybridCorrectionModel, GlobalCorrectionModel
                    if isinstance(correction_combined, GlobalCorrectionModel):
                        # GlobalCorrectionModel - no theta dependence
                        cov_matrix = correction_combined.get_covariance_matrix()
                        print(f"  Epoch {epoch}: Global covariance matrix =")
                        print(f"    {cov_matrix}")
                        print(f"  Global mean shift: {correction_combined.mu_global}")
                    elif isinstance(correction_combined, (HybridCorrectionModel, MuHybridCorrectionModel, FullNeuralCorrectionModel)):
                        # Create dummy theta for covariance matrix evaluation
                        dummy_theta = jnp.zeros(self.flow_dimension)
                        cov_matrix = correction_combined.get_covariance_matrix(dummy_theta)
                        if isinstance(correction_combined, HybridCorrectionModel):
                            model_type = "Hybrid"
                        elif isinstance(correction_combined, MuHybridCorrectionModel):
                            model_type = "Mu-Hybrid"
                        else:
                            model_type = "Full Neural"
                        print(f"  Epoch {epoch}: {model_type} covariance matrix (at theta=0) =")
                        print(f"    {cov_matrix}")
                    elif hasattr(correction_combined, 'get_covariance_matrix'):
                        cov_matrix = correction_combined.get_covariance_matrix()
                        print(f"  Epoch {epoch}: Covariance matrix =")
                        print(f"    {cov_matrix}")
                    print(f"  Epoch {epoch}: Loss = {avg_train_loss:.6f}")
                        
                # Monitor individual loss components on a sample batch

            # Compute validation loss on observed data (stochastic due to posterior sampling)
            val_losses = []
            for x_obs_val_batch in x_obs_batches:
                key, subkey = jr.split(key)
                val_loss = loss_fn(
                    params_flow=params_flow,
                    static_flow=static_flow,
                    params_embedding=params_embedding,
                    static_embedding=static_embedding,
                    params_correction=params_correction,
                    static_correction=static_correction,
                    simulator_flow=self.simulator_flow,
                    x_obs=x_obs_val_batch,
                    key=subkey,
                    embedding_stats=embedding_stats
                )
                val_losses.append(val_loss)

            avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float('inf')
            losses["val"].append(avg_val_loss)

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_params_flow = params_flow
                best_params_correction = params_correction
                patience_counter = 0
            else:
                patience_counter += 1
                
            if epoch % 20 == 0:
                print(f"RVNP Loss Epoch {epoch}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
                
                # Also log the loss components computed above if they exist
                # (This will show the breakdown from the sample batch computed earlier)
                
            if patience_counter >= max_patience:
                print("Early stopping for simplified loss training!")
                break
        
        # Update correction model with trained parameters
        correction_model = eqx.combine(best_params_correction if return_best else params_correction, static_correction)
        
        # Return best flow model
        dist = eqx.combine(best_params_flow if return_best else params_flow, static_flow)
        summary_writer.close()
        return dist, correction_model, losses

    def train_embedding_networks(self,key,train_data,inference_data):
        """
        Stage 1: Train embedding networks (encoder/decoder) if needed.
        This implements dedicated embedding training before posterior training.
        """
        if not self.use_embeddings:
            print("No embeddings needed for this dataset, skipping embedding training.")
            return key
            
        # Add config check BEFORE file existence check
        if not getattr(self.config.model, 'train_embeddings', True):
            print("Embedding training disabled by config (train_embeddings=False).")
            return key

        # Get embedding training parameters
        embedding_epochs = getattr(self.config.training, 'embedding_epochs', 200)
        max_patience = getattr(self.config.training, 'max_patience', 50)
        
        # Set up optimizers and loss function
        optimizer_embedding = self.optimizer
        embedding_loss_fn = self.embedding_loss_function
        
        # Partition embedding, discriminator, and decoder parameters
        params_embedding, static_embedding = eqx.partition(
            self.embedding,
            eqx.is_inexact_array,
            is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
        )
        params_discriminator, static_discriminator = eqx.partition(
            self.discriminator,
            eqx.is_inexact_array,
            is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
        )
        params_decoder, static_decoder = eqx.partition(
            self.decoder,
            eqx.is_inexact_array,
            is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
        )
        
        params_embedding_components = (params_embedding, params_discriminator, params_decoder)
        static_embedding_components = (static_embedding, static_discriminator, static_decoder)
        
        # Initialize optimizer
        opt_state_embedding = optimizer_embedding.init(params_embedding_components)
        
        # Set up inference data for embedding training
        inference_batch = inference_data.reshape(-1, inference_data.shape[-1])
        # Extract only the high-dimensional part (x) from inference data, not the theta part
        inference_x_only = inference_batch[..., self.config.model.flow_dimension:]
        
        # Training step function
        @eqx.filter_jit
        def embedding_step(_params, batch, inference_x, _opt_state, key):
            loss_val, grads = eqx.filter_value_and_grad(embedding_loss_fn)(
                _params, static_embedding_components, 
                batch[..., self.config.model.flow_dimension:],  # x (high-dim data)
                inference_x,  # x_real (real observations, high-dim only)
                batch[..., :self.config.model.flow_dimension],  # condition (theta)
                key
            )
            updates, _opt_state = optimizer_embedding.update(grads, _opt_state, _params)
            _params = eqx.apply_updates(_params, updates)
            return _params, _opt_state, loss_val
        
        # Training loop
        best_params_embedding = params_embedding_components
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Training embedding networks for {embedding_epochs} epochs...")
        
        for epoch in tqdm(range(embedding_epochs), desc="Embedding Training"):
            train_losses = []
            key, epoch_key = jr.split(key)  # Split self.key for this epoch
            
            # Training step
            for batch_data in train_data:
                batch = batch_data
                epoch_key, batch_key = jr.split(epoch_key)  # Use epoch_key instead of key
                params_embedding_components, opt_state_embedding, train_loss = embedding_step(
                    params_embedding_components, batch, inference_x_only, opt_state_embedding, batch_key
                )
                train_losses.append(train_loss)
            
            avg_train_loss = sum(train_losses) / len(train_losses)
            
            # Validation step  
            val_losses = []
            for val_batch_data in self.eval_data:
                val_batch = val_batch_data
                epoch_key, val_key = jr.split(epoch_key)  # Continue using epoch_key
                val_loss = embedding_loss_fn(
                    params_embedding_components, static_embedding_components,
                    val_batch[..., self.config.model.flow_dimension:],
                    inference_x_only,
                    val_batch[..., :self.config.model.flow_dimension],
                    val_key
                )
                val_losses.append(val_loss)
            
            avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float('inf')
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_params_embedding = params_embedding_components
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Update embedding, discriminator, and decoder with trained parameters
        params_emb_final, params_discriminator_final, params_decoder_final = best_params_embedding
        self.embedding = eqx.combine(params_emb_final, static_embedding)
        self.discriminator = eqx.combine(params_discriminator_final, static_discriminator)
        self.decoder = eqx.combine(params_decoder_final, static_decoder)
        
        print("Embedding networks training complete!")
        
        # Save embedding models immediately after training
        print("Saving trained embedding models...")
        eqx.tree_serialise_leaves(self.file_name_embedding, self.embedding)
        eqx.tree_serialise_leaves(self.file_name_discriminator, self.discriminator)
        eqx.tree_serialise_leaves(self.file_name_decoder, self.decoder)
        print("Embedding models saved!")
        
        print("Stage 1.1: Computing embedding statistics...")
        
        # Compute embedding statistics from trained embedding
        key, subkey = jr.split(key)  # Split self.key for this epoch

        key=self.compute_and_save_embedding_stats(subkey,train_data)
        
        print("Embedding statistics computed and saved!")
        return key

    def compute_and_save_embedding_stats(self,key,train_data):
        """Compute and save embedding statistics from trained embedding network."""
        print("Computing embedding statistics from all training data...")
        
        # Create embedding function
        embedding = paramax.unwrap(self.embedding)
        p_embedding = lambda x, k: embedding(x, key=k, inference=True)
        
        # Process all training data through embedding
        all_embeddings = []
        for batch_data in train_data:
            batch = batch_data
            key, subkey = jr.split(key)  # Update self.key
            keys = jax.random.split(subkey, batch.shape[0])
            
            # Extract high-dimensional data (x) from batch
            x_data = batch[..., self.config.model.cond_dim:]  # Skip theta/condition dimensions
            
            # Compute embeddings
            batch_embeddings = jax.lax.stop_gradient(
                jax.vmap(p_embedding)(x_data[:, jnp.newaxis, :], keys)
            )
            all_embeddings.append(batch_embeddings)
        
        # Concatenate all embeddings
        all_embeddings = jnp.concatenate(all_embeddings, axis=0)
        
        # Compute statistics
        mean_embed = all_embeddings.mean(0)
        std_embed = all_embeddings.std(0)
        
        # Create embedding stats dictionary
        embedding_stats = {
            'mean': mean_embed, 
            'std': std_embed
        }
        
        # Save embedding stats
        with open(self.file_name_embedding_stats, 'wb') as f:
            pickle.dump(embedding_stats, f)
        
        print(f"Embedding stats saved to: {self.file_name_embedding_stats}")
        print(f"   - Mean shape: {mean_embed.shape}")
        print(f"   - Std shape: {std_embed.shape}")
        print(f"   - Mean values: {mean_embed[:5]}...")  # Show first 5 values
        print(f"   - Std values: {std_embed[:5]}...")   # Show first 5 values
        return key


@register_flow(name='noisy_neural_posterior')
class NNPE_Spike_Slab(Normalizing_Flow):
    """
    Noisy Neural Posterior Estimation with Spike-and-Slab corruption.
    
    Loads pre-trained embeddings and trains posterior with corrupted data.
    """
    
    def __init__(self, config, key, data=None, eval_data=None, mean=None, std=None, experiment_idx=None):
        super().__init__(config)
        self.key, self.subkey = jr.split(key)
        
        # Model architecture parameters (copied from RQS class)
        self.nn_depth = config.model.nn_depth_bnaf
        self.nn_block_dim = config.model.nn_block_dim
        self.cond_dim = config.model.cond_dim
        self.activation = get_activation(config)
        self.flow_dimension = config.model.flow_dimension
        
        # Spike-and-slab parameters
        self.slab_scale = getattr(config.model, 'slab_scale', 1.0)
        self.spike_scale = getattr(config.model, 'spike_scale', 0.01)
        
        # File paths for loading pre-trained components
        # Use the original RQS model name for loading embeddings, not the NNPE name
        embedding_base_shared = f"nlpe_rqs_posterior_{config.data.dataset}_{config.data.num_simulations}"
        weights_folder = os.path.join(config.data.data_path, f"{config.data.dataset}_weights")
        
        self.file_name_embedding = os.path.join(weights_folder, f"embedding_{embedding_base_shared}.eqx")
        self.file_name_decoder = os.path.join(weights_folder, f"decoder_{embedding_base_shared}.eqx")
        self.file_name_discriminator = os.path.join(weights_folder, f"discriminator_{embedding_base_shared}.eqx")
        self.file_name_embedding_stats = os.path.join(weights_folder, f"embedding_stats_{embedding_base_shared}.npy")
        
        # NNPE-specific file path  
        nnpe_base_shared = f"{config.model.name}_{config.data.dataset}_{config.data.num_simulations}"
        test_suffix = ""
        if hasattr(config.data, 'num_tests'):
            shrinkage_val = getattr(config.model, 'lambda_shrinkage', 0.0)
            test_suffix = f"_tests{config.data.num_tests}_nnpe_shrink{shrinkage_val}"
            
            # Add experiment index for multiple experiment scenarios
            if experiment_idx is not None:
                test_suffix += f"_exp{experiment_idx}"
        self.file_name = os.path.join(weights_folder, f"nnpe_{nnpe_base_shared}{test_suffix}.eqx")
        
        # Check if we use embeddings
        self.use_embeddings = config.data.dataset in ['spectra', 'pendulum']
        
        if self.use_embeddings:
            self.embedding_dim = config.model.embedding_dim
            flow_cond_dim = self.embedding_dim
        else:
            flow_cond_dim = config.data.vector_dim
        
        self.flow_cond_dim = flow_cond_dim
        self.key, subkey = jr.split(self.key)
        
        # Initialize posterior flow
        self.flow = masked_autoregressive_flow(
            key=subkey,
            cond_dim=flow_cond_dim,
            base_dist=Normal(jnp.zeros(self.flow_dimension), jnp.ones(self.flow_dimension)),
            transformer=RationalQuadraticSpline(knots=10, interval=5),
            flow_layers=self.nn_depth,
            nn_width=self.nn_block_dim,
            invert=False
        )

        # Initialize embedding components using factory function
        if self.use_embeddings:
            self.key, subkey = jr.split(self.key)
            self.embedding, _, _, _ = create_embedding_models(
                key=subkey,
                config=config,
                embedding_dim=self.embedding_dim,
                cond_dim=self.flow_dimension
            )
        else:
            self.embedding = None

        self.optimizer_flow = get_optimizer(self.config)

        print(f"NNPE initialized with spike_scale={self.spike_scale}, slab_scale={self.slab_scale}")
    
    def add_spike_and_slab_error(self, key, x, slab_scale=None, spike_scale=None):
        """Add spike-and-slab corruption to data."""
        if slab_scale is None:
            slab_scale = self.slab_scale
        if spike_scale is None:
            spike_scale = self.spike_scale
            
        keys = jr.split(key, 3)
        misspecified = jr.bernoulli(keys[0], shape=x.shape)
        spike = jr.normal(keys[2], shape=x.shape) * spike_scale
        slab = jr.cauchy(keys[1], shape=x.shape) * slab_scale
        return x + misspecified * slab + (1 - misspecified) * spike
    
    def build(self, train_data=None, eval_data=None, inference_data=None, mean=None, std=None):
        """Build and train NNPE model."""
        self.key, subkey = jr.split(self.key)
        self.train_data = train_data
        self.eval_data = eval_data
        self.inference_data = inference_data
        
        assert(mean is not None and std is not None), "Mean and std must be provided"
        self.mean, self.std = mean, std
        self.inference_data = (inference_data - self.mean) / self.std
        
        # Compute theta bounds and empirical bias
        all_theta = jnp.array(train_data)[..., :self.flow_dimension].reshape(-1, self.flow_dimension)
        self.theta_min = jnp.min(all_theta, axis=0)
        self.theta_max = jnp.max(all_theta, axis=0)
        self.theta_std= jnp.std(all_theta, axis=0)
        
        train_x = jnp.array(train_data)[..., self.flow_dimension:]
        inference_x = self.inference_data[..., self.flow_dimension:]
        self.empirical_bias = jnp.mean(inference_x, axis=0) - jnp.mean(train_x.reshape(-1, train_x.shape[-1]), axis=0)
        
        self.prior = get_prior_from_config(self.config, subkey, mean=self.mean, std=self.std,
                                         theta_min=self.theta_min-2.8, theta_max=self.theta_max+2.8)
        
        if self.train_bool:
            # Load pre-trained embeddings
            if self.use_embeddings:
                print("Loading pre-trained embeddings...")
                try:
                    self.embedding = eqx.tree_deserialise_leaves(self.file_name_embedding, self.embedding)
                    print(f"Loaded embedding from {self.file_name_embedding}")
                except FileNotFoundError:
                    raise FileNotFoundError(f"Pre-trained embedding not found: {self.file_name_embedding}")
            
            # Train NNPE posterior with corrupted data
            print("Training NNPE posterior with spike-and-slab corruption...")
            self.key = self.train_nnpe_posterior(subkey, train_data, inference_data)
            
            # Save NNPE model
            eqx.tree_serialise_leaves(self.file_name, self.flow)
            print(f"NNPE model saved to {self.file_name}")
        else:
            # Load trained NNPE model
            try:
                self.flow = eqx.tree_deserialise_leaves(self.file_name, self.flow)
                if self.use_embeddings:
                    self.embedding = eqx.tree_deserialise_leaves(self.file_name_embedding, self.embedding)
                print(f"Loaded NNPE model from {self.file_name}")
            except FileNotFoundError:
                print(f"NNPE model not found: {self.file_name}. Training new model...")
                self.key = self.train_nnpe_posterior(subkey, train_data, inference_data)
                eqx.tree_serialise_leaves(self.file_name, self.flow)
    
    def train_nnpe_posterior(self, key, train_data, inference_data):
        """Train NNPE posterior with spike-and-slab corrupted data."""
        print("Training NNPE posterior with corrupted data...")
        
        # Apply corruption to training data
        corrupted_train_data = []
        for batch_data in train_data:
            key, batch_key = jr.split(key)
            batch = jnp.array(batch_data) if not isinstance(batch_data, jnp.ndarray) else batch_data
            
            # Extract theta and x components
            theta = batch[..., :self.flow_dimension]
            x_data = batch[..., self.flow_dimension:]
            
            # Apply corruption to x_data only
            x_corrupted = self.add_spike_and_slab_error(batch_key, x_data)
            
            # Recombine
            batch_corrupted = jnp.concatenate([theta, x_corrupted], axis=-1)
            corrupted_train_data.append(batch_corrupted)
        
        print(f"Applied spike-and-slab corruption to {len(corrupted_train_data)} training batches")
        
        # Use maximum likelihood training for NNPE
        key, self.flow = self.fit_nnpe_with_ml_loss(key, corrupted_train_data)
        
        return key
    
    def fit_nnpe_with_ml_loss(self, key, corrupted_train_data):
        """Fit NNPE using maximum likelihood loss."""
        loss_fn = MaximumLikelihoodLoss()
        
        params_flow, static_flow = eqx.partition(
            self.flow, eqx.is_inexact_array,
            is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
        )
        
        if self.use_embeddings:
            with open(self.file_name_embedding_stats, 'rb') as f:
                embedding_stats = pickle.load(f)
            embedding = paramax.unwrap(self.embedding)
            p_embedding = lambda x, k: embedding(x, key=k, inference=True)
        
        optimizer = self.optimizer_flow
        opt_state = optimizer.init(params_flow)
        
        max_epochs = self.config.training.n_iters
        best_params = params_flow
        best_loss = float('inf')
        patience_counter = 0
        max_patience = self.config.training.max_patience
        
        @eqx.filter_jit
        def step(params, batch, opt_state, key_step):
            theta = batch[..., :self.flow_dimension]
            x_data = batch[..., self.flow_dimension:]
            
            if self.use_embeddings:
                batch_size = x_data.shape[0]
                keys = jax.random.split(key_step, batch_size)
                x_processed = jax.lax.stop_gradient(
                    jax.vmap(p_embedding)(x_data[:, jnp.newaxis, :], keys)
                )
                x_processed = (x_processed - embedding_stats['mean']) / embedding_stats['std']
            else:
                x_processed = x_data
            
            loss_val, grads = eqx.filter_value_and_grad(loss_fn)(params, static_flow, theta, x_processed, None)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = eqx.apply_updates(params, updates)
            
            return params, opt_state, loss_val
        
        for epoch in tqdm(range(max_epochs), desc="NNPE Training"):
            epoch_losses = []
            for batch_data in corrupted_train_data:
                key, subkey = jr.split(key)
                params_flow, opt_state, loss_val = step(params_flow, batch_data, opt_state, subkey)
                epoch_losses.append(loss_val)
            
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_params = params_flow
                patience_counter = 0
            else:
                patience_counter += 1
                
            if epoch % 50 == 0:
                print(f"NNPE Epoch {epoch}, Loss: {avg_loss:.6f}")
                
            if patience_counter >= max_patience:
                print("Early stopping for NNPE training!")
                break
        
        flow = eqx.combine(best_params, static_flow)
        return key, flow


@register_flow(name='npe')
class NPE(Normalizing_Flow):
    """
    Standard Neural Posterior Estimation (NPE).
    
    Loads pre-trained embeddings and trains posterior using standard maximum likelihood
    without any corruption or data augmentation.
    """
    
    def __init__(self, config, key, data=None, eval_data=None, mean=None, std=None, experiment_idx=None):
        super().__init__(config)
        self.key, self.subkey = jr.split(key)
        
        # Model architecture parameters
        self.nn_depth = config.model.nn_depth_bnaf
        self.nn_block_dim = config.model.nn_block_dim
        self.cond_dim = config.model.cond_dim
        self.activation = get_activation(config)
        self.flow_dimension = config.model.flow_dimension
        
        # File paths for loading pre-trained components
        # Use the original RQS model name for loading embeddings
        embedding_base_shared = f"nlpe_rqs_posterior_{config.data.dataset}_{config.data.num_simulations}"
        weights_folder = os.path.join(config.data.data_path, f"{config.data.dataset}_weights")
        
        self.file_name_embedding = os.path.join(weights_folder, f"embedding_{embedding_base_shared}.eqx")
        self.file_name_decoder = os.path.join(weights_folder, f"decoder_{embedding_base_shared}.eqx")
        self.file_name_discriminator = os.path.join(weights_folder, f"discriminator_{embedding_base_shared}.eqx")
        self.file_name_embedding_stats = os.path.join(weights_folder, f"embedding_stats_{embedding_base_shared}.npy")
        
        # NPE-specific file path
        npe_base_shared = f"{config.model.name}_{config.data.dataset}_{config.data.num_simulations}"
        test_suffix = ""
        if hasattr(config.data, 'num_tests'):
            shrinkage_val = getattr(config.model, 'lambda_shrinkage', 0.0)
            test_suffix = f"_tests{config.data.num_tests}_npe_shrink{shrinkage_val}"
            
            # Add experiment index for multiple experiment scenarios
            if experiment_idx is not None:
                test_suffix += f"_exp{experiment_idx}"
        self.file_name = os.path.join(weights_folder, f"npe_{npe_base_shared}{test_suffix}.eqx")
        
        # Check if we use embeddings
        self.use_embeddings = config.data.dataset in ['spectra', 'pendulum']
        
        if self.use_embeddings:
            self.embedding_dim = config.model.embedding_dim
            flow_cond_dim = self.embedding_dim
        else:
            flow_cond_dim = config.data.vector_dim
        
        self.flow_cond_dim = flow_cond_dim
        self.key, subkey = jr.split(self.key)
        
        # Initialize posterior flow
        self.flow = masked_autoregressive_flow(
            key=subkey,
            cond_dim=flow_cond_dim,
            base_dist=Normal(jnp.zeros(self.flow_dimension), jnp.ones(self.flow_dimension)),
            transformer=RationalQuadraticSpline(knots=10, interval=5),
            flow_layers=self.nn_depth,
            nn_width=self.nn_block_dim,
            invert=False
        )

        # Initialize embedding components using factory function
        if self.use_embeddings:
            self.key, subkey = jr.split(self.key)
            self.embedding, _, _, _ = create_embedding_models(
                key=subkey,
                config=config,
                embedding_dim=self.embedding_dim,
                cond_dim=self.flow_dimension
            )
        else:
            self.embedding = None

        self.optimizer_flow = get_optimizer(self.config)

        print(f"NPE initialized for standard neural posterior estimation")
    
    def build(self, train_data=None, eval_data=None, inference_data=None, mean=None, std=None):
        """Build and train NPE model."""
        self.key, subkey = jr.split(self.key)
        self.train_data = train_data
        self.eval_data = eval_data
        self.inference_data = inference_data
        
        assert(mean is not None and std is not None), "Mean and std must be provided"
        self.mean, self.std = mean, std
        self.inference_data = (inference_data - self.mean) / self.std
        
        # Compute theta bounds and empirical bias
        all_theta = jnp.array(train_data)[..., :self.flow_dimension].reshape(-1, self.flow_dimension)
        self.theta_min = jnp.min(all_theta, axis=0)
        self.theta_max = jnp.max(all_theta, axis=0)
        
        train_x = jnp.array(train_data)[..., self.flow_dimension:]
        inference_x = self.inference_data[..., self.flow_dimension:]
        self.empirical_bias = jnp.mean(inference_x, axis=0) - jnp.mean(train_x.reshape(-1, train_x.shape[-1]), axis=0)
        
        self.prior = get_prior_from_config(self.config, subkey, mean=self.mean, std=self.std,
                                         theta_min=self.theta_min-2.8, theta_max=self.theta_max+2.8)
        
        if self.train_bool:
            # Load pre-trained embeddings if available
            if self.use_embeddings:
                print("Loading pre-trained embeddings...")
                try:
                    self.embedding = eqx.tree_deserialise_leaves(self.file_name_embedding, self.embedding)
                    print(f"Loaded embedding from {self.file_name_embedding}")
                except FileNotFoundError:
                    print(f"Pre-trained embedding not found: {self.file_name_embedding}")
                    print("Will train NPE without pre-trained embeddings")
            
            # Train NPE posterior with standard data
            print(f"Training NPE posterior with standard maximum likelihood...")
            self.key = self.train_npe_posterior(subkey, train_data)
            
            # Save NPE model
            eqx.tree_serialise_leaves(self.file_name, self.flow)
            print(f"NPE model saved to {self.file_name}")
        else:
            # Load trained NPE model
            try:
                self.flow = eqx.tree_deserialise_leaves(self.file_name, self.flow)
                if self.use_embeddings:
                    self.embedding = eqx.tree_deserialise_leaves(self.file_name_embedding, self.embedding)
                print(f"Loaded NPE model from {self.file_name}")
            except FileNotFoundError:
                print(f"NPE model not found: {self.file_name}. Training new model...")
                self.key = self.train_npe_posterior(subkey, train_data)
                eqx.tree_serialise_leaves(self.file_name, self.flow)
    
    def train_npe_posterior(self, key, train_data):
        """Train NPE posterior with standard maximum likelihood."""
        print(f"Training NPE posterior with standard data...")
        
        # Use maximum likelihood training for NPE
        key, self.flow = self.fit_npe_with_ml_loss(key, train_data)
        
        return key
    
    def fit_npe_with_ml_loss(self, key, train_data):
        """Fit NPE using standard maximum likelihood loss."""
        loss_fn = MaximumLikelihoodLoss()
        
        params_flow, static_flow = eqx.partition(
            self.flow, eqx.is_inexact_array,
            is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
        )
        
        if self.use_embeddings:
            try:
                with open(self.file_name_embedding_stats, 'rb') as f:
                    embedding_stats = pickle.load(f)
                embedding = paramax.unwrap(self.embedding)
                p_embedding = lambda x, k: embedding(x, key=k, inference=True)
            except FileNotFoundError:
                print("Warning: Embedding stats not found. Using raw data.")
                embedding_stats = None
                p_embedding = None
        
        optimizer = self.optimizer_flow
        opt_state = optimizer.init(params_flow)
        
        max_epochs = self.config.training.n_iters
        best_params = params_flow
        best_loss = float('inf')
        patience_counter = 0
        max_patience = self.config.training.max_patience
        
        @eqx.filter_jit
        def step(params, batch, opt_state, key_step):
            batch = jnp.array(batch) if not isinstance(batch, jnp.ndarray) else batch
            theta = batch[..., :self.flow_dimension]
            x_data = batch[..., self.flow_dimension:]
            
            # Process through embeddings if available
            if self.use_embeddings and embedding_stats is not None and p_embedding is not None:
                batch_size = x_data.shape[0]
                keys = jax.random.split(key_step, batch_size)
                x_processed = jax.lax.stop_gradient(
                    jax.vmap(p_embedding)(x_data[:, jnp.newaxis, :], keys)
                )
                x_processed = (x_processed - embedding_stats['mean']) / embedding_stats['std']
            else:
                x_processed = x_data
            
            loss_val, grads = eqx.filter_value_and_grad(loss_fn)(params, static_flow, theta, x_processed, None)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = eqx.apply_updates(params, updates)
            
            return params, opt_state, loss_val
        
        for epoch in tqdm(range(max_epochs), desc="NPE Training"):
            epoch_losses = []
            for batch_data in train_data:
                key, subkey = jr.split(key)
                params_flow, opt_state, loss_val = step(params_flow, batch_data, opt_state, subkey)
                epoch_losses.append(loss_val)
            
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_params = params_flow
                patience_counter = 0
            else:
                patience_counter += 1
                
            if epoch % 50 == 0:
                print(f"NPE Epoch {epoch}, Loss: {avg_loss:.6f}")
                
            if patience_counter >= max_patience:
                print("Early stopping for NPE training!")
                break
        
        flow = eqx.combine(best_params, static_flow)
        return key, flow


@register_flow(name='ranpt')
class RANPT(Normalizing_Flow):
    """
    Robust Amortized Neural Posterior Training (RANPT).
    
    Loads pre-trained embeddings and correction model from RQS training,
    then trains posterior using the pre-trained correction model for data augmentation.
    """
    
    def __init__(self, config, key, data=None, eval_data=None, mean=None, std=None, experiment_idx=None):
        super().__init__(config)
        self.key, self.subkey = jr.split(key)
        
        # Model architecture parameters
        self.nn_depth = config.model.nn_depth_bnaf
        self.nn_block_dim = config.model.nn_block_dim
        self.cond_dim = config.model.cond_dim
        self.activation = get_activation(config)
        self.flow_dimension = config.model.flow_dimension
        
        # RANPT-specific parameters
        self.correction_model_name = config.model.correction_model_name  # Which correction model to load
        
        # File paths for loading pre-trained components
        # Use the original RQS model name for loading embeddings and correction model
        rqs_base_shared = f"nlpe_rqs_posterior_{config.data.dataset}_{config.data.num_simulations}"
        weights_folder = os.path.join(config.data.data_path, f"{config.data.dataset}_weights")
        
        self.file_name_embedding = os.path.join(weights_folder, f"embedding_{rqs_base_shared}.eqx")
        self.file_name_decoder = os.path.join(weights_folder, f"decoder_{rqs_base_shared}.eqx")
        self.file_name_discriminator = os.path.join(weights_folder, f"discriminator_{rqs_base_shared}.eqx")
        self.file_name_embedding_stats = os.path.join(weights_folder, f"embedding_stats_{rqs_base_shared}.npy")
        
        # Correction model file path - needs test suffix to match the trained correction model
        correction_test_suffix = ""
            # Match the correction model that was trained with the RQS class
        correction_shrinkage = config.model.correction_lambda_shrinkage  # Shrinkage used in correction training
        correction_type = config.model.correction_type
        correction_test_suffix = f"_tests{config.data.num_tests}_{correction_type}_shrink{correction_shrinkage}"
        
        # Add experiment index for multiple experiment scenarios
        if experiment_idx is not None:
            correction_test_suffix += f"_exp{experiment_idx}"
    
        self.file_name_correction = os.path.join(weights_folder, f"correction_{rqs_base_shared}{correction_test_suffix}.eqx")
        
        # RQS posterior file path for initialization
        rqs_test_suffix = ""
        shrinkage_val = config.model.lambda_shrinkage
        rqs_test_suffix = f"_tests{config.data.num_tests}_{correction_type}_shrink{shrinkage_val}"
        
        # Add experiment index for multiple experiment scenarios
        if experiment_idx is not None:
            rqs_test_suffix += f"_exp{experiment_idx}"
        self.file_name_rqs_init = os.path.join(weights_folder, f"{rqs_base_shared}{rqs_test_suffix}.eqx")
        
        # RANPT-specific file path
        ranpt_base_shared = f"{config.model.name}_{config.data.dataset}_{config.data.num_simulations}"
        test_suffix = ""
        shrinkage_val = getattr(config.model, 'lambda_shrinkage', 0.0)
        test_suffix = f"_tests{config.data.num_tests}_ranpt_{self.correction_model_name}_shrink{shrinkage_val}"
        
        # Add experiment index for multiple experiment scenarios
        if experiment_idx is not None:
            test_suffix += f"_exp{experiment_idx}"
        self.file_name = os.path.join(weights_folder, f"ranpt_{ranpt_base_shared}{test_suffix}.eqx")
        
        # Check if we use embeddings
        self.use_embeddings = config.data.dataset in ['spectra', 'pendulum']
        
        if self.use_embeddings:
            self.embedding_dim = config.model.embedding_dim
            flow_cond_dim = self.embedding_dim
        else:
            flow_cond_dim = config.data.vector_dim
        
        self.flow_cond_dim = flow_cond_dim
        self.key, subkey = jr.split(self.key)
        
        # Initialize posterior flow
        self.flow = masked_autoregressive_flow(
            key=subkey,
            cond_dim=flow_cond_dim,
            base_dist=Normal(jnp.zeros(self.flow_dimension), jnp.ones(self.flow_dimension)),
            transformer=RationalQuadraticSpline(knots=10, interval=5),
            flow_layers=self.nn_depth,
            nn_width=self.nn_block_dim,
            invert=False
        )
        
        # Initialize embedding components (will be loaded)
        if self.use_embeddings:
            self.key, subkey = jr.split(self.key)
            embedding_type = config.model.embedding
            if config.data.dataset == 'spectra':
                self.embedding = StatisticEmbedding_spectra(
                    key=subkey, in_channels=1, how=embedding_type,
                    hidden_scale=config.model.hidden_scale, z_dim=self.embedding_dim, dropout_rate=0.1
                )
            else:
                self.embedding = StatisticEmbedding_pendulum(
                    key=subkey, in_channels=1, how=embedding_type,
                    hidden_scale=config.model.hidden_scale, z_dim=self.embedding_dim, dropout_rate=0.1
                )
        else:
            self.embedding = None
        
        # Initialize correction model (will be loaded)
        self.key, subkey = jr.split(self.key)
        correction_type = self.correction_model_name
        correction_dim = self.embedding_dim if self.use_embeddings else config.data.vector_dim
        
        if correction_type == 'simple':
            self.correction_model = SimpleCorrectionModel(key=subkey, dim=correction_dim)
        elif correction_type == 'diagonal_neural':
            self.correction_model = DiagonalNeuralCorrectionModel(key=subkey, theta_dim=self.flow_dimension, output_dim=correction_dim)
        elif correction_type == 'hybrid':
            self.correction_model = HybridCorrectionModel(key=subkey, theta_dim=self.flow_dimension, output_dim=correction_dim)
        elif correction_type == 'NN':
            self.correction_model = MuHybridCorrectionModel(key=subkey, theta_dim=self.flow_dimension, output_dim=correction_dim)
        elif correction_type == 'global':
            self.correction_model = GlobalCorrectionModel(key=subkey, output_dim=correction_dim)
        elif correction_type == 'full_neural':
            self.correction_model = FullNeuralCorrectionModel(key=subkey, theta_dim=self.flow_dimension, output_dim=correction_dim)
        else:
            raise ValueError(f"Unknown correction model type: {correction_type}")
            
        self.optimizer_flow = get_optimizer(self.config)
        
        print(f"RANPT initialized with correction_model={self.correction_model_name}")
    
    def build(self, train_data=None, eval_data=None, inference_data=None, mean=None, std=None):
        """Build and train RANPT model."""
        self.key, subkey = jr.split(self.key)
        self.train_data = train_data
        self.eval_data = eval_data
        self.inference_data = inference_data
        
        assert(mean is not None and std is not None), "Mean and std must be provided"
        self.mean, self.std = mean, std
        self.inference_data = (inference_data - self.mean) / self.std
        
        # Compute theta bounds and empirical bias
        all_theta = jnp.array(train_data)[..., :self.flow_dimension].reshape(-1, self.flow_dimension)
        self.theta_min = jnp.min(all_theta, axis=0)
        self.theta_max = jnp.max(all_theta, axis=0)
        
        train_x = jnp.array(train_data)[..., self.flow_dimension:]
        inference_x = self.inference_data[..., self.flow_dimension:]
        self.empirical_bias = jnp.mean(inference_x, axis=0) - jnp.mean(train_x.reshape(-1, train_x.shape[-1]), axis=0)
        
        self.prior = get_prior_from_config(self.config, subkey, mean=self.mean, std=self.std,
                                         theta_min=self.theta_min-2.8, theta_max=self.theta_max+2.8)
        
        if self.train_bool:
            # Load pre-trained embeddings
            if self.use_embeddings:
                print("Loading pre-trained embeddings...")
                try:
                    self.embedding = eqx.tree_deserialise_leaves(self.file_name_embedding, self.embedding)
                    print(f"Loaded embedding from {self.file_name_embedding}")
                except FileNotFoundError:
                    raise FileNotFoundError(f"Pre-trained embedding not found: {self.file_name_embedding}")
            
            # Load pre-trained correction model
            print(f"Loading pre-trained correction model ({self.correction_model_name})...")
            try:
                self.correction_model = eqx.tree_deserialise_leaves(self.file_name_correction, self.correction_model)
                print(f"Loaded correction model from {self.file_name_correction}")
            except FileNotFoundError:
                raise FileNotFoundError(f"Pre-trained correction model not found: {self.file_name_correction}")
            
            # Initialize from pre-trained RQS posterior
            print(f"Initializing RANPT posterior from RQS model...")
            try:
                self.flow = eqx.tree_deserialise_leaves(self.file_name_rqs_init, self.flow)
                print(f"Initialized RANPT flow from RQS posterior: {self.file_name_rqs_init}")
            except FileNotFoundError:
                raise FileNotFoundError(f"Pre-trained RQS posterior not found: {self.file_name_rqs_init}. RANPT requires RQS initialization.")
            
            # Train RANPT posterior with correction-augmented data
            print(f"Training RANPT posterior with {self.correction_model_name} correction model...")
            self.key = self.train_ranpt_posterior(subkey, train_data, inference_data)
            
            # Save RANPT model
            eqx.tree_serialise_leaves(self.file_name, self.flow)
            print(f"RANPT model saved to {self.file_name}")

            # Log parameter counts
            self._log_parameter_counts()
        else:
            # Load trained RANPT model
            try:
                self.flow = eqx.tree_deserialise_leaves(self.file_name, self.flow)
                if self.use_embeddings:
                    self.embedding = eqx.tree_deserialise_leaves(self.file_name_embedding, self.embedding)
                self.correction_model = eqx.tree_deserialise_leaves(self.file_name_correction, self.correction_model)
                print(f"Loaded RANPT model from {self.file_name}")

                # Log parameter counts for loaded model
                self._log_parameter_counts()
            except FileNotFoundError:
                print(f"RANPT model not found: {self.file_name}. Training new model...")
                self.key = self.train_ranpt_posterior(subkey, train_data, inference_data)
                eqx.tree_serialise_leaves(self.file_name, self.flow)
    
    def train_ranpt_posterior(self, key, train_data, inference_data):
        """Train RANPT posterior with correction model corruption."""
        print(f"Training RANPT posterior with JIT correction model corruption...")
        
        # Use maximum likelihood training for RANPT with JIT corruption
        key, self.flow = self.fit_ranpt_with_ml_loss(key, train_data)
        
        return key
    
    def apply_correction_corruption_vectorized(self, key, x_data, theta):
        """Apply correction model as corruption using vmap for speed, generating 32 samples per (theta, x_sim) pair."""
        batch_size = x_data.shape[0]
        n_samples = 32  # Number of corrected samples per (theta, x_sim) pair
        
        # Generate keys for batch_size x n_samples  
        # Use a different approach: split the key hierarchically
        batch_keys = jr.split(key, batch_size)  # Shape: (batch_size, 2)
        
        def generate_sample_keys(batch_key):
            return jr.split(batch_key, n_samples)  # Shape: (n_samples, 2)
        
        keys = jax.vmap(generate_sample_keys)(batch_keys)  # Shape: (batch_size, n_samples, 2)
        
        def corrupt_single_pair(x_single, theta_single, keys_for_pair):
            """Generate 32 corrected samples for a single (theta, x) pair."""
            
            def corrupt_one_sample(key_single):
                # Use correction model to generate corrected sample
            
                # Neural correction models need theta input
                mu, sigma = self.correction_model(x_single[None, :], theta_single[None, :])
                mu, sigma = mu[0], sigma[0]

                
                # Sample from correction distribution
                if len(sigma.shape) == 1:  # Diagonal covariance
                    corrected_sample = mu + jnp.sqrt(sigma) * jr.normal(key_single, x_single.shape)
                else:  # Full covariance matrix
                    L = jnp.linalg.cholesky(sigma + 1e-6 * jnp.eye(sigma.shape[0]))
                    corrected_sample = mu + L @ jr.normal(key_single, x_single.shape)
                
                return corrected_sample
            
            # Vectorize over the 32 samples for this (theta, x) pair
            return jax.vmap(corrupt_one_sample)(keys_for_pair)
        
        # Vectorize over the batch, generating 32 samples per pair
        corrupted_samples = jax.vmap(corrupt_single_pair)(x_data, theta, keys)  # Shape: (batch_size, 32, x_dim)
        
        # Reshape to (batch_size * 32, x_dim)
        corrupted_samples_flat = corrupted_samples.reshape(-1, x_data.shape[-1])
        
        # Repeat theta 32 times to match the corrupted samples
        theta_repeated = jnp.repeat(theta, n_samples, axis=0)  # Shape: (batch_size * 32, theta_dim)
        
        # Combine theta and corrupted x samples
        return jnp.concatenate([theta_repeated, corrupted_samples_flat], axis=-1)
    
    def fit_ranpt_with_ml_loss(self, key, train_data):
        """Fit RANPT using maximum likelihood loss with JIT corruption."""
        loss_fn = MaximumLikelihoodLoss()
        
        params_flow, static_flow = eqx.partition(
            self.flow, eqx.is_inexact_array,
            is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
        )
        
        if self.use_embeddings:
            with open(self.file_name_embedding_stats, 'rb') as f:
                embedding_stats = pickle.load(f)
            embedding = paramax.unwrap(self.embedding)
            p_embedding = lambda x, k: embedding(x, key=k, inference=True)
        
        optimizer = self.optimizer_flow
        opt_state = optimizer.init(params_flow)
        
        max_epochs = self.config.training.n_iters
        best_params = params_flow
        best_loss = float('inf')
        patience_counter = 0
        max_patience = self.config.training.max_patience
        
        @eqx.filter_jit
        def step(params, batch, opt_state, key_step):
            batch = jnp.array(batch) if not isinstance(batch, jnp.ndarray) else batch
            theta = batch[..., :self.flow_dimension]
            x_data = batch[..., self.flow_dimension:]
            
            # For embedding tasks, process through embeddings FIRST, then apply correction
            if self.use_embeddings:
                batch_size = x_data.shape[0]
                key_embed, key_corrupt = jr.split(key_step)
                keys = jax.random.split(key_embed, batch_size)
                
                # Apply embeddings to raw data to get embedded data
                x_embedded = jax.lax.stop_gradient(
                    jax.vmap(p_embedding)(x_data[:, jnp.newaxis, :], keys)
                )
                x_embedded = (x_embedded - embedding_stats['mean']) / embedding_stats['std']
                
                # Apply correction model corruption to embedded data
                expanded_batch = self.apply_correction_corruption_vectorized(key_corrupt, x_embedded, theta)
                
                # Extract theta and x from the expanded batch (x is already processed/embedded)
                theta_expanded = expanded_batch[..., :self.flow_dimension]
                x_processed = expanded_batch[..., self.flow_dimension:]
            else:
                # For non-embedding tasks, apply correction to raw data as before
                key_corrupt, key_step = jr.split(key_step)
                expanded_batch = self.apply_correction_corruption_vectorized(key_corrupt, x_data, theta)
                
                # Extract theta and x from the expanded batch
                theta_expanded = expanded_batch[..., :self.flow_dimension]
                x_processed = expanded_batch[..., self.flow_dimension:]
            
            loss_val, grads = eqx.filter_value_and_grad(loss_fn)(params, static_flow, theta_expanded, x_processed, None)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = eqx.apply_updates(params, updates)
            
            return params, opt_state, loss_val
        
        for epoch in tqdm(range(max_epochs), desc="RANPT Training"):
            epoch_losses = []
            for batch_data in train_data:
                key, subkey = jr.split(key)
                params_flow, opt_state, loss_val = step(params_flow, batch_data, opt_state, subkey)
                epoch_losses.append(loss_val)
            
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_params = params_flow
                patience_counter = 0
            else:
                patience_counter += 1
                
            if epoch % 50 == 0:
                print(f"RANPT Epoch {epoch}, Loss: {avg_loss:.6f}")
                
            if patience_counter >= max_patience:
                print("Early stopping for RANPT training!")
                break
        
        flow = eqx.combine(best_params, static_flow)
        return key, flow

    def _log_parameter_counts(self):
        """Log parameter counts for all model components."""
        print("\n" + "="*80)
        print("MODEL PARAMETER BREAKDOWN")
        print("="*80)

        # Get comprehensive breakdown
        breakdown = get_ranpt_model_breakdown(self)

        # Print it
        print_parameter_breakdown(breakdown)

        # Save to file in workspace
        if hasattr(self, 'workspace_path'):
            param_file = os.path.join(self.workspace_path, 'parameter_breakdown.json')
            save_parameter_breakdown(breakdown, param_file)

        print("="*80)
        print()


def log_correction_model_diagnostics(correction_model, stage_name: str, eval_data=None):
    """
    Log correction model diagnostics at the end of each stage.
    
    Args:
        correction_model: The correction model to analyze
        stage_name: Name of the stage that just completed
        eval_data: Optional evaluation data for computing log probabilities
    """
    print(f"\n🔍 === CORRECTION MODEL DIAGNOSTICS: {stage_name} ===")

    from .models.correction_model import SimpleCorrectionModel, HybridCorrectionModel, FullNeuralCorrectionModel, MuHybridCorrectionModel, GlobalCorrectionModel
    
    if isinstance(correction_model, SimpleCorrectionModel):
        # Get the covariance matrix from Cholesky parameterization
        covariance_matrix = correction_model.get_covariance_matrix()
        diagonal_variances = jnp.diag(covariance_matrix)
        
        print(f"📊 Diagonal Variances: {diagonal_variances}")
        print(f"📊 Covariance Matrix Shape: {covariance_matrix.shape}")
        print(f"📊 Covariance Determinant: {jnp.linalg.det(covariance_matrix):.6f}")
        print(f"📊 Off-diagonal Elements (max abs): {jnp.max(jnp.abs(covariance_matrix - jnp.diag(diagonal_variances))):.6f}")
    
    elif isinstance(correction_model, GlobalCorrectionModel):
        # GlobalCorrectionModel - no theta dependence
        covariance_matrix = correction_model.get_covariance_matrix()
        diagonal_variances = jnp.diag(covariance_matrix)
        
        print(f"📊 Global Model:")
        print(f"📊 Diagonal Variances: {diagonal_variances}")
        print(f"📊 Covariance Matrix Shape: {covariance_matrix.shape}")
        print(f"📊 Covariance Determinant: {jnp.linalg.det(covariance_matrix):.6f}")
        print(f"📊 Off-diagonal Elements (max abs): {jnp.max(jnp.abs(covariance_matrix - jnp.diag(diagonal_variances))):.6f}")
        print(f"📊 Global Mean Shift: {correction_model.mu_global}")
    
    elif isinstance(correction_model, (HybridCorrectionModel, MuHybridCorrectionModel, FullNeuralCorrectionModel)):
        # For theta-dependent models, use dummy theta = 0
        # Get the correct theta dimension from the neural network input size
        if isinstance(correction_model, (HybridCorrectionModel, MuHybridCorrectionModel)):
            flow_dim = correction_model.local_cholesky_net.layers[0].in_features
        else:  # FullNeuralCorrectionModel
            flow_dim = correction_model.cholesky_net.layers[0].in_features
        dummy_theta = jnp.zeros(flow_dim)
        covariance_matrix = correction_model.get_covariance_matrix(dummy_theta)
        diagonal_variances = jnp.diag(covariance_matrix)
        
        if isinstance(correction_model, HybridCorrectionModel):
            model_type = "Hybrid"
        elif isinstance(correction_model, MuHybridCorrectionModel):
            model_type = "Mu-Hybrid"
        else:
            model_type = "Full Neural"
        print(f"📊 {model_type} Model (at theta=0):")
        print(f"📊 Diagonal Variances: {diagonal_variances}")
        print(f"📊 Covariance Matrix Shape: {covariance_matrix.shape}")
        print(f"📊 Covariance Determinant: {jnp.linalg.det(covariance_matrix):.6f}")
        print(f"📊 Off-diagonal Elements (max abs): {jnp.max(jnp.abs(covariance_matrix - jnp.diag(diagonal_variances))):.6f}")
        
        # Compute average log probability if evaluation data is provided
        if eval_data is not None:
            try:
                # Use a small sample for efficiency - use all dimensions that model expects
                x_sim_sample = eval_data[:20]  # First 20 samples, all dimensions
                x_obs_sample = eval_data[:20]  # Use same for simplicity
                
                # Compute log probabilities
                log_probs = []
                for i in range(min(20, x_sim_sample.shape[0])):
                    log_prob = correction_model.log_prob(x_obs_sample[i], x_sim_sample[i])
                    log_probs.append(log_prob)
                
                avg_log_prob = jnp.mean(jnp.array(log_probs))
                print(f"📊 Average Log Probability (sample): {avg_log_prob:.4f}")
            except Exception as e:
                print(f"📊 Could not compute log probabilities: {e}")
    else:
        print(f"📊 Correction model type: {type(correction_model)}")
    
    print(f"🔍 === END DIAGNOSTICS: {stage_name} ===\n")
