import jax
import tensorflow as tf
import os
import numpy as np
import pickle
import jax.numpy as jnp
from jax import random
from utils import CS,SIR
from typing import Tuple, Optional


def get_dataset(config, evaluation=False,load_data=True,return_summary=False):
    """Load and preprocess training dataset for RVNP training.

    Loads the dataset specified in config, performs validation splitting, and batches
    the data for efficient training. Handles normalization statistics if requested.

    Args:
        config: Configuration object containing dataset and training parameters.
               Required attributes: training.batch_size, eval.batch_size, data.dataset,
               data.num_simulations
        evaluation: If True, uses eval batch size and single epoch. If False, uses
                   training batch size. Default: False
        load_data: If True, loads actual data from disk. If False, only returns
                  statistics (used for memory efficiency). Default: True
        return_summary: If True, additionally returns (data_mean, data_std) for
                       normalization. Default: False

    Returns:
        If return_summary=False:
            tuple: (train_ds, eval_ds)
                - train_ds: Batched training dataset
                - eval_ds: Batched validation dataset

        If return_summary=True:
            tuple: (train_ds, eval_ds, data_mean, data_std)
                - train_ds: Batched training dataset
                - eval_ds: Batched validation dataset
                - data_mean: Mean for normalization (array)
                - data_std: Standard deviation for normalization (array)

    Raises:
        ValueError: If batch_size is not divisible by number of devices

    Notes:
        - Automatically handles multi-device setups (sharding across GPUs/TPUs)
        - Applies preprocessing and batching via preprocess_and_batch()
        - Uses random seed 0 for reproducible train/val split
        - Loads from disk if data exists, otherwise generates via load_custom_dataset()

    Example:
        >>> train_ds, eval_ds = get_dataset(config)
        >>> train_ds, eval_ds, mean, std = get_dataset(config, return_summary=True)
    """
    batch_size = config.training.batch_size if not evaluation else config.eval.batch_size
    if batch_size % jax.device_count() != 0:
        raise ValueError(f'Batch size ({batch_size}) must be divisible by the number of devices ({jax.device_count()})')
    
    per_device_batch_size = batch_size 
    num_epochs = None if not evaluation else 1
    batch_dims = [ per_device_batch_size] 
    dataset_name = config.data.dataset
    num_simulations=config.data.num_simulations
    if(return_summary):
        train_ds, eval_ds,data_mean,data_std = load_custom_dataset(dataset_name, config,num_simulations,load_data,return_summary)
        key, subkey = random.split(random.PRNGKey(0))
        train_ds = preprocess_and_batch(train_ds, batch_dims, key, evaluation)
        key, subkey = random.split(key)
        eval_ds = preprocess_and_batch(eval_ds, batch_dims, key, evaluation)
        return train_ds, eval_ds,data_mean,data_std
    else:
        key, subkey = random.split(random.PRNGKey(0))
        train_ds, eval_ds = load_custom_dataset(dataset_name, config,num_simulations,load_data)
        train_ds = preprocess_and_batch(train_ds, batch_dims, key, evaluation)
        key, subkey = random.split(key)
        eval_ds = preprocess_and_batch(eval_ds, batch_dims, key, evaluation)
        return train_ds, eval_ds


def get_inference_dataset(config, evaluation=False):
    """Load inference dataset containing observed data for posterior inference.

    Loads the dataset used for computing posteriors on observed data x_obs. This is
    separate from the training dataset and represents the real observations for which
    we want to infer parameters.

    Args:
        config: Configuration object containing inference dataset parameters.
               Required attributes: training.batch_size, eval.batch_size,
               data.inference_dataset
        evaluation: If True, uses eval batch size. If False, uses training batch size.
                   Default: False

    Returns:
        tuple: (train_ds, eval_ds)
            - train_ds: Batched inference dataset (typically single observation)
            - eval_ds: Batched inference dataset (for validation)

    Notes:
        - Used during evaluation/inference phase, not training
        - Dataset typically contains real observed data x_obs
        - Returns normalized data if training data was normalized
        - For single observation inference, batch may contain just 1 sample

    Example:
        >>> inference_ds = get_inference_dataset(config)
        >>> # Use for computing posteriors on observed data
    """
    batch_size = config.training.batch_size if not evaluation else config.eval.batch_size
    num_simulations=config.data.inference_simulations
    if batch_size % jax.device_count() != 0:
        raise ValueError(f'Batch size ({batch_size}) must be divisible by the number of devices ({jax.device_count()})')
    
    per_device_batch_size = batch_size 
    num_epochs = None if not evaluation else 1
    batch_dims = [ per_device_batch_size] 
    inference_dataset_name = config.data.inference_dataset


    train_ds, eval_ds = load_custom_dataset(inference_dataset_name, config,num_simulations,load_data=True,return_summary=False,inference=True) #return summary always false for this, load data alwyas true
    key, subkey = random.split(random.PRNGKey(0))
    train_ds = preprocess_and_batch(train_ds, batch_dims, key, evaluation)
    key, subkey = random.split(key)
    eval_ds = preprocess_and_batch(eval_ds, batch_dims, key, evaluation)
    return train_ds, eval_ds
    
    

def load_custom_dataset(dataset_name, config,num_simulations,load_data=True,return_summary=False,inference=False):
    """Load custom dataset from external files or generate it if needed.
    
    This function loads a dataset from external files if they exist. If the files do not exist,
    it generates the dataset using the specified configuration. The function also computes the 
    mean and standard deviation of the dataset for normalization purposes. The standardized data 
    is only saved if it's training data (inference=False); otherwise, the regular, unscaled data 
    is saved. The function returns the training and evaluation datasets, and optionally the mean 
    and standard deviation if return_summary is True.
    """
    data_path = os.path.join(config.data.data_path, f"{dataset_name}_{num_simulations}.npy")
    stats_path = os.path.join(config.data.data_path, f"{dataset_name}_stats_{num_simulations}.npy")

    if os.path.exists(data_path):
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)  # Load saved stats if they exist
        mean, std = stats['mean'], stats['std']
        if(load_data==False):
            if(return_summary):
                data=jnp.zeros((1,config.data.summary_dim+config.model.cond_dim+config.data.vector_dim))
            else:
                raise ValueError("Data not loaded, must have return_summary= False then load_data=True")
        else:
            data = np.load(data_path, allow_pickle=True)
    else:
        if(config.data.benchmark==True):
            data = generate_sbibm_data(config)            
            # Compute mean and std for training data (joint_data)
            mean = np.mean(data[:, :], axis=0)
            std = np.std(data[:, :], axis=0)
        else:
            data = generate_other_data(config,dataset_name,num_simulations)
            # Compute mean and std for training data (joint_data)
            mean = np.mean(data[:, :], axis=0)
            std = np.std(data[:, :], axis=0)
        if(inference==False):
            #std[3:]=1 #set the std of the last 6 dimensions to 1, this is for the spectra data
            # we only scale data for the training, not for the inference!
            np.save(data_path, (data-mean)/std) 
        else:
            assert inference==True #"Inference must be true"
            np.save(data_path, data)
        
        # Save mean and std to a file
        with open(stats_path, 'wb') as f:
            pickle.dump({'mean': mean, 'std': std}, f)
            print('finished')

    
    dataset=data
    n = len(dataset)

    # Split dataset into training and evaluation sets
    if not inference:
        p = config.training.validation_split
    else:
        p = 1  # Use entire dataset as eval

    split_idx = int((1 - p) * n)

    train_ds = dataset[:split_idx]
    eval_ds = dataset[split_idx:]

    if return_summary:
        return train_ds, eval_ds, mean, std
    else: 
        return train_ds, eval_ds



def generate_other_data(config,dataset_name,num_simulations):
    """Generate data using other task and simulator, with conditional prior sampling."""
    other_tasks=['gaussian_embedding','gaussian_embedding_inference','CS','CS_inference',
                 'gaussian_mixture','gaussian_mixture_inference','SIR','SIR_inference','spectra_inference','spectra','pendulum','pendulum_inference']
    assert dataset_name in other_tasks, f"Dataset name should be one of {other_tasks}"
    num_samples=num_simulations
    cond_dim=config.model.cond_dim
    vector_dim=config.model.flow_dimension

    if dataset_name=='gaussian_embedding':
        # Generate synthetic Gaussian data
        theta = np.random.normal(loc=0,scale=5, size=(num_samples, vector_dim))
        eps= np.random.normal(loc=0,scale=1, size=(num_samples, 100))
        data = theta + eps
        feature_mean=data.mean(-1)[:,None]
        feature_std=data.std(-1)[:,None]
        data = np.concatenate((theta, feature_mean,feature_std), axis=1)  # Concatenate theta and eps to form the data
        

    if dataset_name=='gaussian_embedding_inference':
        # Generate synthetic Gaussian data
        theta_list = []
        eps_list = []
        feature_mean_list = []
        feature_std_list = []

        for i in range(num_samples):
            rng = np.random.default_rng(seed=42 + i)  # Fixed base seed + index

            theta_i = rng.normal(loc=0, scale=5, size=(vector_dim,))
            eps_i = rng.normal(loc=0, scale=np.sqrt(2), size=(100,))
            data_i = theta_i + eps_i

            feature_mean = np.mean(data_i)
            feature_std = np.std(data_i)

            theta_list.append(theta_i)
            feature_mean_list.append([feature_mean])  # keep shape (1,)
            feature_std_list.append([feature_std])

        # Stack into arrays
        theta = np.array(theta_list)                    # shape: (num_samples, cond_dim)
        feature_mean = np.array(feature_mean_list)      # shape: (num_samples, 1)
        feature_std = np.array(feature_std_list)        # shape: (num_samples, 1)

        # Concatenate features: theta + mean + std
        data = np.concatenate((theta, feature_mean, feature_std), axis=1)  # shape: (num_samples, cond_dim + 2)



    if dataset_name=='CS':
        # Generate synthetic Gaussian data
        key, subkey = random.split(random.PRNGKey(0))
        base = num_samples // 50
        remainder = num_samples % 50
        parts = [base + 1 if i < remainder else base for i in range(50)]
        thetas=[]
        datas=[]
        for part in parts:
            data=CS().generate_dataset(key=subkey, n=part, misspecified=False,scale=False)
            key, subkey = random.split(key)
            theta=data['theta']
            data=data['x']
            thetas.append(theta)
            datas.append(data)
        data=np.concatenate(datas,axis=0)
        theta=np.concatenate(thetas,axis=0)
        assert data.shape[-1]==config.data.vector_dim
        assert theta.shape[-1]==config.model.flow_dimension
        data = np.concatenate((theta, data), axis=1)  # Concatenate theta and eps to form the data
        

    if dataset_name=='CS_inference':
        # Generate synthetic Gaussian data
        
        key, subkey = random.split(random.PRNGKey(0))
        outs=[]
        for  i in range(num_samples):
            key, subkey = random.split(key)
            outs.append(CS().generate_observation(subkey, misspecified=True))
            

        theta,data,_=zip(*outs)
        theta=np.array(theta)
        data=np.array(data)


        assert data.shape[-1]==config.data.vector_dim
        assert theta.shape[-1]==config.model.flow_dimension
        data = np.concatenate((theta, data), axis=1)
        print(data.shape)


    if dataset_name=='SIR':
        # Generate synthetic Gaussian data
        key, subkey = random.split(random.PRNGKey(0))
        base = num_samples // 10
        remainder = num_samples % 10
        parts = [base + 1 if i < remainder else base for i in range(10)]
        thetas=[]
        datas=[]
        for part in parts:
            print(part)
            data=SIR(config.data.julia_env_path).generate_dataset(key=subkey, n=part, misspecified=True,scale=False)
            key, subkey = random.split(key)
            theta=data['theta']
            data=data['x']
            thetas.append(theta)
            datas.append(data)
        data=np.concatenate(datas,axis=0)
        theta=np.concatenate(thetas,axis=0)
        assert data.shape[-1]==config.data.vector_dim
        assert theta.shape[-1]==config.model.cond_dim
        data = np.concatenate((theta, data), axis=1)  # Concatenate theta and eps to form the data
        print(data.shape)

    if dataset_name=='SIR_inference':
        # Generate synthetic Gaussian data
        

        key, subkey = random.split(random.PRNGKey(42))
        base = num_samples // 1
        remainder = num_samples % 1
        parts = [base + 1 if i < remainder else base for i in range(1)]
        thetas=[]
        datas=[]
        for part in parts:
            print(part)
            data=SIR(config.data.julia_env_path).generate_dataset(key=subkey, n=part, misspecified=True,scale=False)
            key, subkey = random.split(key)
            theta=data['theta']
            data=data['y']
            print(data.shape)
            thetas.append(theta)
            datas.append(data)
        data=np.concatenate(datas,axis=0)
        theta=np.concatenate(thetas,axis=0)
        assert data.shape[-1]==config.data.vector_dim
        assert theta.shape[-1]==config.model.cond_dim
        data = np.concatenate((theta, data), axis=1)  # Concatenate theta and eps to form the data
        print(data.shape)


    if dataset_name=='gaussian_mixture':
        # Generate synthetic Gaussian data
        theta = np.random.uniform(-10,10, size=(num_samples, vector_dim))

        # Step 2: Sample component IDs (0 or 1)
        component_ids = np.random.choice([0, 0], size=num_samples)
        offsets = np.array([0, 0])  # Corresponding offsets for each component
        vars= np.array([np.sqrt(0.01),1])  # Corresponding variances for each component
        # Step 3: Compute component mean for each sample
        # Shape: (num_samples, cond_dim)
        component_offsets = offsets[component_ids][:, np.newaxis]  # Shape: (num_samples, 1)
        component_means = theta + component_offsets  # Add +/-2 to each dimension of theta

        # Step 4: Sample from N(mean, I)
        eps = np.random.normal(loc=0.0, scale=1.0, size=(num_samples, vector_dim))
 
        data = component_means + eps* vars[component_ids][:, np.newaxis] 

        # Step 5: Concatenate theta and data
        data = np.concatenate((theta, data), axis=1)
        print(data.shape)

    if dataset_name == 'gaussian_mixture_inference':
        # Step 1: Sample theta
        theta = np.random.uniform(-10,10, size=(num_samples, vector_dim))

        # Step 2: Sample component IDs (0 or 1)
        component_ids = np.random.choice([0, 1], size=num_samples)
        offsets = np.array([0, 0])  # Corresponding offsets for each component
        vars= np.array([np.sqrt(0.01),1])  # Corresponding variances for each component
        # Step 3: Compute component mean for each sample
        # Shape: (num_samples, cond_dim)
        component_offsets = offsets[component_ids][:, np.newaxis]  # Shape: (num_samples, 1)
        component_means = theta + component_offsets  # Add +/-2 to each dimension of theta

        # Step 4: Sample from N(mean, I)
        eps = np.random.normal(loc=0.0, scale=1.0, size=(num_samples, vector_dim))
 
        data = component_means + eps* vars[component_ids][:, np.newaxis] 

        # Step 5: Concatenate theta and data
        data = np.concatenate((theta, data), axis=1)
        print(data.shape)






    if dataset_name=='pendulum':
        # Generate synthetic Gaussian data

        num_samples = num_samples  # number of data points to generate
        num_timesteps = 200#config.model.cond_dim
        time_duration = 10  # seconds

        # Time vector
        t = np.linspace(0, time_duration, num_timesteps)[np.newaxis, :]  # shape: (1, 200)

        # Step 1: Sample θ = [ω₀, A] and φ
        omega0 = np.random.uniform(0.5, 3.0, size=(num_samples, 1))  # ω₀ ∈ [0, 3]
        A = np.random.uniform(0.5, 10.0, size=(num_samples, 1))      # A ∈ [0.5, 10]
        phi = np.random.uniform(-np.pi, np.pi, size=(num_samples, 1))  # φ ∈ [−π, π]

        # Step 2: Compute θ(t) = A * cos(ω₀ * t + φ)
        theta_t = A * np.cos(omega0 * t + phi)  # in radians

        # Step 3: Sample noise
        eps = 0.1*np.random.normal(loc=0.0, scale=1.0, size=(num_samples, num_timesteps))
        data = theta_t + eps 

        theta_params = np.concatenate((omega0, A), axis=1)  # shape: (num_samples, 2)
        data = np.concatenate((theta_params, data), axis=1)  # shape: (num_samples, 2 + 200 + 200) = (num_samples, 402)
        print(data.shape)


    if dataset_name == 'pendulum_inference':
        # Step 1: Sample theta
        num_samples = num_samples  # number of data points to generate
        num_timesteps = 200#config.model.cond_dim
        time_duration = 12  # seconds

        # Time vector
        t = np.linspace(0, time_duration, num_timesteps)[np.newaxis, :]  # shape: (1, 200)
        omega0 = []
        A = []
        phi = []
        eps = []

        for i in range(num_samples):
            rng = np.random.default_rng(seed=42 + i)  # or any fixed base seed

            omega0.append(rng.uniform(0.5, 3.0))
            A.append(rng.uniform(0.5, 10.0))
            phi.append(rng.uniform(-np.pi, np.pi))
            eps.append(0.01 * rng.normal(loc=0.0, scale=1.0, size=(num_timesteps,)))

        omega0 = np.array(omega0).reshape(-1, 1)
        A = np.array(A).reshape(-1, 1)
        phi = np.array(phi).reshape(-1, 1)
        eps = np.array(eps)  # shape: (num_samples, num_timesteps)
        # Step 2: Compute θ(t) = A * cos(ω₀ * t + φ)
        theta_t =A * np.cos(omega0 * t + phi)  # in radians

        # Step 3: Sample noise
        eps = 0.1*np.random.normal(loc=0.0, scale=1.0, size=(num_samples, num_timesteps))
        data = theta_t + eps 

        theta_params = np.concatenate((omega0, A), axis=1)  # shape: (num_samples, 2)
        data = np.concatenate((theta_params, data), axis=1)  # shape: (num_samples, 2 + 200 + 200) = (num_samples, 402)
        print(data.shape)





    if dataset_name=='spectra':
        #data=np.load('Data/isochrone_informed_data_raw_mag.npz',allow_pickle=True)
        data=np.load('Data/isochrone_informed_data_raw_col.npz',allow_pickle=True)

        thetas=data['features'][:,2:5]
        indices=np.where((thetas[:,2]<5.1) * (thetas[:,2]>4.0))
        thetas=thetas[indices]
        fluxes = data['fluxes'][:, 1:]
        fluxes=fluxes[indices]
        print(fluxes.shape)
        fluxes #+= noise
        print(fluxes.shape)
        np.random.seed(42)  # for reproducibility
        shuffled_indices = np.random.permutation(thetas.shape[0])
        data = np.concatenate((thetas[shuffled_indices][:], fluxes[shuffled_indices][:]), axis=-1)
        

    if dataset_name=='spectra_inference':
        data=np.load('Data/bprp_spectra_data.npz',allow_pickle=True)
        thetas=data['features'][:,:3]
        fluxes=data['fluxes'][:,:]

        np.random.seed(42)  # for reproducibility
        shuffled_indices = np.random.permutation(thetas.shape[0])
        data = np.concatenate((thetas[shuffled_indices][:], fluxes[shuffled_indices][:]), axis=-1)[:num_samples]
        print(data.shape)

    return data




def generate_sbibm_data(config):
    """Generate data using SBIBM task and simulator, with conditional prior sampling."""
    task = sbibm.get_task(config.data.task)
    print(task)
    if config.data.task_prior:  # Use task prior if the config specifies so
        prior = task.get_prior()
        simulator = task.get_simulator()
        theta = prior(num_samples=config.data.num_simulations)
        data = simulator(theta).numpy()
    
    else: 
        # Define a custom prior or uniform prior if task_prior is False
        # For simplicity, define a uniform prior over a custom range for theta
        reference_samples = np.concatenate([task.get_reference_posterior_samples(num_observation=_).numpy() for _ in range(1, 11)])
        
        low_bounds = jax.nn.relu(reference_samples.min(0) - 0.05)
        high_bounds = jax.nn.relu(reference_samples.max(0) + 0.05)
        theta = np.random.uniform(low=low_bounds, high=high_bounds, size=(config.data.num_simulations, 4))        
        data = task.get_simulator()(theta).numpy()  # Simulate data based on sampled theta
    
    # Concatenate theta and data for the final dataset
    return np.concatenate((theta, data), axis=1)





def preprocess_and_batch(
    dataset: jax.Array,      # Single JAX array (e.g., shape [num_samples, ...])
    batch_dims: Tuple[int, ...],  # E.g., (32, 256) for 2D batching
    key: Optional[jax.random.PRNGKey] = None,  # Optional for shuffling
    inference: bool = False,
) -> jax.Array:
    """
    JAX-compatible batching for a single array dataset.
    
    Args:
        dataset: Input JAX array of shape (num_samples, ...)
        batch_dims: Tuple of batch sizes (outermost first)
        key: PRNG key for shuffling (ignored if inference=True)
        inference: If True, skip shuffling
        
    Returns:
        Batched array with shape (batch_dim1, batch_dim2, ...)
    """
    # --- Step 1: Shuffle (if not inference) ---
    if inference:
      return dataset
    
    if not inference and key is not None:
        dataset = jax.random.permutation(key, dataset, axis=0)

    # --- Step 2: Batch along each dimension ---
    for batch_size in reversed(batch_dims):
        num_samples = dataset.shape[0]
        num_batches = num_samples // batch_size
        dataset = dataset[:num_batches * batch_size]  # Trim if needed
        dataset = dataset.reshape(num_batches, batch_size, *dataset.shape[1:])

    return dataset