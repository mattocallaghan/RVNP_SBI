"""
Model utilities for parameter counting, analysis, and model creation
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from typing import Dict, Any, Tuple, Optional
import json

from .models.correction_model import (
    SimpleCorrectionModel,
    DiagonalNeuralCorrectionModel,
    HybridCorrectionModel,
    FullNeuralCorrectionModel,
    MuHybridCorrectionModel,
    GlobalCorrectionModel,
    CorrectionModel
)
from .models.embeddings import (
    StatisticEmbedding_spectra,
    StatisticEmbedding_pendulum,
    Discriminator,
    ReconstructionDecoder,
    ReconstructionDecoderSimple
)
from .losses import ShannonLossEmbedding


def create_correction_model(key, config, correction_dim: int, flow_dimension: int):
    """
    Factory function to create correction model based on config.

    Args:
        key: JAX random key
        config: Configuration object with model settings
        correction_dim: Dimension of correction model output (embedding_dim or vector_dim)
        flow_dimension: Dimension of theta parameters

    Returns:
        Initialized correction model instance
    """
    correction_type = getattr(config.model, 'correction_type', 'simple')

    if correction_type == 'simple':
        # Use default initialization: L[i,i] = 1.0 (unit variance)
        # No initial_covariance passed - model will initialize with log(1.0) = 0.0 in log-space
        return SimpleCorrectionModel(
            key=key,
            dim=correction_dim
        )

    elif correction_type == 'diagonal_neural':
        return DiagonalNeuralCorrectionModel(
            key=key,
            theta_dim=flow_dimension,
            output_dim=correction_dim
        )

    elif correction_type == 'hybrid':
        return HybridCorrectionModel(
            key=key,
            theta_dim=flow_dimension,
            output_dim=correction_dim
        )

    elif correction_type == 'NN':
        return MuHybridCorrectionModel(
            key=key,
            theta_dim=flow_dimension,
            output_dim=correction_dim
        )

    elif correction_type == 'global':
        return GlobalCorrectionModel(
            key=key,
            output_dim=correction_dim
        )

    elif correction_type == 'full_neural':
        return FullNeuralCorrectionModel(
            key=key,
            theta_dim=flow_dimension,
            output_dim=correction_dim
        )

    else:
        return CorrectionModel(
            key=key,
            input_dim=correction_dim
        )


def create_embedding_models(key, config, embedding_dim: int, cond_dim: int) -> Tuple[eqx.Module, Optional[eqx.Module], Optional[eqx.Module], Optional[Any]]:
    """
    Factory function to create embedding, discriminator, decoder, and embedding loss.

    Args:
        key: JAX random key
        config: Configuration object with model settings
        embedding_dim: Dimension of embedding output
        cond_dim: Dimension of conditioning (flow_dimension)

    Returns:
        Tuple of (embedding, discriminator, decoder, embedding_loss_function)
        Returns (None, None, None, None) if embeddings not needed
    """
    dataset = config.data.dataset

    # Only create embeddings for these datasets
    if dataset not in ['spectra', 'pendulum']:
        return None, None, None, None

    key, key_emb = jr.split(key)
    embedding_type = config.model.embedding
    hidden_scale = config.model.hidden_scale

    # Create embedding network
    if dataset == 'spectra':
        embedding = StatisticEmbedding_spectra(
            key=key_emb,
            in_channels=1,
            how=embedding_type,
            hidden_scale=hidden_scale,
            z_dim=embedding_dim,
            dropout_rate=0.1
        )
    else:  # pendulum
        embedding = StatisticEmbedding_pendulum(
            key=key_emb,
            in_channels=1,
            how=embedding_type,
            hidden_scale=hidden_scale,
            z_dim=embedding_dim,
            dropout_rate=0.1
        )

    # Handle VAE vs InfoMax
    if embedding_type == 'vae':
        raise NotImplementedError(
            "VAE embedding type is not yet implemented. "
            "Use embedding_type='IM' for InfoMax (mutual information) embedding instead."
        )

    assert embedding_type == 'IM', "Embedding type must be 'vae' or 'IM'"

    # Create discriminator and decoder for InfoMax
    hidden_dim = 100
    key, key_disc, key_dec = jr.split(key, 3)

    discriminator = Discriminator(
        key=key_disc,
        z_dim=embedding_dim,
        theta_dim=cond_dim,
        hidden_dim=hidden_dim
    )

    # Determine output shape for decoder
    if dataset == 'pendulum':
        output_shape = config.data.vector_dim_inference  # 200 for pendulum
    elif dataset == 'spectra':
        output_shape = config.data.vector_dim  # 300 for spectra
    else:
        output_shape = config.data.vector_dim

    print(f"Using output shape for decoder: {output_shape}")

    # Choose decoder type based on dataset
    if dataset == 'pendulum':
        decoder = ReconstructionDecoderSimple(
            key=key_dec,
            latent_dim=embedding_dim,
            out_channels=1,
            output_shape=output_shape
        )
        print("Using ReconstructionDecoderSimple for pendulum dataset")
    else:
        decoder = ReconstructionDecoder(
            key=key_dec,
            latent_dim=embedding_dim,
            out_channels=1,
            output_shape=output_shape
        )
        print(f"Using ReconstructionDecoder with final linear layer for {dataset}")

    # Set reconstruction loss parameters
    if dataset == 'spectra':
        lambda_reconstruction = 0.0
        use_decoder = False
    else:
        lambda_reconstruction = 0.0
        use_decoder = False

    embedding_loss_function = ShannonLossEmbedding(
        lambda_reconstruction=lambda_reconstruction,
        use_decoder=use_decoder
    )

    return embedding, discriminator, decoder, embedding_loss_function


def count_parameters(model: eqx.Module) -> int:
    """
    Count total number of trainable parameters in an Equinox model.

    Args:
        model: Equinox module

    Returns:
        Total number of parameters
    """
    params, _ = eqx.partition(model, eqx.is_array)
    return sum(p.size for p in jax.tree_util.tree_leaves(params))


def get_parameter_breakdown(model: eqx.Module, name: str = "model") -> Dict[str, Any]:
    """
    Get detailed parameter breakdown for an Equinox model.

    Args:
        model: Equinox module
        name: Name of the model

    Returns:
        Dictionary with parameter counts and breakdown
    """
    params, _ = eqx.partition(model, eqx.is_array)

    breakdown = {
        'total_params': 0,
        'components': {}
    }

    # Count parameters per component
    for key, value in model.__dict__.items():
        if isinstance(value, eqx.Module):
            # Recursively count parameters in submodules
            component_params = count_parameters(value)
            if component_params > 0:
                breakdown['components'][key] = {
                    'params': component_params,
                    'type': type(value).__name__
                }
                breakdown['total_params'] += component_params
        elif isinstance(value, (jnp.ndarray, jax.Array)):
            # Direct array parameters
            param_count = value.size
            breakdown['components'][key] = {
                'params': param_count,
                'shape': tuple(value.shape),
                'type': 'Parameter'
            }
            breakdown['total_params'] += param_count

    breakdown['model_name'] = name
    breakdown['model_type'] = type(model).__name__

    return breakdown


def get_ranpt_model_breakdown(flowclass) -> Dict[str, Any]:
    """
    Get parameter breakdown for RANPT model with all components.

    Args:
        flowclass: RANPT flowclass instance

    Returns:
        Dictionary with detailed parameter breakdown
    """
    breakdown = {
        'total_params': 0,
        'models': {}
    }

    # Posterior flow
    if hasattr(flowclass, 'flow') and flowclass.flow is not None:
        flow_breakdown = get_parameter_breakdown(flowclass.flow, 'posterior_flow')
        breakdown['models']['posterior_flow'] = flow_breakdown
        breakdown['total_params'] += flow_breakdown['total_params']

    # Simulator flow
    if hasattr(flowclass, 'simulator_flow') and flowclass.simulator_flow is not None:
        sim_breakdown = get_parameter_breakdown(flowclass.simulator_flow, 'simulator_flow')
        breakdown['models']['simulator_flow'] = sim_breakdown
        breakdown['total_params'] += sim_breakdown['total_params']

    # Correction model
    if hasattr(flowclass, 'correction_model') and flowclass.correction_model is not None:
        corr_breakdown = get_parameter_breakdown(flowclass.correction_model, 'correction_model')
        breakdown['models']['correction_model'] = corr_breakdown
        breakdown['total_params'] += corr_breakdown['total_params']

    # Embedding (if exists)
    if hasattr(flowclass, 'embedding') and flowclass.embedding is not None:
        emb_breakdown = get_parameter_breakdown(flowclass.embedding, 'embedding')
        breakdown['models']['embedding'] = emb_breakdown
        breakdown['total_params'] += emb_breakdown['total_params']

    return breakdown


def print_parameter_breakdown(breakdown: Dict[str, Any], indent: int = 0):
    """
    Pretty print parameter breakdown.

    Args:
        breakdown: Parameter breakdown dictionary
        indent: Indentation level
    """
    prefix = "  " * indent

    if 'model_name' in breakdown:
        print(f"{prefix}{breakdown['model_name']} ({breakdown['model_type']})")
        print(f"{prefix}  Total parameters: {breakdown['total_params']:,}")

        if breakdown['components']:
            print(f"{prefix}  Components:")
            for comp_name, comp_info in breakdown['components'].items():
                if 'shape' in comp_info:
                    print(f"{prefix}    - {comp_name}: {comp_info['params']:,} params {comp_info['shape']}")
                else:
                    print(f"{prefix}    - {comp_name} ({comp_info['type']}): {comp_info['params']:,} params")

    elif 'models' in breakdown:
        print(f"{prefix}Full Model Breakdown")
        print(f"{prefix}Total parameters: {breakdown['total_params']:,}")
        print()

        for model_name, model_breakdown in breakdown['models'].items():
            print(f"{prefix}{model_name}:")
            print_parameter_breakdown(model_breakdown, indent + 1)
            print()


def save_parameter_breakdown(breakdown: Dict[str, Any], filepath: str):
    """
    Save parameter breakdown to JSON file.

    Args:
        breakdown: Parameter breakdown dictionary
        filepath: Path to save JSON file
    """
    # Convert to JSON-serializable format
    def convert_to_serializable(obj):
        if isinstance(obj, (jnp.ndarray, jax.Array)):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    serializable_breakdown = convert_to_serializable(breakdown)

    with open(filepath, 'w') as f:
        json.dump(serializable_breakdown, f, indent=2)

    print(f"Parameter breakdown saved to: {filepath}")


if __name__ == '__main__':
    print("Model utilities for parameter counting")
    print("Import this module to use parameter counting functions")
