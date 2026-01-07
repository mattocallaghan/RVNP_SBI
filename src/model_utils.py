"""
Model utilities for parameter counting and analysis
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Dict, Any
import json


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
