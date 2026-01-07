#!/usr/bin/env python3
"""
Verify all config files are correct
"""

import os
import sys
import importlib.util
from pathlib import Path
from typing import Dict, List

def load_config(config_path: str):
    """Load config from file"""
    spec = importlib.util.spec_from_file_location("config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()

def verify_config(config_path: str) -> Dict:
    """Verify a single config file"""
    results = {
        'path': config_path,
        'status': 'OK',
        'errors': [],
        'warnings': [],
        'info': {}
    }

    try:
        config = load_config(config_path)

        # Extract key info
        results['info']['model_name'] = config.model.name
        results['info']['correction_type'] = config.model.correction_type
        results['info']['task'] = config.data.dataset
        results['info']['num_tests'] = config.data.num_tests
        results['info']['flow_dimension'] = config.model.flow_dimension
        results['info']['cond_dim'] = config.model.cond_dim

        # Check required fields
        required_fields = [
            'model.name',
            'model.correction_type',
            'data.dataset',
            'data.num_tests',
            'model.flow_dimension',
            'model.cond_dim',
            'training.workspace',
        ]

        for field in required_fields:
            parts = field.split('.')
            obj = config
            try:
                for part in parts:
                    obj = getattr(obj, part)
            except AttributeError:
                results['errors'].append(f"Missing required field: {field}")
                results['status'] = 'ERROR'

        # Check correction_type matches filename
        filename = Path(config_path).name
        if 'mu_hybrid' in filename:
            if config.model.correction_type != 'mu_hybrid':
                results['errors'].append(
                    f"Filename contains 'mu_hybrid' but correction_type='{config.model.correction_type}'"
                )
                results['status'] = 'ERROR'
        elif 'simple' in filename:
            if config.model.correction_type != 'simple':
                results['warnings'].append(
                    f"Filename contains 'simple' but correction_type='{config.model.correction_type}'"
                )
        elif 'npe' in filename:
            if config.model.name != 'npe':
                results['errors'].append(
                    f"Filename contains 'npe' but model.name='{config.model.name}'"
                )
                results['status'] = 'ERROR'

        # Check model.name vs correction_type consistency
        if config.model.name == 'ranpt':
            if config.model.correction_type not in ['simple', 'mu_hybrid', 'hybrid', 'diagonal_neural', 'neural']:
                results['errors'].append(
                    f"RANPT model has invalid correction_type: '{config.model.correction_type}'"
                )
                results['status'] = 'ERROR'
        elif config.model.name == 'npe':
            if config.model.correction_type not in ['none', None]:
                results['warnings'].append(
                    f"NPE model should have correction_type='none', got '{config.model.correction_type}'"
                )

        # Check workspace matches config
        workspace = config.training.workspace
        if str(config.data.num_tests) not in workspace:
            results['warnings'].append(
                f"Workspace '{workspace}' doesn't contain num_tests={config.data.num_tests}"
            )

        # Check for common issues
        if config.model.correction_type == 'hybrid':
            results['warnings'].append(
                "Using 'hybrid' instead of 'mu_hybrid' - are you sure? 'hybrid' only has neural covariance, no neural mean"
            )

    except Exception as e:
        results['status'] = 'ERROR'
        results['errors'].append(f"Failed to load config: {str(e)}")

    return results

def main():
    """Main verification function"""
    print("="*80)
    print("CONFIG FILE VERIFICATION")
    print("="*80)
    print()

    # Find all config files
    tasks = ['CS_task', 'SIR', 'Pendulum', 'Spectra']
    methods = ['mu_hybrid', 'simple', 'npe']
    nobs_values = [1, 10, 100, 1000, 10000]

    all_results = []

    for task in tasks:
        task_dir = f'configs/{task}'
        if not os.path.exists(task_dir):
            print(f"⚠ Task directory not found: {task_dir}")
            continue

        print(f"\n{'='*80}")
        print(f"Task: {task}")
        print(f"{'='*80}\n")

        # Check for expected configs
        for method in methods:
            for nobs in nobs_values:
                if task == 'Spectra' and nobs > 100:
                    continue  # Spectra only has 1, 10, 100

                if method == 'npe':
                    config_file = f"{task_dir}/npe_{task.lower().replace('_task', '')}_task.py"
                    # NPE doesn't vary by nobs in filename
                    if nobs != nobs_values[0]:
                        continue
                else:
                    config_file = f"{task_dir}/ranpt_{nobs}_{method}.py"

                if os.path.exists(config_file):
                    result = verify_config(config_file)
                    all_results.append(result)

                    # Print result
                    status_symbol = {
                        'OK': '✓',
                        'ERROR': '✗',
                        'WARNING': '⚠'
                    }.get(result['status'], '?')

                    print(f"{status_symbol} {Path(config_file).name}")
                    print(f"   Model: {result['info'].get('model_name', 'N/A')}")
                    print(f"   Correction: {result['info'].get('correction_type', 'N/A')}")
                    print(f"   Nobs: {result['info'].get('num_tests', 'N/A')}")
                    print(f"   Flow dim: {result['info'].get('flow_dimension', 'N/A')}, Cond dim: {result['info'].get('cond_dim', 'N/A')}")

                    if result['errors']:
                        for error in result['errors']:
                            print(f"   ERROR: {error}")

                    if result['warnings']:
                        for warning in result['warnings']:
                            print(f"   WARNING: {warning}")

                    print()
                else:
                    print(f"✗ MISSING: {config_file}")
                    print()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    total = len(all_results)
    ok = sum(1 for r in all_results if r['status'] == 'OK')
    errors = sum(1 for r in all_results if r['status'] == 'ERROR')
    warnings = sum(1 for r in all_results if r['status'] == 'WARNING' or r['warnings'])

    print(f"Total configs checked: {total}")
    print(f"✓ OK: {ok}")
    print(f"✗ Errors: {errors}")
    print(f"⚠ Warnings: {warnings}")
    print()

    if errors > 0:
        print("❌ VERIFICATION FAILED - Please fix errors above")
        return 1
    elif warnings > 0:
        print("⚠ VERIFICATION PASSED WITH WARNINGS - Review warnings above")
        return 0
    else:
        print("✅ ALL CONFIGS VERIFIED SUCCESSFULLY")
        return 0

if __name__ == '__main__':
    sys.exit(main())
