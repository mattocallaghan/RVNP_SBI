"""Test that all expected config files can be discovered by integrated_pipeline.py"""
import os
import sys
from pathlib import Path

# Expected configs
EXPECTED_CONFIGS = {
    'main': [
        # CS task (11 configs: 5 simple + 5 mu_hybrid + NPE)
        *[f'configs/CS_task/ranpt_{n}_simple.py' for n in [1, 10, 100, 1000, 10000]],
        *[f'configs/CS_task/ranpt_{n}_mu_hybrid.py' for n in [1, 10, 100, 1000, 10000]],
        'configs/CS_task/npe_cs_task.py',
        # SIR task (11 configs)
        *[f'configs/SIR/ranpt_{n}_simple.py' for n in [1, 10, 100, 1000, 10000]],
        *[f'configs/SIR/ranpt_{n}_mu_hybrid.py' for n in [1, 10, 100, 1000, 10000]],
        'configs/SIR/npe_sir_task.py',
        # Pendulum task (11 configs)
        *[f'configs/Pendulum/ranpt_{n}_simple.py' for n in [1, 10, 100, 1000, 10000]],
        *[f'configs/Pendulum/ranpt_{n}_mu_hybrid.py' for n in [1, 10, 100, 1000, 10000]],
        'configs/Pendulum/npe_pendulum_task.py',
        # Spectra task (7 configs: 3 simple + 3 mu_hybrid + NPE)
        *[f'configs/Spectra/ranpt_{n}_simple.py' for n in [1, 10, 100]],
        *[f'configs/Spectra/ranpt_{n}_mu_hybrid.py' for n in [1, 10, 100]],
        'configs/Spectra/npe_spectra_task.py',
    ],
    'ablation': [
        *[f'configs/CS_task/ablation_params_{s}.py' for s in ['tiny', 'small', 'medium', 'large', 'huge']],
        *[f'configs/CS_task/ablation_shrink_{s}.py' for s in ['0', 'weak', 'medium', 'strong', 'verystrong']],
    ]
}

def check_configs():
    """Check all expected configs exist"""
    missing = []
    found = []

    for category, configs in EXPECTED_CONFIGS.items():
        print(f"\nChecking {category} configs...")
        for config_path in configs:
            if os.path.exists(config_path):
                found.append(config_path)
                print(f"  ✓ {config_path}")
            else:
                missing.append(config_path)
                print(f"  ✗ {config_path} - MISSING")

    print(f"\n{'='*60}")
    print(f"Summary: {len(found)}/{len(found) + len(missing)} configs found")

    if missing:
        print(f"\n❌ {len(missing)} configs missing:")
        for path in missing:
            print(f"  - {path}")
        return False
    else:
        print("\n✅ All configs found!")
        return True

if __name__ == '__main__':
    success = check_configs()
    sys.exit(0 if success else 1)
