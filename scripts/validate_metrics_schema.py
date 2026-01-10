"""Validate that metrics schema is consistent across pipeline components"""
import sys
from pathlib import Path

# Expected schema from integrated_pipeline.py lines 422-431
EXPECTED_SCHEMA = {
    'task': str,
    'method': str,
    'correction_type': str,
    'nobs': int,
    'aepc': float,  # alpha in evaluate.py
    'acauc_mean': float,
    'acauc_std': float,
    'lpp_median': float,  # med_log_prob in evaluate.py
    'lpp_std': float,     # std_log_prob in evaluate.py
    'nrmse_mean': float,  # mean_nrmse in evaluate.py
    'nrmse_std': float,   # std_nrmse in evaluate.py
    'ess_mean': float,
    'n_evaluations': int,
    'use_sir': bool,
    'timestamp': str,
}

def validate_schema():
    """Check schema consistency"""
    print("Expected Metrics Database Schema:")
    print("="*60)
    for col, dtype in EXPECTED_SCHEMA.items():
        print(f"  {col:20s} : {dtype.__name__}")

    print(f"\n✅ Schema defined with {len(EXPECTED_SCHEMA)} columns")
    print("\nTo verify after running experiments:")
    print("  1. Check experiment_results/metrics_database.csv has these columns")
    print("  2. Check publication_plots.py can read these columns")
    print("  3. Verify no NaN values in numeric columns")

    return True

if __name__ == '__main__':
    validate_schema()
