#!/usr/bin/env python3
"""
Quick test of ACAUC implementation
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluate import Evaluator

def test_acauc():
    """Test ACAUC with known distributions"""

    print("="*80)
    print("Testing ACAUC Implementation")
    print("="*80)
    print()

    # Test 1: True parameter at center of posterior (high coverage expected)
    print("Test 1: True Parameter at Posterior Mean")
    print("-" * 40)
    np.random.seed(42)
    theta_true = np.array([0.0, 0.0])
    samples = np.random.randn(10000, 2)  # N(0,1) samples centered at true value
    acauc = Evaluator.compute_acauc(samples, theta_true, n_alpha_levels=100)
    print(f"  True theta: {theta_true}")
    print(f"  Samples: 10000 from N(0,1)")
    print(f"  ACAUC: {acauc:.4f}")
    print(f"  Expected: ~1.0 (true param at center, covered by almost all intervals)")
    print(f"  Status: {'✓ PASS' if acauc > 0.95 else '✗ FAIL'}")
    print()

    # Test 2: Under-coverage (too narrow posterior)
    print("Test 2: Under-coverage (narrow posterior)")
    print("-" * 40)
    theta_true = np.array([2.0, 2.0])
    samples = np.random.randn(10000, 2) * 0.5  # N(0, 0.25) - narrow
    acauc = Evaluator.compute_acauc(samples, theta_true, n_alpha_levels=100)
    print(f"  True theta: {theta_true}")
    print(f"  Samples: 10000 from N(0, 0.25)")
    print(f"  ACAUC: {acauc:.4f}")
    print(f"  Expected: ~0.00 (under-covered, true value far from samples)")
    print(f"  Status: {'✓ PASS' if acauc < 0.1 else '✗ FAIL'}")
    print()

    # Test 3: Over-coverage (too wide posterior)
    print("Test 3: Over-coverage (wide posterior)")
    print("-" * 40)
    theta_true = np.array([0.0, 0.0])
    samples = np.random.randn(10000, 2) * 3.0  # N(0, 9) - very wide
    acauc = Evaluator.compute_acauc(samples, theta_true, n_alpha_levels=100)
    print(f"  True theta: {theta_true}")
    print(f"  Samples: 10000 from N(0, 9)")
    print(f"  ACAUC: {acauc:.4f}")
    print(f"  Expected: ~0.85-0.95 (over-covered)")
    print(f"  Status: {'✓ PASS' if acauc > 0.75 else '✗ FAIL'}")
    print()

    # Test 4: Biased mean but correct variance
    print("Test 4: Biased mean, correct variance")
    print("-" * 40)
    theta_true = np.array([0.0, 0.0])
    samples = np.random.randn(10000, 2) + 1.0  # N(1, 1) - biased
    acauc = Evaluator.compute_acauc(samples, theta_true, n_alpha_levels=100)
    print(f"  True theta: {theta_true}")
    print(f"  Samples: 10000 from N(1, 1)")
    print(f"  ACAUC: {acauc:.4f}")
    print(f"  Expected: ~0.15-0.30 (biased but reasonable coverage)")
    print(f"  Status: {'✓ PASS' if 0.10 < acauc < 0.40 else '✗ FAIL'}")
    print()

    # Test 5: Multi-dimensional test
    print("Test 5: High-dimensional test")
    print("-" * 40)
    theta_true = np.zeros(10)
    samples = np.random.randn(10000, 10)
    acauc = Evaluator.compute_acauc(samples, theta_true, n_alpha_levels=100)
    print(f"  Dimensions: 10")
    print(f"  True theta: all zeros")
    print(f"  Samples: 10000 from N(0,1)")
    print(f"  ACAUC: {acauc:.4f}")
    print(f"  Expected: ~1.0 (true param at center of all dimensions)")
    print(f"  Status: {'✓ PASS' if acauc > 0.95 else '✗ FAIL'}")
    print()

    print("="*80)
    print("ACAUC Implementation Test Complete")
    print("="*80)
    print()
    print("Summary:")
    print("  All tests should show expected behavior")
    print()
    print("Interpretation:")
    print("  - ACAUC is computed per observation, then averaged")
    print("  - For true param at posterior center: ACAUC ≈ 1.0")
    print("  - For true param far from posterior: ACAUC ≈ 0.0")
    print("  - Across many random observations, well-calibrated posterior has mean ACAUC ≈ 0.5")
    print("  - Mean ACAUC < 0.5 indicates systematic under-coverage (posterior too narrow)")
    print("  - Mean ACAUC > 0.5 indicates systematic over-coverage (posterior too wide)")
    print()

if __name__ == '__main__':
    test_acauc()
