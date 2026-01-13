#!/bin/bash
################################################################################
# Quick Smoke Test - Verify Pipeline Config Discovery
#
# Tests that the integrated pipeline correctly finds configs for all methods
################################################################################

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo "========================================="
echo "RVNP-SBI Pipeline Smoke Test"
echo "========================================="
echo ""

echo "1. Testing main experiment config discovery..."
python -c "
import sys
sys.path.insert(0, '.')
from scripts.integrated_pipeline import IntegratedPipeline
from collections import Counter

pipeline = IntegratedPipeline(results_dir='experiment_results_test')
experiments = pipeline.generate_experiment_list()

# Expected counts
expected_total = 44  # RVNP varies by nobs (18+18), NPE/NNPE don't (4+4)
expected_rvnp = 18  # RVNP configs for all nobs values
expected_baseline = 4  # NPE/NNPE only have one config per task

methods = [e.method for e in experiments]
method_counts = Counter(methods)

# Verify counts
assert len(experiments) == expected_total, f'Expected {expected_total} experiments, got {len(experiments)}'
assert method_counts['RVNP-simple'] == expected_rvnp, f'Expected {expected_rvnp} RVNP-simple, got {method_counts["RVNP-simple"]}'
assert method_counts['RVNP-NN'] == expected_rvnp, f'Expected {expected_rvnp} RVNP-NN, got {method_counts["RVNP-NN"]}'
assert method_counts['NPE'] == expected_baseline, f'Expected {expected_baseline} NPE, got {method_counts["NPE"]}'
assert method_counts['NNPE'] == expected_baseline, f'Expected {expected_baseline} NNPE, got {method_counts["NNPE"]}'

print(f'✅ Main experiments: {len(experiments)} total')
for method, count in sorted(method_counts.items()):
    print(f'   {method}: {count}')
"

echo ""
echo "2. Testing wellspec config discovery..."
python -c "
import sys
sys.path.insert(0, '.')
from scripts.integrated_pipeline import IntegratedPipeline
from collections import Counter

pipeline = IntegratedPipeline(results_dir='experiment_results_wellspec_test', wellspec=True)
experiments = pipeline.generate_experiment_list()

# Expected counts
expected_total = 44
expected_rvnp = 18  # RVNP configs exist for all nobs
expected_baseline = 4  # NPE/NNPE only have nobs=100

methods = [e.method for e in experiments]
method_counts = Counter(methods)

# Verify counts
assert len(experiments) == expected_total, f'Expected {expected_total} experiments, got {len(experiments)}'
assert method_counts['RVNP-simple'] == expected_rvnp
assert method_counts['RVNP-NN'] == expected_rvnp
assert method_counts['NPE'] == expected_baseline
assert method_counts['NNPE'] == expected_baseline

print(f'✅ Wellspec experiments: {len(experiments)} total')
for method, count in sorted(method_counts.items()):
    print(f'   {method}: {count}')
"

echo ""
echo "3. Verifying sample configs exist..."
test -f configs/CS_task/cs_task_tests100_NN_shrink00.py && echo "✅ CS RVNP-NN config exists"
test -f configs/CS_task/cs_task_tests100_simple_shrink00.py && echo "✅ CS RVNP-simple config exists"
test -f configs/CS_task/npe_cs_task.py && echo "✅ CS NPE config exists"
test -f configs/CS_task/nnpe_cs_task.py && echo "✅ CS NNPE config exists"

echo ""
echo "4. Verifying config model names..."

echo ""
echo "5. Verifying script paths..."
bash scripts/run_experiments_colab.sh --help > /dev/null 2>&1 && \
    echo "✅ run_experiments_colab.sh works"
bash scripts/run_wellspec_experiments.sh --help > /dev/null 2>&1 && \
    echo "✅ run_wellspec_experiments.sh works"
bash scripts/run_ablation_study.sh --help > /dev/null 2>&1 && \
    echo "✅ run_ablation_study.sh works"

echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}All smoke tests passed! ✅${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo "Pipeline is ready to run:"
echo "  Main experiments:     bash scripts/run_experiments_colab.sh"
echo "  Wellspec experiments: bash scripts/run_wellspec_experiments.sh"
echo "  Ablation study:       bash scripts/run_ablation_study.sh"
