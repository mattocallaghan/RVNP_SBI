#!/bin/bash
# Quick smoke test: single minimal experiment (~5 min)

set -e  # Exit on error

echo "=========================================="
echo "RVNP-SBI Quick Smoke Test"
echo "=========================================="
echo ""

# Test config
CONFIG="configs/CS_task/ranpt_1_simple.py"
TEST_DIR="experiment_results_smoke_test"

# Clean previous test
rm -rf "$TEST_DIR" output/ranpt_cs_task_1_simple

echo "1. Testing training..."
python scripts/main_train_eval.py --config="$CONFIG" || {
    echo "❌ Training failed"
    exit 1
}
echo "✅ Training passed"

echo ""
echo "2. Testing evaluation..."
python scripts/evaluate.py \
    --config="$CONFIG" \
    --n_samples=1000 \
    --n_eval_points=10 \
    --save_dir="$TEST_DIR" || {
    echo "❌ Evaluation failed"
    exit 1
}
echo "✅ Evaluation passed"

echo ""
echo "3. Testing integrated pipeline..."
python scripts/integrated_pipeline.py \
    --task=CS --method=RVNP-simple --nobs=1 \
    --results-dir="$TEST_DIR" --no-sir || {
    echo "❌ Pipeline failed"
    exit 1
}
echo "✅ Pipeline passed"

echo ""
echo "4. Checking outputs..."
if [ -f "output/ranpt_cs_task_1_simple/checkpoints/checkpoint_posterior_0.eqx" ]; then
    echo "✅ Checkpoint exists"
else
    echo "❌ Checkpoint missing"
    exit 1
fi

if [ -f "$TEST_DIR/metrics_database.csv" ]; then
    echo "✅ Metrics database created"
    echo ""
    echo "Metrics database content:"
    cat "$TEST_DIR/metrics_database.csv"
else
    echo "❌ Metrics database missing"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ ALL TESTS PASSED"
echo "=========================================="
echo "Pipeline is ready for full experiments!"
