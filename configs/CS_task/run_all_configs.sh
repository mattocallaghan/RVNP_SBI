#!/bin/bash

# CS Task Runner - runs hybrid configs only
# No complex logging, just straightforward execution

echo "=== CS Task Configuration Runner (Hybrid Only) ==="
echo "Started at: $(date)"
echo ""

# Change to project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "Running from: $PROJECT_ROOT"
echo ""

# Create logs directory in CS_task folder
LOGS_DIR="configs/CS_task/logs"
mkdir -p "$LOGS_DIR"
echo "Logs will be saved to: $LOGS_DIR"
echo ""

# 1. Run base config first
echo "[1/7] Running cs_task.py..."
python main_train_eval.py --config=configs/CS_task/cs_task.py --mode=train > "$LOGS_DIR/cs_task.log" 2>&1
echo "✓ cs_task.py completed (log: $LOGS_DIR/cs_task.log)"
echo ""

# 2. Run NNPE config second  
echo "[2/7] Running nnpe_cs_task.py..."
python main_train_eval.py --config=configs/CS_task/nnpe_cs_task.py --mode=train > "$LOGS_DIR/nnpe_cs_task.log" 2>&1
echo "✓ nnpe_cs_task.py completed (log: $LOGS_DIR/nnpe_cs_task.log)"
echo ""

# 3. Run hybrid test configs only (no shrinkage)
echo "[3/7] Running cs_task_tests1_hybrid_shrink00.py..."
python main_train_eval.py --config=configs/CS_task/cs_task_tests1_hybrid_shrink00.py --mode=train > "$LOGS_DIR/cs_task_tests1_hybrid_shrink00.log" 2>&1
echo "✓ cs_task_tests1_hybrid_shrink00.py completed (log: $LOGS_DIR/cs_task_tests1_hybrid_shrink00.log)"
echo ""

echo "[4/7] Running cs_task_tests5_hybrid_shrink00.py..."
python main_train_eval.py --config=configs/CS_task/cs_task_tests5_hybrid_shrink00.py --mode=train > "$LOGS_DIR/cs_task_tests5_hybrid_shrink00.log" 2>&1
echo "✓ cs_task_tests5_hybrid_shrink00.py completed (log: $LOGS_DIR/cs_task_tests5_hybrid_shrink00.log)"
echo ""

echo "[5/7] Running cs_task_tests10_hybrid_shrink00.py..."
python main_train_eval.py --config=configs/CS_task/cs_task_tests10_hybrid_shrink00.py --mode=train > "$LOGS_DIR/cs_task_tests10_hybrid_shrink00.log" 2>&1
echo "✓ cs_task_tests10_hybrid_shrink00.py completed (log: $LOGS_DIR/cs_task_tests10_hybrid_shrink00.log)"
echo ""

echo "[6/7] Running cs_task_tests100_hybrid_shrink00.py..."
python main_train_eval.py --config=configs/CS_task/cs_task_tests100_hybrid_shrink00.py --mode=train > "$LOGS_DIR/cs_task_tests100_hybrid_shrink00.log" 2>&1
echo "✓ cs_task_tests100_hybrid_shrink00.py completed (log: $LOGS_DIR/cs_task_tests100_hybrid_shrink00.log)"
echo ""

echo "[7/7] Running cs_task_tests1000_hybrid_shrink00.py..."
python main_train_eval.py --config=configs/CS_task/cs_task_tests1000_hybrid_shrink00.py --mode=train > "$LOGS_DIR/cs_task_tests1000_hybrid_shrink00.log" 2>&1
echo "✓ cs_task_tests1000_hybrid_shrink00.py completed (log: $LOGS_DIR/cs_task_tests1000_hybrid_shrink00.log)"
echo ""

echo "=== ALL DONE ==="
echo "Completed at: $(date)"
echo "All 7 CS Task configurations have been run!"