#!/bin/bash

# SIR Task Runner - runs only remaining missing configs
# No complex logging, just straightforward execution

echo "=== SIR Task Configuration Runner (Final Missing Configs) ==="
echo "Started at: $(date)"
echo ""

# Change to project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "Running from: $PROJECT_ROOT"
echo ""

# Create logs directory in SIR folder
LOGS_DIR="configs/SIR/logs"
mkdir -p "$LOGS_DIR"
echo "Logs will be saved to: $LOGS_DIR"
echo ""

# 1. Run base config first
echo "[1/4] Running sir_task.py..."
python main_train_eval.py --config=configs/SIR/sir_task.py --mode=train > "$LOGS_DIR/sir_task.log" 2>&1
echo "✓ sir_task.py completed (log: $LOGS_DIR/sir_task.log)"
echo ""

# 2. Run NNPE config second  
echo "[2/4] Running nnpe_sir_task.py..."
python main_train_eval.py --config=configs/SIR/nnpe_sir_task.py --mode=train > "$LOGS_DIR/nnpe_sir_task.log" 2>&1
echo "✓ nnpe_sir_task.py completed (log: $LOGS_DIR/nnpe_sir_task.log)"
echo ""

# 3. Run only remaining missing test configs (tests1000 only)
echo "[3/4] Running sir_task_tests1000_simple_shrink00.py..."
python main_train_eval.py --config=configs/SIR/sir_task_tests1000_simple_shrink00.py --mode=train > "$LOGS_DIR/sir_task_tests1000_simple_shrink00.log" 2>&1
echo "✓ sir_task_tests1000_simple_shrink00.py completed (log: $LOGS_DIR/sir_task_tests1000_simple_shrink00.log)"
echo ""

echo "[4/4] Running sir_task_tests1000_hybrid_shrink00.py..."
python main_train_eval.py --config=configs/SIR/sir_task_tests1000_hybrid_shrink00.py --mode=train > "$LOGS_DIR/sir_task_tests1000_hybrid_shrink00.log" 2>&1
echo "✓ sir_task_tests1000_hybrid_shrink00.py completed (log: $LOGS_DIR/sir_task_tests1000_hybrid_shrink00.log)"
echo ""

echo "=== ALL DONE ==="
echo "All 4 remaining SIR Task configurations have been run!"
echo "Note: Skipped already trained configs (tests1, tests10, tests100) to save time"