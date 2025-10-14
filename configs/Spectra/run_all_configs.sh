#!/bin/bash

# Simple Spectra Task Runner - runs all configs in explicit order
# No complex logging, just straightforward execution

echo "=== Spectra Task Configuration Runner (Simple) ==="
echo "Started at: $(date)"
echo ""

# Change to project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "Running from: $PROJECT_ROOT"
echo ""

# Create logs directory in Spectra folder
LOGS_DIR="configs/Spectra/logs"
mkdir -p "$LOGS_DIR"
echo "Logs will be saved to: $LOGS_DIR"
echo ""

# 1. Run base config first
echo "[1/22] Running spectra_task.py..."
python main_train_eval.py --config=configs/Spectra/spectra_task.py --mode=train > "$LOGS_DIR/spectra_task.log" 2>&1
echo "✓ spectra_task.py completed (log: $LOGS_DIR/spectra_task.log)"
echo ""

# 2. Run NNPE config second  
echo "[2/22] Running nnpe_spectra_task.py..."
python main_train_eval.py --config=configs/Spectra/nnpe_spectra_task.py --mode=train > "$LOGS_DIR/nnpe_spectra_task.log" 2>&1
echo "✓ nnpe_spectra_task.py completed (log: $LOGS_DIR/nnpe_spectra_task.log)"
echo ""

# 1. Run test config
echo "[3/22] Running spectra_task_tests1_simple_shrink00.py..."
python main_train_eval.py --config=configs/Spectra/spectra_task_tests1_simple_shrink00.py --mode=train > "$LOGS_DIR/spectra_task_tests1_simple_shrink00.log" 2>&1
echo "✓ spectra_task_tests1_simple_shrink00.py completed (log: $LOGS_DIR/spectra_task_tests1_simple_shrink00.log)"
echo ""

# 2. Run test config
echo "[4/22] Running spectra_task_tests1_simple_shrink10.py..."
python main_train_eval.py --config=configs/Spectra/spectra_task_tests1_simple_shrink10.py --mode=train > "$LOGS_DIR/spectra_task_tests1_simple_shrink10.log" 2>&1
echo "✓ spectra_task_tests1_simple_shrink10.py completed (log: $LOGS_DIR/spectra_task_tests1_simple_shrink10.log)"
echo ""

# 3. Run test config
echo "[5/22] Running spectra_task_tests1_hybrid_shrink00.py..."
python main_train_eval.py --config=configs/Spectra/spectra_task_tests1_hybrid_shrink00.py --mode=train > "$LOGS_DIR/spectra_task_tests1_hybrid_shrink00.log" 2>&1
echo "✓ spectra_task_tests1_hybrid_shrink00.py completed (log: $LOGS_DIR/spectra_task_tests1_hybrid_shrink00.log)"
echo ""

# 4. Run test config
echo "[6/22] Running spectra_task_tests1_hybrid_shrink10.py..."
python main_train_eval.py --config=configs/Spectra/spectra_task_tests1_hybrid_shrink10.py --mode=train > "$LOGS_DIR/spectra_task_tests1_hybrid_shrink10.log" 2>&1
echo "✓ spectra_task_tests1_hybrid_shrink10.py completed (log: $LOGS_DIR/spectra_task_tests1_hybrid_shrink10.log)"
echo ""

# 5. Run test config
echo "[7/22] Running spectra_task_tests5_simple_shrink00.py..."
python main_train_eval.py --config=configs/Spectra/spectra_task_tests5_simple_shrink00.py --mode=train > "$LOGS_DIR/spectra_task_tests5_simple_shrink00.log" 2>&1
echo "✓ spectra_task_tests5_simple_shrink00.py completed (log: $LOGS_DIR/spectra_task_tests5_simple_shrink00.log)"
echo ""

# 6. Run test config
echo "[8/22] Running spectra_task_tests5_simple_shrink10.py..."
python main_train_eval.py --config=configs/Spectra/spectra_task_tests5_simple_shrink10.py --mode=train > "$LOGS_DIR/spectra_task_tests5_simple_shrink10.log" 2>&1
echo "✓ spectra_task_tests5_simple_shrink10.py completed (log: $LOGS_DIR/spectra_task_tests5_simple_shrink10.log)"
echo ""

# 7. Run test config
echo "[9/22] Running spectra_task_tests5_hybrid_shrink00.py..."
python main_train_eval.py --config=configs/Spectra/spectra_task_tests5_hybrid_shrink00.py --mode=train > "$LOGS_DIR/spectra_task_tests5_hybrid_shrink00.log" 2>&1
echo "✓ spectra_task_tests5_hybrid_shrink00.py completed (log: $LOGS_DIR/spectra_task_tests5_hybrid_shrink00.log)"
echo ""

# 8. Run test config
echo "[10/22] Running spectra_task_tests5_hybrid_shrink10.py..."
python main_train_eval.py --config=configs/Spectra/spectra_task_tests5_hybrid_shrink10.py --mode=train > "$LOGS_DIR/spectra_task_tests5_hybrid_shrink10.log" 2>&1
echo "✓ spectra_task_tests5_hybrid_shrink10.py completed (log: $LOGS_DIR/spectra_task_tests5_hybrid_shrink10.log)"
echo ""

# 9. Run test config
echo "[11/22] Running spectra_task_tests10_simple_shrink00.py..."
python main_train_eval.py --config=configs/Spectra/spectra_task_tests10_simple_shrink00.py --mode=train > "$LOGS_DIR/spectra_task_tests10_simple_shrink00.log" 2>&1
echo "✓ spectra_task_tests10_simple_shrink00.py completed (log: $LOGS_DIR/spectra_task_tests10_simple_shrink00.log)"
echo ""

# 10. Run test config
echo "[12/22] Running spectra_task_tests10_simple_shrink10.py..."
python main_train_eval.py --config=configs/Spectra/spectra_task_tests10_simple_shrink10.py --mode=train > "$LOGS_DIR/spectra_task_tests10_simple_shrink10.log" 2>&1
echo "✓ spectra_task_tests10_simple_shrink10.py completed (log: $LOGS_DIR/spectra_task_tests10_simple_shrink10.log)"
echo ""

# 11. Run test config
echo "[13/22] Running spectra_task_tests10_hybrid_shrink00.py..."
python main_train_eval.py --config=configs/Spectra/spectra_task_tests10_hybrid_shrink00.py --mode=train > "$LOGS_DIR/spectra_task_tests10_hybrid_shrink00.log" 2>&1
echo "✓ spectra_task_tests10_hybrid_shrink00.py completed (log: $LOGS_DIR/spectra_task_tests10_hybrid_shrink00.log)"
echo ""

# 12. Run test config
echo "[14/22] Running spectra_task_tests10_hybrid_shrink10.py..."
python main_train_eval.py --config=configs/Spectra/spectra_task_tests10_hybrid_shrink10.py --mode=train > "$LOGS_DIR/spectra_task_tests10_hybrid_shrink10.log" 2>&1
echo "✓ spectra_task_tests10_hybrid_shrink10.py completed (log: $LOGS_DIR/spectra_task_tests10_hybrid_shrink10.log)"
echo ""

# 13. Run test config
echo "[15/22] Running spectra_task_tests100_simple_shrink00.py..."
python main_train_eval.py --config=configs/Spectra/spectra_task_tests100_simple_shrink00.py --mode=train > "$LOGS_DIR/spectra_task_tests100_simple_shrink00.log" 2>&1
echo "✓ spectra_task_tests100_simple_shrink00.py completed (log: $LOGS_DIR/spectra_task_tests100_simple_shrink00.log)"
echo ""

# 14. Run test config
echo "[16/22] Running spectra_task_tests100_simple_shrink10.py..."
python main_train_eval.py --config=configs/Spectra/spectra_task_tests100_simple_shrink10.py --mode=train > "$LOGS_DIR/spectra_task_tests100_simple_shrink10.log" 2>&1
echo "✓ spectra_task_tests100_simple_shrink10.py completed (log: $LOGS_DIR/spectra_task_tests100_simple_shrink10.log)"
echo ""

# 15. Run test config
echo "[17/22] Running spectra_task_tests100_hybrid_shrink00.py..."
python main_train_eval.py --config=configs/Spectra/spectra_task_tests100_hybrid_shrink00.py --mode=train > "$LOGS_DIR/spectra_task_tests100_hybrid_shrink00.log" 2>&1
echo "✓ spectra_task_tests100_hybrid_shrink00.py completed (log: $LOGS_DIR/spectra_task_tests100_hybrid_shrink00.log)"
echo ""

# 16. Run test config
echo "[18/22] Running spectra_task_tests100_hybrid_shrink10.py..."
python main_train_eval.py --config=configs/Spectra/spectra_task_tests100_hybrid_shrink10.py --mode=train > "$LOGS_DIR/spectra_task_tests100_hybrid_shrink10.log" 2>&1
echo "✓ spectra_task_tests100_hybrid_shrink10.py completed (log: $LOGS_DIR/spectra_task_tests100_hybrid_shrink10.log)"
echo ""

# 17. Run test config
echo "[19/22] Running spectra_task_tests1000_simple_shrink00.py..."
python main_train_eval.py --config=configs/Spectra/spectra_task_tests1000_simple_shrink00.py --mode=train > "$LOGS_DIR/spectra_task_tests1000_simple_shrink00.log" 2>&1
echo "✓ spectra_task_tests1000_simple_shrink00.py completed (log: $LOGS_DIR/spectra_task_tests1000_simple_shrink00.log)"
echo ""

# 18. Run test config
echo "[20/22] Running spectra_task_tests1000_simple_shrink10.py..."
python main_train_eval.py --config=configs/Spectra/spectra_task_tests1000_simple_shrink10.py --mode=train > "$LOGS_DIR/spectra_task_tests1000_simple_shrink10.log" 2>&1
echo "✓ spectra_task_tests1000_simple_shrink10.py completed (log: $LOGS_DIR/spectra_task_tests1000_simple_shrink10.log)"
echo ""

# 19. Run test config
echo "[21/22] Running spectra_task_tests1000_hybrid_shrink00.py..."
python main_train_eval.py --config=configs/Spectra/spectra_task_tests1000_hybrid_shrink00.py --mode=train > "$LOGS_DIR/spectra_task_tests1000_hybrid_shrink00.log" 2>&1
echo "✓ spectra_task_tests1000_hybrid_shrink00.py completed (log: $LOGS_DIR/spectra_task_tests1000_hybrid_shrink00.log)"
echo ""

# 20. Run test config
echo "[22/22] Running spectra_task_tests1000_hybrid_shrink10.py..."
python main_train_eval.py --config=configs/Spectra/spectra_task_tests1000_hybrid_shrink10.py --mode=train > "$LOGS_DIR/spectra_task_tests1000_hybrid_shrink10.log" 2>&1
echo "✓ spectra_task_tests1000_hybrid_shrink10.py completed (log: $LOGS_DIR/spectra_task_tests1000_hybrid_shrink10.log)"
echo ""

echo "=== ALL DONE ==="
echo "Completed at: $(date)"
echo "All 22 Spectra Task configurations have been run!"