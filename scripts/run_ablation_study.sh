#!/bin/bash
################################################################################
# RVNP-SBI Ablation Study Runner
#
# Runs ablation studies for CS task examining:
# 1. Effect of correction model parameter count (nn_block_dim)
# 2. Effect of shrinkage prior strength (lambda_shrinkage)
#
# Usage:
#   bash run_ablation_study.sh                     # Run all ablations
#   bash run_ablation_study.sh --params-only       # Only parameter size ablation
#   bash run_ablation_study.sh --shrinkage-only    # Only shrinkage ablation
#   bash run_ablation_study.sh --status            # Check progress
#   bash run_ablation_study.sh --plots-only        # Only generate plots
#
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PARAMS_ONLY=false
SHRINKAGE_ONLY=false
STATUS_ONLY=false
PLOTS_ONLY=false
SKIP_TRAINING=false
SKIP_EVALUATION=false
NO_SIR=true  # Ablation studies don't need SIR evaluation
RESULTS_DIR="experiment_results_ablation"
PLOTS_DIR="publication_plots_ablation"

# Ablation config lists
PARAM_CONFIGS=(
    "configs/CS_task/ablation_params_tiny.py"
    "configs/CS_task/ablation_params_small.py"
    "configs/CS_task/ablation_params_medium.py"
    "configs/CS_task/ablation_params_large.py"
    "configs/CS_task/ablation_params_huge.py"
)

SHRINKAGE_CONFIGS=(
    "configs/CS_task/ablation_shrink_0.py"
    "configs/CS_task/ablation_shrink_weak.py"
    "configs/CS_task/ablation_shrink_medium.py"
    "configs/CS_task/ablation_shrink_strong.py"
    "configs/CS_task/ablation_shrink_verystrong.py"
)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --params-only)
            PARAMS_ONLY=true
            shift
            ;;
        --shrinkage-only)
            SHRINKAGE_ONLY=true
            shift
            ;;
        --status)
            STATUS_ONLY=true
            shift
            ;;
        --plots-only)
            PLOTS_ONLY=true
            shift
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --skip-evaluation)
            SKIP_EVALUATION=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "RVNP-SBI Ablation Study for CS Task"
            echo ""
            echo "Ablation Studies:"
            echo "  Parameter Size: nn_block_dim = [16, 32, 52, 128, 256]"
            echo "  Shrinkage Prior: lambda_shrinkage = [0.0, 0.01, 0.1, 1.0, 10.0]"
            echo ""
            echo "Options:"
            echo "  --params-only        Run only parameter size ablation"
            echo "  --shrinkage-only     Run only shrinkage prior ablation"
            echo "  --skip-training      Skip training, only evaluate existing models"
            echo "  --skip-evaluation    Skip evaluation, only train"
            echo "  --status             Check current progress and exit"
            echo "  --plots-only         Only generate plots from existing metrics"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  bash run_ablation_study.sh"
            echo "  bash run_ablation_study.sh --params-only"
            echo "  bash run_ablation_study.sh --status"
            echo "  bash run_ablation_study.sh --plots-only"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Print header
echo -e "${CYAN}================================================================================${NC}"
echo -e "${CYAN}                    RVNP-SBI Ablation Study Runner${NC}"
echo -e "${CYAN}                         CS Task - Nobs=100${NC}"
echo -e "${CYAN}================================================================================${NC}"
echo ""

# Check Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python not found. Please install Python 3.8+${NC}"
    exit 1
fi

# Check if in correct directory
if [ ! -f "main_train_eval.py" ]; then
    echo -e "${RED}Error: main_train_eval.py not found.${NC}"
    echo -e "${RED}Please run from RVNP_SBI directory.${NC}"
    exit 1
fi

# Create directories
mkdir -p "$RESULTS_DIR"
mkdir -p "$PLOTS_DIR"
mkdir -p Data
mkdir -p logs

# Status check
if [ "$STATUS_ONLY" = true ]; then
    echo -e "${BLUE}=== Ablation Study Status ===${NC}"
    echo ""

    echo -e "${YELLOW}Parameter Size Ablation:${NC}"
    for config in "${PARAM_CONFIGS[@]}"; do
        config_name=$(basename $config .py)
        workspace="output/ablation_cs_${config_name#ablation_}_nobs100"
        if [ -d "$workspace" ]; then
            echo -e "  ${GREEN}✓${NC} $config_name (workspace exists)"
        else
            echo -e "  ${YELLOW}○${NC} $config_name (not started)"
        fi
    done

    echo ""
    echo -e "${YELLOW}Shrinkage Prior Ablation:${NC}"
    for config in "${SHRINKAGE_CONFIGS[@]}"; do
        config_name=$(basename $config .py)
        workspace="output/ablation_cs_${config_name#ablation_}_nobs100"
        if [ -d "$workspace" ]; then
            echo -e "  ${GREEN}✓${NC} $config_name (workspace exists)"
        else
            echo -e "  ${YELLOW}○${NC} $config_name (not started)"
        fi
    done

    echo ""
    echo -e "${BLUE}=== Metrics Database ===${NC}"
    if [ -f "$RESULTS_DIR/ablation_metrics.csv" ]; then
        echo "Metrics database exists with $(tail -n +2 "$RESULTS_DIR/ablation_metrics.csv" 2>/dev/null | wc -l) entries"
    else
        echo "No metrics database found yet."
    fi

    exit 0
fi

# Plots only mode
if [ "$PLOTS_ONLY" = true ]; then
    echo -e "${GREEN}=== Generating Ablation Plots ===${NC}"
    echo ""

    if [ ! -f "$RESULTS_DIR/ablation_metrics.csv" ]; then
        echo -e "${RED}Error: No metrics database found at $RESULTS_DIR/ablation_metrics.csv${NC}"
        echo -e "${RED}Please run the ablation study first to collect metrics.${NC}"
        exit 1
    fi

    python scripts/ablation_plots.py \
        --metrics-db="$RESULTS_DIR/ablation_metrics.csv" \
        --save-dir="$PLOTS_DIR"

    echo ""
    echo -e "${GREEN}✓ Plots saved to $PLOTS_DIR/${NC}"
    ls -lh "$PLOTS_DIR"/*.png 2>/dev/null
    exit 0
fi

# Print configuration
echo -e "${YELLOW}Configuration:${NC}"
echo "  Results directory: $RESULTS_DIR"
echo "  Plots directory: $PLOTS_DIR"
if [ "$PARAMS_ONLY" = true ]; then
    echo "  Mode: Parameter size ablation only"
elif [ "$SHRINKAGE_ONLY" = true ]; then
    echo "  Mode: Shrinkage prior ablation only"
else
    echo "  Mode: All ablations (10 experiments)"
fi
echo "  Training: $([ "$SKIP_TRAINING" = true ] && echo "skipped" || echo "enabled")"
echo "  Evaluation: $([ "$SKIP_EVALUATION" = true ] && echo "skipped" || echo "enabled")"
echo ""

# Determine which configs to run
CONFIGS_TO_RUN=()
if [ "$PARAMS_ONLY" = true ]; then
    CONFIGS_TO_RUN=("${PARAM_CONFIGS[@]}")
elif [ "$SHRINKAGE_ONLY" = true ]; then
    CONFIGS_TO_RUN=("${SHRINKAGE_CONFIGS[@]}")
else
    CONFIGS_TO_RUN=("${PARAM_CONFIGS[@]}" "${SHRINKAGE_CONFIGS[@]}")
fi

echo -e "${GREEN}=== Running Ablation Study ===${NC}"
echo "Total experiments: ${#CONFIGS_TO_RUN[@]}"
echo ""

START_TIME=$(date +%s)
SUCCESS_COUNT=0
FAILED_COUNT=0

# Initialize metrics CSV
METRICS_FILE="$RESULTS_DIR/ablation_metrics.csv"
if [ ! -f "$METRICS_FILE" ]; then
    echo "config,nn_block_dim,lambda_shrinkage,acauc_mean,acauc_std,lpp_median,lpp_std,nrmse_mean,nrmse_std,training_time" > "$METRICS_FILE"
fi

# Run each config
for i in "${!CONFIGS_TO_RUN[@]}"; do
    config="${CONFIGS_TO_RUN[$i]}"
    config_name=$(basename $config .py)

    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}Experiment $((i+1))/${#CONFIGS_TO_RUN[@]}: $config_name${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    # Build command
    CMD="python scripts/main_train_eval.py --config=$config"

    if [ "$SKIP_TRAINING" = false ]; then
        CMD="$CMD --train"
    fi

    if [ "$SKIP_EVALUATION" = false ]; then
        CMD="$CMD --evaluate"
    fi

    echo "Command: $CMD"
    echo ""

    # Run experiment
    if eval $CMD; then
        echo -e "${GREEN}✓ Experiment completed successfully${NC}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo -e "${RED}✗ Experiment failed${NC}"
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo -e "${CYAN}================================================================================${NC}"
echo -e "${CYAN}                         Ablation Study Summary${NC}"
echo -e "${CYAN}================================================================================${NC}"
echo ""
echo "Total experiments: ${#CONFIGS_TO_RUN[@]}"
echo -e "Success: ${GREEN}$SUCCESS_COUNT${NC}"
echo -e "Failed:  ${RED}$FAILED_COUNT${NC}"
echo ""
echo "Runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""

# Generate plots if we have metrics
if [ -f "$METRICS_FILE" ]; then
    NUM_METRICS=$(tail -n +2 "$METRICS_FILE" 2>/dev/null | wc -l)

    if [ "$NUM_METRICS" -gt 0 ]; then
        echo -e "${GREEN}=== Generating Ablation Plots ===${NC}"
        echo "Found $NUM_METRICS metric entries"
        echo ""

        if [ -f "ablation_plots.py" ]; then
            python scripts/ablation_plots.py \
                --metrics-db="$METRICS_FILE" \
                --save-dir="$PLOTS_DIR" \
                2>&1 | head -20

            echo ""
            echo -e "${GREEN}✓ Plots saved to $PLOTS_DIR/${NC}"

            if [ -d "$PLOTS_DIR" ]; then
                NUM_PLOTS=$(ls -1 "$PLOTS_DIR"/*.png 2>/dev/null | wc -l)
                if [ "$NUM_PLOTS" -gt 0 ]; then
                    echo "Generated $NUM_PLOTS plot files:"
                    ls -lh "$PLOTS_DIR"/*.png | awk '{print "  - " $9 " (" $5 ")"}'
                fi
            fi
        else
            echo -e "${YELLOW}ablation_plots.py not found - skipping plot generation${NC}"
        fi
    fi
fi

echo ""
echo -e "${CYAN}================================================================================${NC}"
echo -e "${CYAN}                              Next Steps${NC}"
echo -e "${CYAN}================================================================================${NC}"
echo ""
echo "1. Check progress:"
echo "   bash run_ablation_study.sh --status"
echo ""
echo "2. Generate/update plots:"
echo "   bash run_ablation_study.sh --plots-only"
echo ""
echo "3. Results are saved in:"
echo "   - Metrics: $RESULTS_DIR/ablation_metrics.csv"
echo "   - Plots: $PLOTS_DIR/"
echo "   - Model outputs: output/ablation_cs_*/"
echo ""
echo -e "${CYAN}================================================================================${NC}"
