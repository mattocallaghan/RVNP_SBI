#!/bin/bash
################################################################################
# RVNP-SBI Complete Pipeline for Google Colab
#
# This script runs the complete experimental pipeline:
# 1. Training all models
# 2. Evaluation (with and without SIR)
# 3. Metrics collection
# 4. Publication plots generation
#
# Designed to be resumable - if Colab times out, simply re-run this script
# and it will continue from where it left off.
#
# Usage:
#   bash run_experiments_colab.sh                    # Run complete pipeline
#   bash run_experiments_colab.sh --task=CS          # Run only CS task
#   bash run_experiments_colab.sh --status           # Check progress
#   bash run_experiments_colab.sh --plots-only       # Only generate plots
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
TASK=""
METHOD=""
NOBS=""
STATUS_ONLY=false
PLOTS_ONLY=false
NO_SIR=false
SKIP_TRAINING=false
SKIP_EVALUATION=false
RESULTS_DIR="experiment_results"
PLOTS_DIR="publication_plots"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task=*)
            TASK="${1#*=}"
            shift
            ;;
        --method=*)
            METHOD="${1#*=}"
            shift
            ;;
        --nobs=*)
            NOBS="${1#*=}"
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
        --no-sir)
            NO_SIR=true
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
            echo "Pipeline Options:"
            echo "  --task=TASK          Filter by task (CS, SIR, Pendulum, Spectra)"
            echo "  --method=METHOD      Filter by method (RVNP-simple, RVNP-NN, NPE)"
            echo "  --nobs=N             Filter by Nobs (1, 10, 100, 1000, 10000)"
            echo "  --no-sir             Disable SIR evaluation"
            echo "  --skip-training      Skip training, only evaluate existing models"
            echo "  --skip-evaluation    Skip evaluation, only train"
            echo ""
            echo "Utility Options:"
            echo "  --status             Check current progress and exit"
            echo "  --plots-only         Only generate plots from existing metrics"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  bash run_experiments_colab.sh"
            echo "  bash run_experiments_colab.sh --task=CS --method=RVNP-NN"
            echo "  bash run_experiments_colab.sh --status"
            echo "  bash run_experiments_colab.sh --plots-only"
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
echo -e "${CYAN}                    RVNP-SBI ICML 2026 Experiment Pipeline${NC}"
echo -e "${CYAN}                      Resumable for Google Colab${NC}"
echo -e "${CYAN}================================================================================${NC}"
echo ""

# Check Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python not found. Please install Python 3.8+${NC}"
    exit 1
fi

# Check if in correct directory
if [ ! -f "scripts/integrated_pipeline.py" ]; then
    echo -e "${RED}Error: scripts/integrated_pipeline.py not found.${NC}"
    echo -e "${RED}Please run from RVNP_SBI root directory.${NC}"
    exit 1
fi

# Create directories
mkdir -p "$RESULTS_DIR"
mkdir -p "$PLOTS_DIR"
mkdir -p Data
mkdir -p logs

# Status check
if [ "$STATUS_ONLY" = true ]; then
    echo -e "${BLUE}=== Pipeline Status ===${NC}"
    python scripts/integrated_pipeline.py --status --results-dir="$RESULTS_DIR"
    echo ""
    echo -e "${BLUE}=== Metrics Database ===${NC}"
    if [ -f "$RESULTS_DIR/metrics_database.csv" ]; then
        echo "Metrics database exists with $(tail -n +2 "$RESULTS_DIR/metrics_database.csv" | wc -l) entries"
        echo ""
        echo "Methods and tasks:"
        tail -n +2 "$RESULTS_DIR/metrics_database.csv" | cut -d',' -f1,2,4 | sort -u
    else
        echo "No metrics database found yet."
    fi
    echo ""
    echo -e "${BLUE}=== Available Plots ===${NC}"
    if [ -d "$PLOTS_DIR" ] && [ "$(ls -A $PLOTS_DIR)" ]; then
        ls -lh "$PLOTS_DIR"/*.png 2>/dev/null || echo "No plots generated yet."
    else
        echo "No plots generated yet."
    fi
    exit 0
fi

# Plots only mode
if [ "$PLOTS_ONLY" = true ]; then
    echo -e "${GREEN}=== Generating Publication Plots ===${NC}"
    echo ""

    if [ ! -f "$RESULTS_DIR/metrics_database.csv" ]; then
        echo -e "${RED}Error: No metrics database found at $RESULTS_DIR/metrics_database.csv${NC}"
        echo -e "${RED}Please run the pipeline first to collect metrics.${NC}"
        exit 1
    fi

    python scripts/publication_plots.py \
        --metrics-db="$RESULTS_DIR/metrics_database.csv" \
        --task=all \
        --save-dir="$PLOTS_DIR"

    echo ""
    echo -e "${GREEN}✓ Plots saved to $PLOTS_DIR/${NC}"
    ls -lh "$PLOTS_DIR"/*.png
    exit 0
fi

# Print configuration
echo -e "${YELLOW}Configuration:${NC}"
echo "  Results directory: $RESULTS_DIR"
echo "  Plots directory: $PLOTS_DIR"
if [ -n "$TASK" ]; then
    echo "  Task filter: $TASK"
fi
if [ -n "$METHOD" ]; then
    echo "  Method filter: $METHOD"
fi
if [ -n "$NOBS" ]; then
    echo "  Nobs filter: $NOBS"
fi
echo "  SIR evaluation: $([ "$NO_SIR" = true ] && echo "disabled" || echo "enabled")"
echo "  Training: $([ "$SKIP_TRAINING" = true ] && echo "skipped" || echo "enabled")"
echo "  Evaluation: $([ "$SKIP_EVALUATION" = true ] && echo "skipped" || echo "enabled")"
echo ""

# Build command
CMD="python scripts/integrated_pipeline.py --results-dir=$RESULTS_DIR"

if [ -n "$TASK" ]; then
    CMD="$CMD --task=$TASK"
fi

if [ -n "$METHOD" ]; then
    CMD="$CMD --method=$METHOD"
fi

if [ -n "$NOBS" ]; then
    CMD="$CMD --nobs=$NOBS"
fi

if [ "$NO_SIR" = true ]; then
    CMD="$CMD --no-sir"
fi

if [ "$SKIP_TRAINING" = true ]; then
    CMD="$CMD --skip-training"
fi

if [ "$SKIP_EVALUATION" = true ]; then
    CMD="$CMD --skip-evaluation"
fi

# Run integrated pipeline
echo -e "${GREEN}=== Starting Integrated Pipeline ===${NC}"
echo ""
echo "Command: $CMD"
echo ""

START_TIME=$(date +%s)

eval $CMD

PIPELINE_EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo -e "${CYAN}================================================================================${NC}"

if [ $PIPELINE_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Pipeline completed successfully!${NC}"
else
    echo -e "${RED}✗ Pipeline encountered errors${NC}"
    echo -e "${YELLOW}This is OK for Colab - progress is saved. Re-run to continue.${NC}"
fi

echo -e "${CYAN}================================================================================${NC}"
echo ""
echo "Runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""

# Generate plots if we have metrics
if [ -f "$RESULTS_DIR/metrics_database.csv" ]; then
    NUM_METRICS=$(tail -n +2 "$RESULTS_DIR/metrics_database.csv" | wc -l)

    if [ "$NUM_METRICS" -gt 0 ]; then
        echo -e "${GREEN}=== Generating Publication Plots ===${NC}"
        echo "Found $NUM_METRICS metric entries"
        echo ""

        python scripts/publication_plots.py \
            --metrics-db="$RESULTS_DIR/metrics_database.csv" \
            --task=all \
            --save-dir="$PLOTS_DIR" \
            2>&1 | head -20  # Limit output

        echo ""
        echo -e "${GREEN}✓ Plots saved to $PLOTS_DIR/${NC}"

        # List generated plots
        if [ -d "$PLOTS_DIR" ]; then
            NUM_PLOTS=$(ls -1 "$PLOTS_DIR"/*.png 2>/dev/null | wc -l)
            if [ "$NUM_PLOTS" -gt 0 ]; then
                echo "Generated $NUM_PLOTS plot files:"
                ls -lh "$PLOTS_DIR"/*.png | awk '{print "  - " $9 " (" $5 ")"}'
            fi
        fi
    else
        echo -e "${YELLOW}No metrics collected yet - plots will be generated after evaluation${NC}"
    fi
else
    echo -e "${YELLOW}No metrics database yet - run evaluation to generate metrics${NC}"
fi

echo ""
echo -e "${CYAN}================================================================================${NC}"
echo -e "${CYAN}                              Next Steps${NC}"
echo -e "${CYAN}================================================================================${NC}"
echo ""
echo "1. Check progress:"
echo "   bash run_experiments_colab.sh --status"
echo ""
echo "2. Continue if interrupted:"
echo "   bash run_experiments_colab.sh"
echo ""
echo "3. Generate/update plots:"
echo "   bash run_experiments_colab.sh --plots-only"
echo ""
echo "4. Results are saved in:"
echo "   - Metrics: $RESULTS_DIR/metrics_database.csv"
echo "   - State: $RESULTS_DIR/pipeline_state.json"
echo "   - Plots: $PLOTS_DIR/"
echo "   - Logs: pipeline.log"
echo ""
echo -e "${CYAN}================================================================================${NC}"
