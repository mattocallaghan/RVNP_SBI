#!/bin/bash
################################################################################
# RVNP-SBI ICML 2026 Experiment Runner
#
# Runs all experiments for the publication systematically:
# - 4 tasks (CS, SIR, Pendulum, Spectra)
# - 5 Nobs values (1, 10, 100, 1000, 10000)
# - 5 methods (RVNP, RVNP-T, RVNP-G, NPE, NNPE)
#
# Usage:
#   bash run_all_experiments.sh                    # Run all experiments
#   bash run_all_experiments.sh --dry-run          # Print commands only
#   bash run_all_experiments.sh --task=CS          # Run only CS task
#   bash run_all_experiments.sh --method=RVNP      # Run only RVNP method
#
# Compatible with Google Colab and local environments.
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DRY_RUN=false
FILTER_TASK=""
FILTER_METHOD=""
FILTER_NOBS=""
OUTPUT_DIR="experiments_output"
RESULTS_DIR="experiment_results"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --task=*)
            FILTER_TASK="${1#*=}"
            shift
            ;;
        --method=*)
            FILTER_METHOD="${1#*=}"
            shift
            ;;
        --nobs=*)
            FILTER_NOBS="${1#*=}"
            shift
            ;;
        --output-dir=*)
            OUTPUT_DIR="${1#*=}"
            shift
            ;;
        --results-dir=*)
            RESULTS_DIR="${1#*=}"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run           Print commands without executing"
            echo "  --task=TASK         Filter by task (CS, SIR, Pendulum, Spectra)"
            echo "  --method=METHOD     Filter by method (RVNP, RVNP-T, RVNP-G, NPE, NNPE)"
            echo "  --nobs=N            Filter by number of observations (1, 10, 100, 1000, 10000)"
            echo "  --output-dir=DIR    Output directory (default: experiments_output)"
            echo "  --results-dir=DIR   Results directory (default: experiment_results)"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Print header
echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}RVNP-SBI ICML 2026 Experiment Runner${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

# Check Python availability
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python not found. Please install Python 3.8+${NC}"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "main_train_eval.py" ]; then
    echo -e "${RED}Error: main_train_eval.py not found. Please run from RVNP_SBI directory.${NC}"
    exit 1
fi

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$RESULTS_DIR"

# Check for Data directory
if [ ! -d "Data" ]; then
    echo -e "${YELLOW}Warning: Data directory not found. Creating it...${NC}"
    mkdir -p Data
    echo -e "${YELLOW}Please ensure you have the required data files in the Data/ directory.${NC}"
fi

# Print configuration
echo "Configuration:"
echo "  Output directory: $OUTPUT_DIR"
echo "  Results directory: $RESULTS_DIR"
if [ "$DRY_RUN" = true ]; then
    echo -e "  ${YELLOW}Mode: DRY RUN (commands will be printed but not executed)${NC}"
fi
if [ -n "$FILTER_TASK" ]; then
    echo "  Task filter: $FILTER_TASK"
fi
if [ -n "$FILTER_METHOD" ]; then
    echo "  Method filter: $FILTER_METHOD"
fi
if [ -n "$FILTER_NOBS" ]; then
    echo "  Nobs filter: $FILTER_NOBS"
fi
echo ""

# Use Python experiment runner
CMD="python experiment_runner.py --output-dir=$OUTPUT_DIR --results-dir=$RESULTS_DIR"

if [ "$DRY_RUN" = true ]; then
    CMD="$CMD --dry-run"
fi

if [ -n "$FILTER_TASK" ]; then
    CMD="$CMD --task=$FILTER_TASK"
fi

if [ -n "$FILTER_METHOD" ]; then
    CMD="$CMD --method=$FILTER_METHOD"
fi

if [ -n "$FILTER_NOBS" ]; then
    CMD="$CMD --nobs=$FILTER_NOBS"
fi

# Run experiment runner
echo -e "${GREEN}Starting experiment runner...${NC}"
echo "Command: $CMD"
echo ""

eval $CMD

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}================================================================================${NC}"
    echo -e "${GREEN}Experiments completed successfully!${NC}"
    echo -e "${GREEN}================================================================================${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Collect metrics:"
    echo "     python metrics_collector.py --summary"
    echo ""
    echo "  2. Generate publication plots:"
    echo "     python scripts/publication_plots.py --task=all --save-dir=publication_plots"
    echo ""
    echo "  3. Create summary table:"
    echo "     python scripts/publication_plots.py --summary"
    echo ""
else
    echo ""
    echo -e "${RED}================================================================================${NC}"
    echo -e "${RED}Experiments failed!${NC}"
    echo -e "${RED}================================================================================${NC}"
    echo ""
    echo "Check the logs in experiment_runner.log for details."
    exit 1
fi
