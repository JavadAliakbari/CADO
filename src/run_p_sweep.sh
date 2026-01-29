#!/bin/bash

# Script to run the p-sweep experiment
# Usage: ./run_p_sweep.sh [test|full]

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate conda environment if needed
if [ -n "$CONDA_DEFAULT_ENV" ] && [ "$CONDA_DEFAULT_ENV" != "FedStruct" ]; then
    echo "Activating conda environment: FedStruct"
    source ~/miniconda3/bin/activate FedStruct
fi

# Default to full experiment
MODE="${1:-full}"

echo "=================================="
echo "CADO P-Sweep Experiment Runner"
echo "=================================="
echo ""

if [ "$MODE" == "test" ]; then
    echo "Running QUICK TEST..."
    echo ""
    python "${SCRIPT_DIR}/test_p_sweep.py"
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "Test completed successfully!"
        echo "To run the full experiment, use: ./run_p_sweep.sh full"
    else
        echo ""
        echo "Test failed with exit code: $exit_code"
    fi
    
    exit $exit_code

elif [ "$MODE" == "full" ]; then
    echo "Running FULL P-SWEEP EXPERIMENT..."
    echo ""
    echo "This will take approximately 1-2 hours."
    echo "Results will be saved to results/p_sweep_YYYYMMDD_HHMMSS/"
    echo ""
    read -p "Continue? (y/n) " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python "${SCRIPT_DIR}/p_sweep_experiment.py"
        exit_code=$?
        
        if [ $exit_code -eq 0 ]; then
            echo ""
            echo "=========================================="
            echo "Experiment completed successfully!"
            echo "=========================================="
            echo ""
            
            # Find the most recent results directory
            RESULTS_DIR=$(ls -td results/p_sweep_* 2>/dev/null | head -1)
            
            if [ -n "$RESULTS_DIR" ]; then
                echo "Results saved to: $RESULTS_DIR"
                echo ""
                read -p "Generate visualizations now? (y/n) " -n 1 -r
                echo ""
                
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    python "${SCRIPT_DIR}/visualize_p_sweep.py" "$RESULTS_DIR"
                    echo ""
                    echo "Visualizations saved to: $RESULTS_DIR"
                fi
            fi
        else
            echo ""
            echo "Experiment failed with exit code: $exit_code"
        fi
        
        exit $exit_code
    else
        echo "Experiment cancelled."
        exit 0
    fi

else
    echo "Invalid mode: $MODE"
    echo ""
    echo "Usage: ./run_p_sweep.sh [test|full]"
    echo ""
    echo "Modes:"
    echo "  test  - Run quick test with minimal parameters"
    echo "  full  - Run full p-sweep experiment (default)"
    echo ""
    exit 1
fi
