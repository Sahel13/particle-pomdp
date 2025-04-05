#!/bin/bash

# Default values
ALGORITHM=""
ENVIRONMENT=""
SEEDS=10
LOG_DIR="logs"
PROJ_NAME="particle-pomdp"
CUDA_DEVICE=0
EXTRA_ARGS=()

# Display help message
function show_help {
    echo "Usage: $0 [options] [--] [extra_args...]"
    echo "Options:"
    echo "  -a, --algorithm ALGORITHM    Algorithm to run (slac, dsmc, etc.)"
    echo "  -e, --environment ENV        Environment to run (cartpole, etc.)"
    echo "  -s, --seeds SEEDS            Number of seeds to run (default: 10)"
    echo "  -g, --cuda-device DEVICE     CUDA device to use (default: 0)"
    echo "  -l, --log-dir DIR            Directory for logs (default: logs)"
    echo "  -p, --project-name PROJECT   Project name for wandb (default: particle-pomdp)"
    echo "  -h, --help                   Show this help message"
    echo ""
    echo "Any additional arguments after '--' will be passed directly to the Python script."
    echo ""
    echo "Example:"
    echo "  $0 -a slac -e cartpole -s 5 -g 1 -- --num_belief_particles 64"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -a|--algorithm)
            ALGORITHM="$2"
            shift 2
            ;;
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -s|--seeds)
            SEEDS="$2"
            shift 2
            ;;
        -l|--log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        -p|--project-name)
            PROJ_NAME="$2"
            shift 2
            ;;
        -g|--cuda-device)
            CUDA_DEVICE=$2
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Check required arguments
if [ -z "$ALGORITHM" ]; then
    echo "Error: Algorithm is required"
    show_help
fi

if [ -z "$ENVIRONMENT" ]; then
    echo "Error: Environment is required"
    show_help
fi

# Specify which GPU to use.
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

# Create a directory for logging.
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${LOG_DIR}/${ALGORITHM}_${ENVIRONMENT}_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "Running $ALGORITHM on $ENVIRONMENT with $SEEDS seeds"
echo "Using CUDA device: $CUDA_DEVICE"
echo "Logs will be saved to $LOG_DIR"
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    echo "Additional arguments: ${EXTRA_ARGS[*]}"
fi

# Loop over seeds
for ((seed=0; seed<SEEDS; seed++))
do
    echo "Running seed $seed:"
    
    # Generate experiment name
    EXP_NAME="${ALGORITHM}-${ENVIRONMENT}-seed-${seed}"
    
    # Build the command with all parameters
    CMD="python run_${ALGORITHM}.py --seed $seed --env_id $ENVIRONMENT"
    
    # Pass logging arguments
    CMD="$CMD --project_name $PROJ_NAME --experiment_name $EXP_NAME --log_dir $LOG_DIR"
    
    # Pass any extra arguments if provided
    if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
        CMD="$CMD ${EXTRA_ARGS[*]}"
    fi
    
    # Execute the command
    $CMD
done

echo "Experiments complete. Results saved in $LOG_DIR."
