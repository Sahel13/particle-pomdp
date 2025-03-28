#!/bin/bash

# Specify which GPU to use.
export CUDA_VISIBLE_DEVICES=0

# Command line arguments for the script.
algorithm=${1:?No algorithm specified. Usage: ./run.sh <algorithm> <environment>}
environment=${2:?No environment specified. Usage: ./run.sh <algorithm> <environment>}

# Create a directory for logging.
log_dir="logs/${algorithm}_${environment}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$log_dir"

# Loop over seeds.
for seed in {0..9}
do
    python "$algorithm.py" --seed "$seed" --env "$environment" --log_dir "$log_dir"
done

# Process the data for plotting.
python post_process.py --folder "./$log_dir"
