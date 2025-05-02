#!/bin/bash

algo="$1"

case "$algo" in
  slac|dsmc|dvrl)
    # allowed, continue
    ;;
  *)
    echo "Error: Invalid input '$algo'. Allowed values are: slac, dsmc, dvrl." >&2
    exit 1
    ;;
esac

#SBATCH --job-name=${algo}_%A # Job name includes algo and array ID
#SBATCH --output=${algo}_%A_%a.out # Output file includes algo name
#SBATCH --error=${algo}_%A_%a.err  # Error file includes algo name
#SBATCH --mail-type=end
#SBATCH --mail-user=sahel.iqbal@aalto.fi

# Array definition: 10 tasks (0-9), max 5 running concurrently
#SBATCH --array=0-9%5

# Resources PER ARRAY TASK:
#SBATCH --ntasks=1           # Each array task runs 1 process
#SBATCH --cpus-per-task=8    # CPUs for that process
#SBATCH --gres=gpu:v100:1    # 1 V100 GPU for that process
#SBATCH --mem=8000M             # Memory for that process (8GB)
#SBATCH --time=03:59:59      # Time limit for EACH array task

# Generate a unique run ID (same for all tasks in the array)
EXPERIMENT_ID=$(uuidgen)

module purge
module load mamba
mamba activate particle-pomdp

python ${algo}_experiment.py \
  --env_id cartpole \
  --num_seeds 1 \
  --cuda_device 0 \
  --experiment_group slac-cartpole \
  --experiment_id ${EXPERIMENT_ID} \
  --starting_seed ${SLURM_ARRAY_TASK_ID}
