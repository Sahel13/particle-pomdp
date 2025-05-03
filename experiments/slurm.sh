#!/bin/bash
#SBATCH --job-name=slac_%A # Job name includes algo and array ID
#SBATCH --output=slac_%A_%a.out # Output file includes algo name
#SBATCH --error=slac_%A_%a.err  # Error file includes algo name
#SBATCH --mail-type=end
#SBATCH --mail-user=sahel.iqbal@aalto.fi

# Array definition: 10 tasks (0-9), max 5 running concurrently
#SBATCH --array=0-9%5

# Resources PER ARRAY TASK:
#SBATCH --ntasks=1           # Each array task runs 1 process
#SBATCH --cpus-per-task=8    # CPUs for that process
#SBATCH --gres=gpu:a100:1    # 1 V100 GPU for that process
#SBATCH --mem=6000M             # Memory for that process (8GB)
#SBATCH --time=02:00:00      # Time limit for EACH array task

# Generate a unique run ID (same for all tasks in the array)
EXPERIMENT_ID="202505021602"

module purge
module load mamba
source activate particle-pomdp

python slac_experiment.py \
  --env_id cartpole \
  --starting-seed ${SLURM_ARRAY_TASK_ID} \
  --num_seeds 1 \
  --cuda_device 0 \
  --experiment_group slac-cartpole \
  --experiment_id ${EXPERIMENT_ID}
