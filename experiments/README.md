# Experiments

This directory contains the scripts to reproduce the training curves reported in the paper for the
following algorithms:

1. **P3O (Particle POMDP Policy Optimization)**
2. **SLAC (Stochastic Latent Actor-Critic)**
3. **DVRL (Deep Variational Reinforcement Learning)**
4. **DSMC (Dual Sequential Monte Carlo)**

They can be evaluated on the following POMDP tasks (defined in `ppomdp/envs/pomdps/`):

- `pendulum`: Pendulum swing-up.
- `cartpole`: Cart-pole swing-up.
- `light-dark-2d`: A 2D navigation task with state-dependent noise.
- `triangulation`: A 2D triangulation task.

## Usage

Each algorithm can be run using Python with the appropriate configuration. We use [wandb](https://wandb.ai/) for logging the results. To run the scripts without wandb logging, add the `--no-use_logger` flag to the following commands. Note that the scripts assume a CUDA device is available.

Example configurations for each algorithm we consider are given below.

1. P3O with recurrent policy:

   ```bash
   python p3o_recurrent_experiment.py \
       --env_id pendulum \
       --num_seeds 10 \
       --total_time_steps 1000000 \
       --num_history_particles 128 \
       --num_belief_particles 32 \
       --slew_rate_penalty 0.05 \
       --tempering 0.5 \
       --backward_sampling \
       --backward_sampling_mult 4 \
       --learning_rate 0.0003 \
       --batch_size 16 \
       --init_std 1.0
   ```

2. P3O with set transformer policy:

   ```bash
   python p3o_attention_experiment.py \
       --env_id pendulum \
       --num_seeds 10 \
       --total_time_steps 1000000 \
       --num_history_particles 128 \
       --num_belief_particles 32 \
       --slew_rate_penalty 0.05 \
       --tempering 0.5 \
       --backward_sampling \
       --backward_sampling_mult 4 \
       --learning_rate 0.0003 \
       --batch_size 256 \
       --init_std 1.0
   ```

3. SLAC

   ```bash
   python slac_experiment.py \
       --env_id pendulum \
       --num_seeds 10 \
       --total_time_steps 1000000 \
       --num_belief_particles 32 \
       --buffer_size 1000000 \
       --learning_starts 5000 \
       --policy_lr 0.0003 \
       --critic_lr 0.001 \
       --batch_size 16 \
       --alpha 0.2 \
       --gamma 0.99 \
       --tau 0.005
   ```

4. DVRL

   ```bash
   python dvrl_experiment.py \
       --env_id pendulum \
       --num_seeds 10 \
       --total_time_steps 1000000 \
       --num_belief_particles 32 \
       --buffer_size 1000000 \
       --learning_starts 5000 \
       --policy_lr 0.0003 \
       --critic_lr 0.001 \
       --batch_size 256 \
       --num_batches 8 \
       --alpha 0.2 \
       --gamma 0.99 \
       --tau 0.005
   ```

5. DSMC

   ```bash
   python dsmc_experiment.py \
       --env_id pendulum \
       --num_seeds 10 \
       --total_time_steps 500_000 \
       --num_planner_steps 3 \
       --num_planner_particles 32 \
       --num_belief_particles 32 \
       --buffer_size 500_000 \
       --learning_starts 5000 \
       --policy_lr 0.0003 \
       --critic_lr 0.001 \
       --batch_size 256 \
       --num_batches 8 \
       --alpha 0.2 \
       --gamma 0.99 \
       --tau 0.005
   ```

### Common Parameters

All algorithms share some additional common parameters:

- `cuda_device`: GPU device ID to use (default: 0)
- `project_name`: Name of the wandb project
- `experiment_group`: Group name for organizing experiments
- `experiment_name`: Custom name for the experiment (optional)
- `experiment_tags`: Tags for organizing experiments in wandb (optional)
