# Experiments

This directory contains the implementation of four main algorithms for solving POMDPs:

1. **P3O (Particle POMDP Policy Optimization)**
2. **SLAC (Stochastic Latent Actor-Critic)**
3. **DVRL (Deep Variational Reinforcement Learning)**
4. **DSMC (Dual Sequential Monte Carlo)**

## Running Experiments

Each algorithm can be run using Python with the appropriate configuration. Below are examples for each algorithm:

### P3O

```bash
# Run p3o on cartpole environment
python p3o_recurrent_experiment.py \
    --env_id light-dark-2d \
    --num_seeds 10 \
    --cuda_device 0 \
    --project_name particle-pomdp \
    --experiment_group p3o-light \
    --experiment_tags p3o light \
    --total_time_steps 500_000 \
    --num_history_particles 256 \
    --num_belief_particles 32 \
    --slew_rate_penalty 0.001 \
    --tempering 0.2 \
    --backward_sampling \
    --backward_sampling_mult 2 \
    --learning_rate 0.0003 \
    --batch_size 64 \
    --init_std 1.0 \
    --no-use_logger
```

### SLAC

```bash
# Run SLAC on CartPole environment
python slac_experiment.py \
    --env_id light-dark-2d \
    --num_seeds 10 \
    --cuda_device 0 \
    --project_name particle-pomdp \
    --experiment_group slac-light \
    --experiment_tags slac light \
    --total_time_steps 500_000 \
    --num_belief_particles 32 \
    --buffer_size 500_000 \
    --learning_starts 5000 \
    --policy_lr 0.0003 \
    --critic_lr 0.001 \
    --batch_size 16 \
    --alpha 0.2 \
    --gamma 0.99 \
    --tau 0.005 \
    --no-use_logger
```

### DVRL

```bash
# Run DVRL on CartPole environment
python dvrl_experiment.py \
    --env_id light-dark-2d \
    --num_seeds 10 \
    --cuda_device 0 \
    --project_name particle-pomdp \
    --experiment_group dvrl-light \
    --experiment_tags dvrl light \
    --total_time_steps 500_000 \
    --num_belief_particles 32 \
    --buffer_size 500_000 \
    --learning_starts 5000 \
    --policy_lr 0.0003 \
    --critic_lr 0.001 \
    --batch_size 256 \
    --num_batches 8 \
    --alpha 0.2 \
    --gamma 0.99 \
    --tau 0.005 \
    --no-use_logger
```

### DSMC

```bash

# Run DSMC on CartPole environment
python dsmc_experiment.py \
    --env_id light-dark-2d \
    --num_seeds 10 \
    --cuda_device 0 \
    --project_name particle-pomdp \
    --experiment_group dsmc-light \
    --experiment_tags dsmc light \
    --total_time_steps 500_000 \
    --num_planner_steps 5 \
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
    --tau 0.005 \
    --no-use_logger
```

## Common Parameters

All algorithms share some common parameters:

- `env_id`: The environment to run the experiment on (e.g., "light-dark-2d", "cartpole")
- `num_seeds`: Number of random seeds to run the experiment with
- `cuda_device`: GPU device ID to use (default: 0)
- `use_logger`: Whether to use wandb for logging (true/false)
- `project_name`: Name of the wandb project
- `experiment_group`: Group name for organizing experiments
- `experiment_name`: Custom name for the experiment (optional)
- `experiment_tags`: Tags for organizing experiments in wandb (optional)

## Environment Support

The algorithms have been tested on the following environments:
- Pendulum: Continuous control task with partial observability
- CartPole: Classic control task with partial observability
- LightDark: A 2D navigation task with partial observability
- Target: A 2D traingulation task with partial observability

## Logging

When `use_logger=true`, experiments are logged to Weights & Biases (wandb) with the following metrics:
- Average return