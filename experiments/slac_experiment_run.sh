#!/bin/bash

# Example script to run SLAC experiments over multiple seeds

# Run SLAC on CartPole environment with 10 seeds
python experiments/run_slac_experiment.py \
  --env_id cartpole \
  --num_seeds 10 \
  --cuda_device 0 \
  --project_name particle-pomdp \
  --experiment_name slac-cartpole-multiseed \
  --experiment_group slac-cartpole \
  --experiment_tags slac cartpole multiseed \
  --total_time_steps 100000 \
  --num_belief_particles 32 \
  --buffer_size 100000 \
  --learning_starts 5000 \
  --policy_lr 0.0003 \
  --critic_lr 0.001 \
  --batch_size 256 \
  --alpha 0.2 \
  --gamma 0.95 \
  --tau 0.005

# Run SLAC on Pendulum environment with 10 seeds
python experiments/run_slac_experiment.py \
  --env_id pendulum \
  --num_seeds 10 \
  --cuda_device 0 \
  --project_name particle-pomdp \
  --experiment_name slac-pendulum-multiseed \
  --experiment_group slac-pendulum \
  --experiment_tags slac pendulum multiseed \
  --total_time_steps 25000 \
  --num_belief_particles 32 \
  --buffer_size 25000 \
  --learning_starts 5000 \
  --policy_lr 0.0003 \
  --critic_lr 0.001 \
  --batch_size 256 \
  --alpha 0.2 \
  --gamma 0.95 \
  --tau 0.005 