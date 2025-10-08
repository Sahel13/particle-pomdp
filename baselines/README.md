# Running Experiments

### DVRL Example

To run the DVRL experiment with default settings on the CartPole environment:

```bash
python experiments/dvrl_experiment.py
```

This will run the experiment with the following default parameters:
- Environment: CartPole
- Number of seeds: 10
- Number of belief particles: 32
- Total time steps: 100,000
- Buffer size: 100,000
- Learning starts: 5,000
- Policy learning rate: 0.0001
- Critic learning rate: 0.001
- Batch size: 256
- Alpha: 0.2
- Gamma: 0.995
- Tau: 0.005

#### Advanced Usage

You can customize the experiment by passing command-line arguments:

```bash
# Run on a different environment
python experiments/dvrl_experiment.py --env_id=pendulum

# Run with a specific GPU
python experiments/dvrl_experiment.py --cuda_device=1

# Run with custom hyperparameters
python experiments/dvrl_experiment.py \
  --num_belief_particles=64 \
  --total_time_steps=200000 \
  --policy_lr=0.0003 \
  --critic_lr=0.0005 \
  --batch_size=512

# Run without logging
python experiments/dvrl_experiment.py --use_logger=False

# Run with custom experiment group and tags
python experiments/dvrl_experiment.py \
  --experiment_group=dvrl-custom \
  --experiment_tags=dvrl pendulum test
```
