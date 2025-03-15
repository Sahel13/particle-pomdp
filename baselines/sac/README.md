# Soft Actor-Critic

Reference implementation for the soft actor-critic algorithm [1, 2]. This is based on the
[cleanrl](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py)
and [Brax](https://github.com/google/brax/tree/main/brax/training/agents/sac)
implementations, although it differs from these in the following respects:
1. The policy has homoscedastic noise.
2. No automatic entropy tuning.

## Usage

Run
```bash
$ python utils.py --env <env_id>
```
where '<env_id>' is one of the following:
1. pendulum
2. cartpole
3. lightdark2d

## References

1. Haarnoja, Tuomas, et al. "Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor." International Conference on Machine Learning. 2018.
2. Haarnoja, Tuomas, et al. "Soft actor-critic algorithms and applications." arXiv preprint arXiv:1812.05905 (2018).
