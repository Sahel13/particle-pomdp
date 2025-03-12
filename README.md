# particle-pomdp

## Installation
Install [JAX](https://github.com/jax-ml/jax?tab=readme-ov-file#installation) for the appropriate hardware. Then run
```bash
$ pip install -e ".[test]"
```
for an editable install with test dependencies.

## Baselines

We provide the following baselines for comparison:
1. **Deep variational reinforcement learning for POMDPs (DVRL)** [1] - See `baselines/dvrl/README.md` for instructions.
2. **Stochastic latent actor-critic (SLAC)** [2] - See `baselines/slac/README.md` for instructions.

## References
1. Igl, M., Zintgraf, L., Le, T.A., Wood, F. and Whiteson, S. Deep variational reinforcement learning for
   POMDPs. In International Conference on Machine Learning, 2018.
2. Lee, Alex X., et al. Stochastic latent actor-critic: Deep reinforcement learning with a latent
   variable model. In Advances in Neural Information Processing Systems, 2020.
