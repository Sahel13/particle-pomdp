# particle-pomdp

## Installation
Install [JAX](https://github.com/jax-ml/jax?tab=readme-ov-file#installation) for the appropriate hardware. Then run
```bash
$ pip install -e ".[test]"
```
for an editable install with test dependencies.

## Baselines
We provide the following baselines for comparison:
1. **Stochastic latent actor-critic (SLAC)** [1] - See `baselines/slac/README.md` for instructions.

## References
1. Lee, Alex X., et al. "Stochastic latent actor-critic: Deep reinforcement
   learning with a latent variable model." Advances in Neural Information
   Processing Systems (2020): 741-752.
