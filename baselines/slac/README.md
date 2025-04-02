# Stochastic Latent Actor-Critic

A modified version of the stochastic latent actor-critic (SLAC) algorithm [1]. It has the following differences to the original implementation:
1. The dynamics and observation models are assumed known and not learned.
2. Instead of parametrizing the belief state with a Gaussian distribution, we use a particle filter.

## Usage
See `experiments/slac.py`.

## References
1. Lee, Alex X., et al. "Stochastic latent actor-critic: Deep reinforcement
learning with a latent variable model." Advances in Neural Information
Processing Systems (2020): 741-752.
