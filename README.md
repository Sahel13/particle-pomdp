# Particle POMDP Policy Optimization (P3O)
A sequential Monte Carlo approach for policy optimization in continuous POMPDs

## Installation

Create new conda environment with the yaml file provided
```bash
$ conda env create -f particle-pomdp.yaml 
```
This environment assumes a GPU is available and installs JAX accordingly. If you have different hardware consult [JAX](https://github.com/jax-ml/jax?tab=readme-ov-file#installation). 

Now, install the `ppomdp` package with
```bash
$ cd particle-pomdp
$ pip install -e .
```
for an editable install.

## Examples

We provide multiple environments to test P3O on:
* `pendulum`: a pendulum swing-up task, where only the angular position is observable
* `cartpole`: a cartpole swing-up task, where only the angular and cartesian positions are observable
* `light-dark-2d`: a 2D navigation task with location-dependent noise
* `triangulation`: a 2D navigation task with heading-only observations

Each evnrionment can be ran with two policies:
   * a policy with history inputs `recurrent`
   * a policy with belief particles inputs `attention`

For example, for the light-dark environment run:
```bash
$ python examples/lightdark2d/p3o_recurrent.py
```
or 
```bash
$ python examples/lightdark2d/p3o_attention.py
```

## Baselines

We provide the following baselines for comparison:
1. **Deep variational reinforcement learning for POMDPs (DVRL)** [1]
2. **Stochastic latent actor-critic (SLAC)** [2]
3. **Dual Sequential Monte Carlo (DSMC)** [3]

More details on the baselines can be found under `experiements/README.md`
```
## References
1. Igl, M., Zintgraf, L., Le, T.A., Wood, F. and Whiteson, S. Deep variational reinforcement learning for
   POMDPs. In International Conference on Machine Learning, 2018.
2. Lee, Alex X., et al. Stochastic latent actor-critic: Deep reinforcement learning with a latent
   variable model. In Advances in Neural Information Processing Systems, 2020.
3. Wang, Y., Liu, B., Wu, J., Zhu, Y., Du, S., Fei-Fei, L., Tenenbaum, J. DualSMC: Tunneling 
   Differentiable Filtering and Planning under Continuous POMDPs, IJCAI, 2020.
