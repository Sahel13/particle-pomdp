# Particle POMDP Policy Optimization (P3O)

Implements the P3O algorithm from the NeurIPS 2025 paper [Sequential Monte
Carlo for Policy Optimization in Continuous POMDPs](https://arxiv.org/abs/2505.16732).
This code was written by [Sahel Iqbal](https://github.com/Sahel13) and [Hany
Abdulsamad](https://github.com/hanyas).

P30 is a policy optimization algorithm for partially observable Markov decision processes (POMDPs) with continuous state, action and observation spaces. See the scripts in `examples/` for demonstrations of how to train policies using P3O.

## Installation

Install [JAX](https://github.com/jax-ml/jax?tab=readme-ov-file#installation) for the available hardware. Then run

```bash
$ pip install -e .
```

for an editable install.

## Exmaples
Run the light-dark environment to demonstrate optimal information gathering behavior
```bash
$ python examples/lightdark2d/p3o_recurrent.py
```

## Baselines

We provide the following baselines for comparison:

1. [Deep Variational Reinforcement Learning for POMDPs (DVRL)](https://proceedings.mlr.press/v80/igl18a/igl18a.pdf) - See `baselines/dvrl`.
2. [Stochastic Latent Actor-Critic (SLAC)](https://arxiv.org/pdf/1907.00953) - See `baselines/slac`.
3. [DualSMC](https://www.ijcai.org/Proceedings/2020/0579.pdf) - See `baselines/dsmc`.

See `baselines/README.md` for details.

## Citation

If you find the code useful, please cite our paper

```bib
@inproceedings{abdulsamad2025sequential,
  title = {Sequential {Monte Carlo} for policy optimization in continuous {POMDPs}},
  author = {Hany Abdulsamad and Sahel Iqbal and Simo S{\"a}rkk{\"a}},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2025},
}
```
