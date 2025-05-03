# Deep Variational Reinforcement Learning for POMDPs

A modified version of the DVRL algorithm [1]. Unlike the original implementation, this version uses
SAC [2] as the base RL algorithm instead of A2C [3]. The action and Q networks have different MLP encoders
for the particle set representing the belief state.

## References

1. Igl, M., Zintgraf, L., Le, T.A., Wood, F. and Whiteson, S. Deep variational reinforcement learning for
   POMDPs. In International Conference on Machine Learning, 2018.
2. T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine. Soft actor-critic: Off-policy maximum entropy deep
   reinforcement learning with a stochastic actor. In International Conference on Machine Learning, 2018.
3. Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver, D., and Kavukcuoglu, K.
   Asynchronous methods for deep reinforcement learning. In International Conference on Machine Learning, 2016.
