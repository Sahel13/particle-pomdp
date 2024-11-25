import time

import jax
from jax import random
import jax.numpy as jnp

from distrax import Chain, MultivariateNormalDiag, ScalarAffine
from flax.linen.initializers import constant
from flax.training.train_state import TrainState

from ppomdp.bijector import Tanh
from ppomdp.policy import GRU, get_recurrent_policy, train_step
from ppomdp.smc import backward_tracing, smc
from ppomdp.utils import batch_data

from copy import deepcopy
import optax
import matplotlib.pyplot as plt

from environment import prior_dist, trans_model, obs_model, reward_fn
from environment import state_dim, action_dim, obs_dim, num_time_steps
from environment import stddev_obs

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)
# jax.config.update("jax_debug_nans", True)

init_log_std = jnp.log(jnp.array([1.0, 1.0]))

network = GRU(
    dim=action_dim,
    feature_fn=lambda x: x,
    encoder_size=[256, 256],
    recurr_size=[64, 64],
    output_size=[256, 256],
    init_log_std=constant(init_log_std),
)

shift = jnp.zeros((action_dim,))
scale = jnp.array([100.0, 100.0])
bijector = Chain([ScalarAffine(shift, scale), Tanh()])
policy = get_recurrent_policy(network, bijector)

rng_key = random.PRNGKey(1337)

num_outer_particles = 256
num_inner_particles = 256
tempering = 0.05
slew_rate_penalty = 1e-3

learning_rate = 3e-4
batch_size = 64
num_epochs = 500

# Initialize training state
key, obs_key, param_key = random.split(rng_key, 3)
init_carry = policy.reset(num_outer_particles)
init_obs = random.normal(obs_key, (num_outer_particles, obs_dim))
init_params = network.init(param_key, init_carry, init_obs)["params"]
train_state = TrainState.create(
    apply_fn=network.apply,
    params=init_params,
    tx=optax.adam(learning_rate)
)

jitted_smc = jax.jit(smc, static_argnums=(1, 2, 3, 4, 5, 6, 7, 9, 10))
jitted_backward_tracing = jax.jit(backward_tracing, static_argnums=(5,))

for i in range(1, num_epochs + 1):
    start_time = time.time()

    # run nested smc
    key, sub_key = random.split(key)
    outer_states, inner_states, inner_info, log_marginal = \
        jitted_smc(
            sub_key,
            num_outer_particles,
            num_inner_particles,
            num_time_steps,
            prior_dist,
            trans_model,
            obs_model,
            policy,
            train_state.params,
            reward_fn,
            tempering,
            slew_rate_penalty,
        )

    # trace ancestors of outer states
    key, sub_key = random.split(key)
    traced_outer, traced_inner, _ = \
        jitted_backward_tracing(sub_key, outer_states, inner_states, inner_info)

    variance = jnp.mean(jnp.var(traced_inner.particles[-1], axis=1), axis=0)

    # update policy parameters
    loss = 0.0
    key, sub_key = random.split(key)
    batch_indices = batch_data(sub_key, num_outer_particles, batch_size)
    for batch_idx in batch_indices:
        outer_batch = jax.tree.map(lambda x: x[:, batch_idx], traced_outer.particles)
        train_state, batch_loss = train_step(policy, train_state, outer_batch)
        loss += batch_loss

    entropy = policy.entropy(train_state.params)
    end_time = time.time()
    time_diff = end_time - start_time

    print(
        f"Epoch: {i:3d}, "
        f"Log marginal: {log_marginal:.3f}, "
        f"Entropy: {entropy:.3f}, "
        f"Time per epoch: {time_diff:.3f}s"
    )


eval_state = deepcopy(train_state)
eval_state.params["log_std"] = -20.0 * jnp.ones((action_dim,))

key, sub_key = random.split(key)
outer_states, inner_states, inner_infos, log_marginal = \
    jitted_smc(
        sub_key,
        num_outer_particles,
        num_inner_particles,
        num_time_steps,
        prior_dist,
        trans_model,
        obs_model,
        policy,
        eval_state.params,
        reward_fn,
        tempering,
        slew_rate_penalty
    )

# trace ancestors of outer states
key, sub_key = random.split(key)
outer_states, inner_states, inner_infos = \
    jitted_backward_tracing(sub_key, outer_states, inner_states, inner_infos)

observations = outer_states.particles[0]
actions = outer_states.particles[1]
state_means = inner_infos.mean
state_covars = inner_infos.covar
rwrds = outer_states.rewards

fig, axs = plt.subplots(4, 1, figsize=(8, 8))
for n in range(num_outer_particles):
    axs[0].plot(state_means[:, n, 0])
    axs[0].set_ylabel('State-1')

    axs[1].plot(state_means[:, n, 1])
    axs[1].set_ylabel('State-2')

    axs[2].plot(actions[:, n, 0])
    axs[2].set_ylabel('Act-1')

    axs[3].plot(actions[:, n, 1])
    axs[3].set_ylabel('Act-2')

    # axs[4].plot(state_covars[:, n, 0, 0])
    # axs[4].set_ylabel('Var-1')
    #
    # axs[5].plot(state_covars[:, n, 1, 1])
    # axs[5].set_xlabel('Time Step')
    # axs[5].set_ylabel('Var-2')

plt.tight_layout()
plt.show()


states = []
actions = []
observations = []

key, state_key, obs_key = random.split(key, 3)
init_dist = MultivariateNormalDiag(
    loc=jnp.array([2.0, 2.0, 0.0, 0.0]),
    scale_diag=jnp.array([1e-2, 1e-2, 1e-2, 1e-2])
)

state = init_dist.sample(seed=state_key)
obs = obs_model.sample(obs_key, state)
carry = policy.reset(1)

states.append(state)
observations.append(obs)

for _ in range(num_time_steps):
    key, state_key, obs_key, action_key = random.split(key, 4)

    carry, action = policy.sample(action_key, obs, carry, eval_state.params)
    state = trans_model.sample(state_key, state, action[0])
    obs = obs_model.sample(obs_key, state)

    states.append(state)
    actions.append(action[0])
    observations.append(obs)

states = jnp.squeeze(jnp.array(states))
actions = jnp.squeeze(jnp.array(actions))
observations = jnp.squeeze(jnp.array(observations))

xgrid = jnp.linspace(-1.0, 6.0, 100)
ygrid = jnp.linspace(-3.0, 2.5, 100)
X, Y = jnp.meshgrid(xgrid, ygrid)

Z = jnp.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        s = jnp.array([X[i, j], Y[i, j], 0.0, 0.0])
        Z = Z.at[i, j].set(jnp.linalg.norm(stddev_obs(s)))

plt.imshow(-Z, extent=(-2.0, 6.0, -3.0, 2.5), origin='lower', cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(states[:, 0], states[:, 1], 'r-')
plt.plot(states[0, 0], states[0, 1], 'bo', marker='o', markersize=12, markeredgewidth=2, markerfacecolor='none')  # Hollow blue circle at the first point
plt.plot(states[-1, 0], states[-1, 1], 'g+', markersize=12, markeredgewidth=2)  # Bigger green cross at the last point
plt.title('Light-Dark Environment')
plt.show()
