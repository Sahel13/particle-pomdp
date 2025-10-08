import pytest

import chex
from jax import random, numpy as jnp

from ppomdp.core import BeliefState, HistoryParticles, HistoryState, BeliefInfo
from ppomdp.smc import backward_tracing


@pytest.fixture
def history_states():
    obs_dim = 3
    action_dim = 2
    num_history_particles = 3
    num_time_steps = 3
    lstm_dim = 32
    key = random.PRNGKey(0)

    # Generate observations
    key, sub_key = random.split(key)
    observations = random.uniform(
        sub_key, (num_time_steps + 1, num_history_particles, obs_dim)
    )

    # Generate actions
    key, sub_key = random.split(key)
    actions = random.uniform(
        sub_key, (num_time_steps + 1, num_history_particles, action_dim)
    )

    # Generate carry states
    key, sub_key1, sub_key2 = random.split(key, 3)
    carry = [(
        random.uniform(sub_key1, (num_time_steps + 1, num_history_particles, lstm_dim)),
        random.uniform(sub_key2, (num_time_steps + 1, num_history_particles, lstm_dim)),
    )]

    # Create history particles
    history_particles = HistoryParticles(
        actions=actions,
        carry=carry,
        observations=observations
    )

    # Create resampling indices
    resampling_indices = random.randint(
        key, (num_time_steps + 1, num_history_particles), 0, num_history_particles
    )

    # Create history state
    return HistoryState(
        particles=history_particles,
        log_weights=jnp.zeros((num_time_steps + 1, num_history_particles)),
        weights=jnp.ones((num_time_steps + 1, num_history_particles)) / num_history_particles,
        resampling_indices=resampling_indices,
        rewards=jnp.zeros((num_time_steps + 1, num_history_particles))
    )


@pytest.fixture
def belief_states(history_states):
    # Create dummy belief state
    num_time_steps, num_history_particles = history_states.weights.shape
    num_time_steps -= 1
    num_belief_particles = 2
    state_dim = 4  # Example state dimension

    # Create dummy arrays for belief state
    particles = jnp.zeros((num_time_steps + 1, num_history_particles, num_belief_particles, state_dim))
    log_weights = jnp.zeros((num_time_steps + 1, num_history_particles, num_belief_particles))
    weights = jnp.ones((num_time_steps + 1, num_history_particles, num_belief_particles)) / num_belief_particles
    resampling_indices = jnp.zeros((num_time_steps + 1, num_history_particles, num_belief_particles), dtype=jnp.int32)

    return BeliefState(
        particles=particles,
        log_weights=log_weights,
        weights=weights,
        resampling_indices=resampling_indices
    )


@pytest.fixture
def belief_infos(belief_states):
    # Create dummy belief infos
    num_time_steps, num_history_particles, num_belief_particles, state_dim = belief_states.particles.shape

    # Create dummy arrays for belief infos - exactly match history_states shape
    # history_states has shape (num_time_steps + 1, num_particles)
    ess = jnp.ones((num_time_steps, num_history_particles))  # Remove +1 to match traced_indices
    mean = jnp.zeros((num_time_steps, num_history_particles, state_dim))
    covar = jnp.zeros((num_time_steps, num_history_particles, state_dim, state_dim))

    # Set diagonal elements of covariance to 1
    covar = covar.at[..., jnp.arange(state_dim), jnp.arange(state_dim)].set(1.0)

    return BeliefInfo(
        ess=ess,
        mean=mean,
        covar=covar
    )


def test_backward_tracing(history_states, belief_states, belief_infos):
    # Set random seed for reproducibility
    rng_key = random.PRNGKey(0)

    # Define test dimensions
    num_time_steps = 3
    num_particles = 3

    # Run backward tracing
    traced_history, traced_belief, traced_belief_info = backward_tracing(
        rng_key,
        history_states,
        belief_states,
        belief_infos,
        resample=False
    )
    resampling_indices = history_states.resampling_indices

    # Manually trace back the ancestry of each selected particle
    def trace_ancestry(final_idx):
        ancestry = [final_idx]
        current_idx = final_idx
        for t in range(num_time_steps, 0, -1):
            current_idx = resampling_indices[t][current_idx]
            ancestry.append(current_idx)
        return list(reversed(ancestry))

    # Check each selected trajectory
    for particle_idx in range(num_particles):
        ancestry = trace_ancestry(particle_idx)

        # Verify that observations, actions, and carry match the ancestry
        for t in range(num_time_steps + 1):
            # Check observations
            assert jnp.all(
                jnp.equal(
                    history_states.particles.observations[t, ancestry[t]],
                    traced_history.observations[t, particle_idx],
                )
            )

            # Check actions
            assert jnp.all(
                jnp.equal(
                    history_states.particles.actions[t, ancestry[t]],
                    traced_history.actions[t, particle_idx],
                )
            )

            # Check carry states
            for carry_idx in range(len(history_states.particles.carry)):
                assert jnp.all(
                    jnp.equal(
                        history_states.particles.carry[carry_idx][0][t, ancestry[t]],
                        traced_history.carry[carry_idx][0][t, particle_idx],
                    )
                )
                assert jnp.all(
                    jnp.equal(
                        history_states.particles.carry[carry_idx][1][t, ancestry[t]],
                        traced_history.carry[carry_idx][1][t, particle_idx],
                    )
                )
