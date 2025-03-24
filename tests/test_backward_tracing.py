import pytest

import chex
from jax import random, numpy as jnp


from ppomdp.core import BeliefState, HistoryParticles, HistoryState
from ppomdp.smc import backward_tracing


@pytest.fixture
def history_states():
    obs_dim = 3
    action_dim = 2
    num_history_particles = 3
    num_time_steps = 3
    lstm_dim = 32
    key = random.PRNGKey(0)
    key, sub_key = random.split(key)
    observations = random.uniform(
        sub_key, (num_time_steps + 1, num_history_particles, obs_dim)
    )

    key, sub_key = random.split(key)
    actions = random.uniform(
        sub_key, (num_time_steps + 1, num_history_particles, action_dim)
    )

    key, sub_key1, sub_key2 = random.split(key, 3)
    carry = [(
        random.uniform(sub_key1, (num_time_steps + 1, num_history_particles, lstm_dim)),
        random.uniform(sub_key2, (num_time_steps + 1, num_history_particles, lstm_dim)),
    )]
    dummy_log_probs = jnp.zeros((num_time_steps + 1, num_history_particles))
    history_particles = HistoryParticles(observations, actions, carry, dummy_log_probs)

    resampling_indices = random.randint(
        key, (num_time_steps + 1, num_history_particles), 0, num_history_particles
    )
    return HistoryState(
        particles=history_particles,
        log_weights=jnp.zeros((num_time_steps + 1, num_history_particles)),
        weights=jnp.ones((num_time_steps + 1, num_history_particles)) / num_history_particles,
        rewards=jnp.zeros((num_time_steps + 1, num_history_particles)),
        resampling_indices=resampling_indices,
    )


@pytest.fixture
def belief_states(history_states):
    # This is just a dummy belief state, not used in the test.
    num_time_steps, num_history_particles = history_states.weights.shape
    num_time_steps -= 1
    num_belief_particles = 2
    dummy_array = jnp.zeros(
        (num_time_steps + 1, num_history_particles, num_belief_particles)
    )
    state = BeliefState(
        particles=dummy_array,
        log_weights=dummy_array,
        weights=dummy_array,
        resampling_indices=dummy_array,
    )
    return state


def test_backward_tracing(history_states, belief_states):
    # Set random seed for reproducibility
    rng_key = random.PRNGKey(0)

    # Define test dimensions
    num_time_steps = 3
    num_particles = 3

    # Run backward tracing
    traced_history, _ = backward_tracing(rng_key, history_states, belief_states, False)
    resampling_indices = history_states.resampling_indices

    # Manually trace back the ancestry of each selected particle
    def trace_ancestry(final_idx):
        ancestry = [final_idx]
        current_idx = final_idx
        for t in range(num_time_steps, 0, -1):
            current_idx = resampling_indices[t][current_idx]
            ancestry.append(current_idx)
        return list(reversed(ancestry))

    # Verify shapes and dtypes
    chex.assert_trees_all_equal_shapes_and_dtypes(history_states, traced_history)

    # Check each selected trajectory
    for particle_idx in range(num_particles):
        ancestry = trace_ancestry(particle_idx)

        # Verify that observations, actions, and carry match the ancestry
        for t in range(num_time_steps + 1):
            assert jnp.all(
                jnp.equal(
                    history_states.particles.observations[t, ancestry[t]],
                    traced_history.particles.observations[t, particle_idx],
                )
            )
            assert jnp.all(
                jnp.equal(
                    history_states.particles.actions[t, ancestry[t]],
                    traced_history.particles.actions[t, particle_idx],
                )
            )
            assert jnp.all(
                jnp.equal(
                    history_states.particles.carry[0][0][t, ancestry[t]],
                    traced_history.particles.carry[0][0][t, particle_idx],
                )
            )
