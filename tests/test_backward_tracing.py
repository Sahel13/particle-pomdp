import jax.numpy as jnp
import pytest
from jax import random
import chex

from ppomdp.core import InnerState, OuterParticles, OuterState
from ppomdp.smc import backward_tracing


@pytest.fixture
def outer_states():
    obs_dim = 3
    action_dim = 2
    num_outer_particles = 3
    num_time_steps = 3
    lstm_dim = 32
    key = random.PRNGKey(0)
    key, sub_key = random.split(key)
    observations = random.uniform(
        sub_key, (num_time_steps + 1, num_outer_particles, obs_dim)
    )

    key, sub_key = random.split(key)
    actions = random.uniform(
        sub_key, (num_time_steps + 1, num_outer_particles, action_dim)
    )

    key, sub_key1, sub_key2 = random.split(key, 3)
    carry = [
        (
            random.uniform(
                sub_key1, (num_time_steps + 1, num_outer_particles, lstm_dim)
            ),
            random.uniform(
                sub_key2, (num_time_steps + 1, num_outer_particles, lstm_dim)
            ),
        )
    ]
    outer_particles = OuterParticles(observations, actions, carry)

    resampling_indices = random.randint(
        key, (num_time_steps + 1, num_outer_particles), 0, num_outer_particles
    )
    return OuterState(
        particles=outer_particles,
        weights=jnp.ones((num_time_steps + 1, num_outer_particles))
        / num_outer_particles,
        rewards=jnp.zeros((num_time_steps + 1, num_outer_particles)),
        resampling_indices=resampling_indices,
    )


@pytest.fixture
def inner_states(outer_states):
    # This is just a dummy inner state, not used in the test.
    num_time_steps, num_outer_particles = outer_states.weights.shape
    num_time_steps -= 1
    num_inner_particles = 2
    dummy_array = jnp.zeros(
        (num_time_steps + 1, num_outer_particles, num_inner_particles)
    )
    state = InnerState(
        particles=dummy_array,
        log_weights=dummy_array,
        weights=dummy_array,
        resampling_indices=dummy_array,
    )
    return state


def test_backward_tracing(outer_states, inner_states):
    # Set random seed for reproducibility
    rng_key = random.PRNGKey(0)

    # Define test dimensions
    num_time_steps = 3
    num_particles = 3

    # Run backward tracing
    traced_outer, _ = backward_tracing(rng_key, outer_states, inner_states, False)
    resampling_indices = outer_states.resampling_indices

    # Manually trace back the ancestry of each selected particle
    def trace_ancestry(final_idx):
        ancestry = [final_idx]
        current_idx = final_idx
        for t in range(num_time_steps, 0, -1):
            current_idx = resampling_indices[t][current_idx]
            ancestry.append(current_idx)
        return list(reversed(ancestry))

    # Verify shapes and dtypes
    chex.assert_trees_all_equal_shapes_and_dtypes(outer_states, traced_outer)

    # Check each selected trajectory
    for particle_idx in range(num_particles):
        ancestry = trace_ancestry(particle_idx)

        # Verify that observations, actions, and carry match the ancestry
        for t in range(num_time_steps + 1):
            assert jnp.all(
                jnp.equal(
                    outer_states.particles.observations[t, ancestry[t]],
                    traced_outer.particles.observations[t, particle_idx],
                )
            )
            assert jnp.all(
                jnp.equal(
                    outer_states.particles.actions[t, ancestry[t]],
                    traced_outer.particles.actions[t, particle_idx],
                )
            )
            assert jnp.all(
                jnp.equal(
                    outer_states.particles.carry[0][0][t, ancestry[t]],
                    traced_outer.particles.carry[0][0][t, particle_idx],
                )
            )
