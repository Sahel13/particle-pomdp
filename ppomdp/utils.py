import math

from jax import Array, random
from jax import numpy as jnp


def batch_data(
    rng_key: Array,
    data_size: int,
    batch_size: int,
    skip_last: bool = False,
) -> list[Array]:
    """Generates batched indices.

    Args:
        rng_key (Array): Random key for shuffling the data.
        data_size (int): The size of the dataset to be batched.
        batch_size (int): The size of each batch.
        skip_last (bool, optional): If True, skips the last incomplete batch. Defaults to False.

    Yields:
        list[Array]: A list of batched indices.
    """
    batch_idx = random.permutation(rng_key, data_size)

    if skip_last:
        # Skip incomplete batch
        num_batches = data_size // batch_size
        batch_idx = batch_idx[: num_batches * batch_size]
    else:
        # Include incomplete batch
        num_batches = math.ceil(data_size / batch_size)

    batch_idx = jnp.array_split(batch_idx, num_batches)
    return batch_idx


def ess(weights: Array) -> Array:
    """Compute the effective sample size."""
    return 1.0 / jnp.sum(jnp.square(weights))


def systematic_resampling(rng_key: Array, weights: Array, num_samples: int) -> Array:
    n = weights.shape[0]
    u = random.uniform(rng_key)
    cum_sum = jnp.cumsum(weights)
    sorted_uniforms = (jnp.arange(num_samples) + u) / num_samples
    idx = jnp.searchsorted(cum_sum, sorted_uniforms)
    return jnp.clip(idx, 0, n - 1).astype(jnp.int_)
