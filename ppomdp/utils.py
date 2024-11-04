from jax import Array
from jax import numpy as jnp
from jax import random

import math


def batch_data(
    rng_key: Array,
    data_size: int,
    batch_size: int,
    skip_last: bool = False,
) -> Array:
    """
    Generates batches of data from the given dataset.

    Args:
        rng_key (Array): Random key for shuffling the data.
        data_size (int): The size dataset to be batched.
        batch_size (int): The size of each batch.
        skip_last (bool, optional): If True, skips the last incomplete batch. Defaults to False.

    Yields:
        Array: A batch of data.
    """
    batch_idx = random.permutation(rng_key, data_size)

    if skip_last:
        # Skip incomplete batch
        samples_per_epoch = data_size // batch_size
        batch_idx = batch_idx[: samples_per_epoch * batch_size]
        batch_idx = batch_idx.reshape((samples_per_epoch, batch_size))
    else:
        # include incomplete batch
        samples_per_epoch = math.ceil(data_size / batch_size)
        batch_idx = jnp.array_split(batch_idx, samples_per_epoch)

    for idx in batch_idx:
        yield idx
