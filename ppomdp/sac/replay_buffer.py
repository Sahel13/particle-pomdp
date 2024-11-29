from functools import partial
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp


class RingBuffer(NamedTuple):
    """A ring buffer for storing arbitrary pytrees of arrays.

    Once full, adding a new element will overwrite the oldest element.
    """

    data: chex.ArrayTree
    head: jax.Array | int
    tail: jax.Array | int
    full: jax.Array | bool


def init(entry_prototype: chex.ArrayTree, buffer_size: int) -> RingBuffer:
    """Initialize a ring buffer with a given entry prototype and buffer size."""
    chex.assert_tree_has_only_ndarrays(entry_prototype)

    def tile(tensor):
        return jnp.repeat(tensor[None, ...], buffer_size, axis=0)

    data = jax.tree.map(tile, entry_prototype)
    return RingBuffer(data, 0, 0, False)


def max_size(buffer: RingBuffer) -> int:
    leaves = jax.tree.leaves(buffer.data)
    return leaves[0].shape[0]


def size(buffer: RingBuffer) -> chex.Numeric:
    """Get the number of elements in the buffer."""
    return jax.lax.select(buffer.full, max_size(buffer), buffer.head - buffer.tail)


def assert_feature_dims(*args):
    """Check that feature sizes are the same."""

    def equality_comparator(a, b):
        if a.ndim > 1 and b.ndim > 1:
            return a.shape[-1] == b.shape[-1]
        else:
            return True

    def error_msg_function(a, b):
        return f"Feature dimensions do not match: {a.shape[-1]} != {b.shape[-1]}"

    chex.assert_trees_all_equal_comparator(
        equality_comparator, error_msg_function, *args
    )


@partial(jax.jit, donate_argnums=0)
def add_batch(buffer: RingBuffer, entries: chex.ArrayTree) -> RingBuffer:
    """Add a batch of entries to the buffer."""
    batch_size = jax.tree.leaves(entries)[0].shape[0]
    # TODO: Check if batch_size is less than max_size.
    chex.assert_tree_has_only_ndarrays(entries)
    chex.assert_tree_shape_prefix(entries, (batch_size,))
    chex.assert_trees_all_equal_structs(buffer.data, entries)
    assert_feature_dims(buffer.data, entries)
    buffer_size = max_size(buffer)
    wrap_idx = buffer_size - buffer.head

    def no_wrap(_buffer, _entries):
        new_data = jax.tree.map(
            lambda tb, t: jax.lax.dynamic_update_slice_in_dim(tb, t, _buffer.head, 0),
            _buffer.data,
            _entries,
        )
        new_head = (_buffer.head + batch_size) % buffer_size
        new_tail = jax.lax.select(_buffer.full, new_head, _buffer.tail)
        new_full = new_head == new_tail
        return RingBuffer(new_data, new_head, new_tail, new_full)

    def wrap(_buffer, _entries):
        shift = -((_buffer.head + batch_size) % buffer_size)

        def roll_and_update_slice(tree, slice):
            return jax.lax.dynamic_update_slice_in_dim(
                jnp.roll(tree, shift, axis=0), slice, _buffer.head + shift, 0
            )

        new_data = jax.tree.map(roll_and_update_slice, _buffer.data, _entries)
        new_head = 0
        new_tail = 0
        new_full = True
        return RingBuffer(new_data, new_head, new_tail, new_full)

    return jax.lax.cond(wrap_idx >= batch_size, no_wrap, wrap, buffer, entries)


@partial(jax.jit, static_argnums=2)
def sample(
    rng_key: chex.PRNGKey, buffer: RingBuffer, batch_size: int
) -> chex.ArrayTree:
    """Sample a batch of elements from the buffer."""
    indices = jax.random.randint(rng_key, (batch_size,), 0, size(buffer))
    return jax.tree.map(lambda leaf: leaf[indices], buffer.data)
