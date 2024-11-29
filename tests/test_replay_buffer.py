import chex
import jax.numpy as jnp
import pytest

from ppomdp.sac.replay_buffer import add_batch, init, max_size, size


@pytest.fixture(params=["list", "dict"])
def entry_prototype(request):
    if request.param == "dict":
        return {
            "x": jnp.zeros(2),
            "y": jnp.zeros(1),
        }
    else:
        return [jnp.zeros(2), jnp.zeros(1)]


@pytest.mark.parametrize("buffer_size", [5, 10])
def test_init_and_max_size(entry_prototype, buffer_size):
    buffer = init(entry_prototype, buffer_size)
    assert size(buffer) == 0
    assert max_size(buffer) == buffer_size


def test_add_batch_and_size():
    # Create a simple entry prototype
    entry_prototype = {
        "x": jnp.zeros(2),
        "y": jnp.zeros(1),
    }

    # Initialize buffer
    buffer = init(entry_prototype, 5)

    def create_batch(start, batch_size):
        return {
            "x": jnp.array(
                [[i, i] for i in range(start, start + batch_size)], dtype=jnp.float32
            ),
            "y": jnp.array(
                [[i] for i in range(start, start + batch_size)], dtype=jnp.float32
            ),
        }

    # Test case 1: Adding when the buffer is empty
    batch1 = create_batch(0, 2)
    buffer = add_batch(buffer, batch1)

    assert size(buffer) == 2
    assert buffer.head == 2
    assert buffer.tail == 0
    assert not buffer.full

    # Test case 2: Adding when the buffer has space
    batch2 = create_batch(2, 2)
    buffer = add_batch(buffer, batch2)

    assert size(buffer) == 4
    assert buffer.head == 4
    assert buffer.tail == 0
    assert not buffer.full

    # Test case 3: Adding when the buffer will wrap around
    batch3 = create_batch(4, 3)
    buffer = add_batch(buffer, batch3)

    assert size(buffer) == 5
    assert buffer.head == 0
    assert buffer.tail == 0
    assert buffer.full

    # Verify the contents after wrapping
    expected_x = jnp.array(
        [
            [2, 2],  # From batch2
            [3, 3],
            [4, 4],  # Start of batch3
            [5, 5],
            [6, 6],  # End of batch3
        ]
    )

    expected_y = jnp.array([[2], [3], [4], [5], [6]])
    expected = {"x": expected_x, "y": expected_y}

    # Check if data wrapped correctly
    print(buffer.data)
    chex.assert_trees_all_close(expected, buffer.data)


def test_add_batch_errors():
    # Test with invalid batch shape
    entry_prototype = {"x": jnp.zeros(2)}
    buffer = init(entry_prototype, 5)

    with pytest.raises(AssertionError):
        # Wrong feature dimension
        invalid_batch = {"x": jnp.zeros((2, 3))}
        add_batch(buffer, invalid_batch)

    with pytest.raises(AssertionError):
        # Wrong structure
        invalid_batch = {"y": jnp.zeros(2)}
        add_batch(buffer, invalid_batch)
