from typing import Dict, Callable, Sequence

from jax import Array
from flax import linen as nn
from ppomdp.core import LSTMCarry


class LSTM(nn.Module):
    """
    LSTM module for processing sequences with optional feature extraction and encoding layers.

    Attributes:
        dim (int): Dimensionality of the output.
        feature_fn (Callable): Function to extract features from the input sequence.
        encoder_size (Sequence[int]): Sizes of the encoding layers.
        recurr_size (Sequence[int]): Sizes of the recurrent layers.
        output_size (Sequence[int]): Sizes of the output layers.
        init_log_std (Callable): Initializer for the log standard deviation parameter.
        init_kernel (Callable): Initializer for the kernel weights.
    """
    dim: int
    feature_fn: Callable
    encoder_size: Sequence[int]
    recurr_size: Sequence[int]
    output_size: Sequence[int]
    init_log_std: Callable = nn.initializers.ones
    init_kernel: Callable = nn.initializers.he_uniform()

    @nn.compact
    def __call__(self, carry: list[LSTMCarry], s: Array) -> tuple[list[LSTMCarry], Array]:
        log_std = self.param('log_std', self.init_log_std, self.dim)

        # pass inputs through features layer
        y = self.feature_fn(s)

        # pass features through encoding layers
        for _size in self.encoder_size:
            y = nn.relu(nn.Dense(_size, self.init_kernel)(y))
        y = nn.Dense(self.recurr_size[0])(y)

        # pass encodings through recurrent layers
        for k, _size in enumerate(self.recurr_size):
            carry[k], y = nn.LSTMCell(_size)(carry[k], y)

        # pass result through output layers
        for _size in self.output_size:
            y = nn.relu(nn.Dense(_size, self.init_kernel)(y))
        a = nn.Dense(self.dim)(y)
        return carry, a
