from typing import List, Tuple, Optional
import dataclasses

import jax
import jax.numpy as jnp
from jax import Array
from chex import PRNGKey

import math
import numpy as np
import flax.linen as nn
from flax.linen.module import compact

import distrax
from distrax._src.bijectors.bijector import Bijector
from distrax._src.bijectors.chain import Chain
from distrax._src.bijectors.inverse import Inverse
from distrax._src.distributions.transformed import Transformed
from distrax._src.bijectors.masked_coupling import MaskedCoupling
from distrax._src.utils import math


class TransformedConditional(Transformed):
    def __init__(self, distribution, flow):
        super().__init__(distribution, flow)

    def sample(self, seed: PRNGKey, sample_shape: List[int], context: Optional[Array] = None) -> Array:
        x = self.distribution.sample(seed=seed, sample_shape=sample_shape)
        y, _ = self.bijector.forward_and_log_det(x, context)
        return y

    def log_prob(self, x: Array, context: Optional[Array] = None) -> Array:
        x, ildj_y = self.bijector.inverse_and_log_det(x, context)
        lp_x = self.distribution.log_prob(x)
        lp_y = lp_x + ildj_y
        return lp_y

    def sample_and_log_prob(self, seed: PRNGKey, sample_shape: List[int], context: Optional[Array] = None) -> Tuple[Array, Array]:
        x, lp_x = self.distribution.sample_and_log_prob(seed=seed, sample_shape=sample_shape)
        y, fldj = jax.vmap(self.bijector.forward_and_log_det)(x, context)
        lp_y = jax.vmap(jnp.subtract)(lp_x, fldj)
        return y, lp_y


class InverseConditional(Inverse):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Array, context: Optional[Array] = None) -> Array:
        return self._bijector.inverse(x, context)

    def inverse(self, y: Array, context: Optional[Array] = None) -> Array:
        return self._bijector.forward(y, context)

    def forward_and_log_det(self, x: Array, context: Optional[Array] = None) -> Tuple[Array, Array]:
        return self._bijector.inverse_and_log_det(x, context)

    def inverse_and_log_det(self, y: Array, context: Optional[Array] = None) -> Tuple[Array, Array]:
        return self._bijector.forward_and_log_det(y, context)


class ChainConditional(Chain):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, x: Array, context: Optional[Array] = None) -> Array:
        for bijector in reversed(self._bijectors):
            x = bijector.forward(x, context)
        return x

    def inverse(self, y: Array, context: Optional[Array] = None) -> Array:
        for bijector in self._bijectors:
            y = bijector.inverse(y, context)
        return y

    def forward_and_log_det(self, x: Array, context: Optional[Array] = None) -> Tuple[Array, Array]:
        x, log_det = self._bijectors[-1].forward_and_log_det(x, context)
        for bijector in reversed(self._bijectors[:-1]):
            x, ld = bijector.forward_and_log_det(x, context)
            log_det += ld
        return x, log_det

    def inverse_and_log_det(self, y: Array, context: Optional[Array] = None) -> Tuple[Array, Array]:
        y, log_det = self._bijectors[0].inverse_and_log_det(y, context)
        for bijector in self._bijectors[1:]:
            y, ld = bijector.inverse_and_log_det(y, context)
            log_det += ld
        return y, log_det


class Permute(Bijector):
    def __init__(self, permutation: Array, axis: int = -1):

        super().__init__(event_ndims_in=1)

        self.permutation = jnp.array(permutation)
        self.axis = axis

    def permute_along_axis(self, x: Array, permutation: Array, axis: int = -1) -> Array:
        x = jnp.moveaxis(x, axis, 0)
        x = x[permutation, ...]
        x = jnp.moveaxis(x, 0, axis)
        return x

    def forward_and_log_det(self, x: Array, context: Optional[Array] = None) -> Tuple[Array, Array]:
        y = self.permute_along_axis(x, self.permutation, axis=self.axis)
        return y, jnp.zeros(x.shape[: -self.event_ndims_in])

    def inverse_and_log_det(self, y: Array, context: Optional[Array] = None) -> Tuple[Array, Array]:
        inv_permutation = jnp.zeros_like(self.permutation)
        inv_permutation = inv_permutation.at[self.permutation].set(jnp.arange(len(self.permutation)))
        x = self.permute_along_axis(y, inv_permutation)
        return x, jnp.zeros(y.shape[: -self.event_ndims_in])


class MaskedCouplingConditional(MaskedCoupling):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward_and_log_det(self, x: Array, context: Optional[Array] = None) -> Tuple[Array, Array]:
        self._check_forward_input_shape(x)
        masked_x = jnp.where(self._event_mask, x, 0.0)
        params = self._conditioner(masked_x, context)
        y0, log_d = self._inner_bijector(params).forward_and_log_det(x)
        y = jnp.where(self._event_mask, x, y0)
        logdet = math.sum_last(jnp.where(self._mask, 0.0, log_d), self._event_ndims - self._inner_event_ndims)
        return y, logdet

    def inverse_and_log_det(self, y: Array, context: Optional[Array] = None) -> Tuple[Array, Array]:
        self._check_inverse_input_shape(y)
        masked_y = jnp.where(self._event_mask, y, 0.0)
        params = self._conditioner(masked_y, context)
        x0, log_d = self._inner_bijector(params).inverse_and_log_det(y)
        x = jnp.where(self._event_mask, y, x0)
        logdet = math.sum_last(jnp.where(self._mask, 0.0, log_d), self._event_ndims - self._inner_event_ndims)
        return x, logdet


class Conditioner(nn.Module):
    event_shape: int
    context_shape: int
    hidden_dims: List[int]
    num_bijector_params: int

    @compact
    def __call__(self, x: Array, context: Array):
        x = jnp.hstack([context, x])

        for hidden_dim in self.hidden_dims:
            x = nn.relu(nn.Dense(hidden_dim)(x))
        x = nn.Dense(
            self.event_shape * self.num_bijector_params,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros
        )(x)

        x = x.reshape(-1, self.event_shape, self.num_bijector_params)
        return x


class NeuralSplineFlow(nn.Module):
    """Based on the implementation in the Distrax repo, https://github.com/deepmind/distrax/blob/master/examples/flow.py"""

    n_dim: int
    n_context: int = 0
    n_transforms: int = 4
    hidden_dims: List[int] = dataclasses.field(default_factory=lambda: [128, 128])
    n_bins: int = 16
    range_min: float = -1.0
    range_max: float = 1.0

    def setup(self):
        def bijector_fn(params: Array):
            return distrax.RationalQuadraticSpline(
                params, range_min=self.range_min, range_max=self.range_max
            )

        event_shape = self.n_dim
        context_shape = self.n_context

        # Alternating binary mask
        mask = jnp.arange(0, np.prod(event_shape)) % 2
        mask = jnp.reshape(mask, event_shape)
        mask = mask.astype(bool)

        # Number of parameters for the rational-quadratic spline:
        # - `num_bins` bin widths
        # - `num_bins` bin heights
        # - `num_bins + 1` knot slopes
        # for a total of `3 * num_bins + 1` parameters
        num_bijector_params = 3 * self.n_bins + 1

        self.conditioner = [
            Conditioner(
                event_shape=event_shape,
                context_shape=context_shape,
                hidden_dims=self.hidden_dims,
                num_bijector_params=num_bijector_params,
                name="conditioner_{}".format(i)
            )
            for i in range(self.n_transforms)
        ]

        bijectors = []
        for i in range(self.n_transforms):
            bijectors.append(MaskedCouplingConditional(
                mask=mask, bijector=bijector_fn, conditioner=self.conditioner[i])
            )
            mask = jnp.logical_not(mask)  # Flip the mask after each layer

        self.bijector = InverseConditional(ChainConditional(bijectors))
        self.base_dist = distrax.MultivariateNormalDiag(jnp.zeros(event_shape), jnp.ones(event_shape))
        self.flow = TransformedConditional(self.base_dist, self.bijector)

    def __call__(self, x: Array, context: Array = None) -> Array:
        return self.flow.log_prob(x, context=context)

    def sample(self, num_samples: int, rng: Array, context: Array = None) -> Array:
        return self.flow.sample(seed=rng, sample_shape=(num_samples,), context=context)




import optax
import sklearn
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from tqdm import tqdm


n_samples = 100_000

x, _ = sklearn.datasets.make_moons(n_samples=n_samples, noise=.06)

scaler = sklearn.preprocessing.StandardScaler()
x = scaler.fit_transform(x)

plt.hist2d(x[:, 0], x[:, 1], bins=100, cmap="cividis")
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.show()


def rotate_points(points, angle):
    """Rotate points by angle around origin.
    """
    angle = jnp.deg2rad(angle)
    rot_matrix = jnp.array([[jnp.cos(angle), -jnp.sin(angle)], [jnp.sin(angle), jnp.cos(angle)]])
    return jnp.dot(points, rot_matrix)


# Rotate points by 60 degrees
x_rot = jax.vmap(rotate_points)(x[None, :], jnp.array([60.]))[0]
plt.hist2d(x_rot[:, 0], x_rot[:, 1], bins=100, cmap="cividis")
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.show()

n_dim = 2
n_context = 1  # Single context parameter

# model = MaskedAutoregressiveFlow(n_dim=n_dim, n_context=n_context, hidden_dims=[128,128], n_transforms=12, activation="tanh", use_random_permutations=False)
model = NeuralSplineFlow(
    n_dim=n_dim, n_context=n_context, hidden_dims=[256, 256],
    n_transforms=8, n_bins=16, range_min=-2, range_max=2,
)

# Initialize model
key = jax.random.PRNGKey(42)
x_test = jax.random.uniform(key=key, shape=(64, n_dim))
context = jax.random.uniform(key=key, shape=(64, n_context))
params = model.init(key, x_test, context)

# Log-prob and sampling
log_prob = model.apply(params, x_test, jnp.ones((x_test.shape[0], n_context)))
samples = model.apply(params, n_samples, key, jnp.ones((n_samples, n_context)), method=model.sample)

optimizer = optax.adam(learning_rate=3e-4)
opt_state = optimizer.init(params)


def loss_fn(params, x, context):
  loss = -jnp.mean(model.apply(params, x, context))
  return loss


@jax.jit
def update(params, opt_state, batch):
    x, context = batch
    grads = jax.grad(loss_fn)(params, x, context)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state


batch_size = 64
n_steps = 10_000

for step in tqdm(range(n_steps)):
    # Sample a batch
    x_batch = x[np.random.choice(np.arange(len(x)), size=batch_size, replace=False)]

    # Rotate each pair of points by a random angle from 0 to pi
    rot = np.random.uniform(low=0., high=np.pi, size=batch_size)
    x_batch = jax.vmap(rotate_points)(x_batch, np.rad2deg(rot))

    # Conditioning context based on rotation angle
    context = jnp.array(rot)[:, None]

    # Update
    batch = (x_batch, context)
    params, opt_state = update(params, opt_state, batch)


def sample_from_flow(rot=45., n_samples=10_000, key=jax.random.PRNGKey(42)):
    """Helper function to sample from the flow model.
    """
    def sample_fn(model):
        x_samples = model.sample(num_samples=n_samples, rng=key, context=np.deg2rad(rot) * jnp.ones((n_samples, 1)))
        return x_samples

    x_samples = nn.apply(sample_fn, model)(params)
    return x_samples


fig = plt.figure(figsize=(12, 6))
gs = GridSpec(2, 4)

rot_list = np.linspace(0., 180., 8)

n_bins = 100

for i in tqdm(range(8)):
    key, subkey = jax.random.split(key)

    x_samples = sample_from_flow(rot=rot_list[i], n_samples=100_000, key=key)
    ax = fig.add_subplot(gs[i])

    ax.hist2d(
        x_samples[:, 0], x_samples[:, 1],
        bins=[np.linspace(-2, 2, 30), np.linspace(-2, 2, n_bins)],
        cmap='cividis'
    )

    # Add text to the upper right
    ax.text(
        0.95, 0.95, "Rotation: {0:.1f}°".format(rot_list[i]),
        verticalalignment='top', horizontalalignment='right',
        transform=ax.transAxes, color='white'
    )

    # Set x and y limits
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    # Turn off the tick labels
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
