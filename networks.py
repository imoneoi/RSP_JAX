from typing import Sequence, Callable, Optional
import math

import jax
import jax.numpy as jnp
import flax.linen as nn

from tensorflow_probability.substrates import jax as tfp


def default_init(scale: Optional[float] = math.sqrt(2)):
    return nn.initializers.orthogonal(scale)


class MLP(nn.Module):
    hidden_dims: Sequence[int] = (256, 256)
    activation: Callable[[jax.Array], jax.Array] = nn.relu

    dropout: float = 0.0

    @nn.compact
    def __call__(self, x: jax.Array, training: bool) -> jax.Array:
        for dims in self.hidden_dims:
            x = nn.Dense(dims, kernel_init=default_init())(x)
            x = self.activation(x)

            if self.dropout > 0:
                x = nn.Dropout(rate=self.dropout, deterministic=not training)(x)

        return x


class MLPNormal(nn.Module):
    output_dims: int

    hidden_dims: Sequence[int] = (256, 256)
    dropout: float = 0.0

    def setup(self):
        self.backbone  = MLP(hidden_dims=self.hidden_dims, dropout=self.dropout)

        self.mean_head = nn.Dense(self.output_dims, kernel_init=default_init())
        self.log_std   = self.param('log_stds', nn.initializers.zeros, (self.output_dims, ))

    def distribution(self, means: jax.Array) -> jax.Array:
        stds = jnp.exp(self.log_std)

        return tfp.distributions.MultivariateNormalDiag(loc=means, scale_diag=stds)

    def __call__(self, x: jax.Array, training: bool) -> jax.Array:
        # Mean
        x = self.backbone(x, training=training)
        means = self.mean_head(x)

        return self.distribution(means)
