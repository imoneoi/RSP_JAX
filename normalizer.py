from typing import NamedTuple

import jax
import jax.numpy as jnp


class NormalizerState(NamedTuple):
    mean: jax.Array
    std:  jax.Array


class Normalizer:
    @staticmethod
    def init_state(dataset, eps=1e-4):
        return NormalizerState(
            mean=jnp.mean(dataset, 0, keepdims=True),
            std=jnp.std(dataset,   0, keepdims=True) + eps
        )

    @staticmethod
    @jax.jit
    def normalize(state, x):
        return (x - state.mean) / state.std

    @staticmethod
    @jax.jit
    def denormalize(state, x):
        return x * state.std + state.mean
