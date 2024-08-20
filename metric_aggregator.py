from typing import NamedTuple
from functools import partial

import jax
import jax.numpy as jnp


class MetricAggregatorState(NamedTuple):
    buffer:  jax.Array
    index:   jax.Array

    @staticmethod
    def create(length):
        return MetricAggregatorState(
            buffer=jnp.zeros(length),
            index=jnp.zeros((), jnp.int32)
        )

    @staticmethod
    def update(state, metric):
        return MetricAggregatorState(
            buffer=state.buffer.at[state.index].set(metric),
            index=state.index + 1
        )

    @staticmethod
    def mean(state):
        return state.buffer.mean()

    @staticmethod
    def clear(state):
        return MetricAggregatorState(
            buffer=jnp.zeros_like(state.buffer),
            index=jnp.zeros_like(state.index)
        )


@partial(jax.jit, donate_argnums=(0,))
def _update_metric_agg_jit(state, metrics):
    return jax.tree_map(MetricAggregatorState.update, state, metrics, is_leaf=lambda x: isinstance(x, MetricAggregatorState))


@partial(jax.jit, donate_argnums=(0,))
def _pop_metric_agg_jit(state):
    mean  = jax.tree_map(MetricAggregatorState.mean,  state, is_leaf=lambda x: isinstance(x, MetricAggregatorState))
    state = jax.tree_map(MetricAggregatorState.clear, state, is_leaf=lambda x: isinstance(x, MetricAggregatorState))

    return state, mean


class MetricAggregator:
    def __init__(self, length: int):
        self.index = 0
        self.length = length
        self.state = None

    def update(self, metric):
        assert self.index < self.length

        if self.state is None:
            self.state = jax.tree_map(lambda m: MetricAggregatorState.create(self.length), metric)

        self.state = _update_metric_agg_jit(self.state, metric)
        self.index += 1

    def pop(self, to_cpu: bool = True):
        assert self.index == self.length

        self.index = 0
        self.state, mean = _pop_metric_agg_jit(self.state)

        if to_cpu:
            mean = jax.device_get(mean)
        return mean
