from typing import Any, Dict, Sequence, List, Callable
from functools import partial
import os

import jax
import jax.numpy as jnp

import numba
import numpy as np

import optax
import flax

from flax.training.train_state import TrainState

from networks import MLPNormal
from normalizer import Normalizer


Dataset = Dict[str, jax.Array]
Params = flax.core.FrozenDict[str, Any]


@numba.njit   # JAX jit inefficient here (because tracing 1M ops) !!!
def _multiscale_deltaseq(obs, done, scales):
    # generate waypoints
    N, D = obs.shape
    NS,  = scales.shape

    result = np.zeros((N, NS, D), dtype=obs.dtype)
    episode_boundary = N - 1
    for i in range(N - 1, -1, -1):
        if done[i]:
            episode_boundary = i

        result[i] = obs[np.minimum(i + scales, episode_boundary)] - np.expand_dims(obs[i], 0)

    return result


def _load_dataset(dataset, scales):
    transformed_dataset = dict(
        obs=dataset["observations"],
        act=dataset["actions"],
    )
    if len(scales):
        transformed_dataset["seq"] = _multiscale_deltaseq(dataset["observations"], dataset["real_episode_terminals"], np.array(scales))

    # To device
    return jax.tree_map(lambda x: jax.device_put(x.astype(jnp.float32)), transformed_dataset)


@partial(jax.jit, static_argnames=["batch_size"])
def _sample_dataset(key: Any, dataset: Dataset, batch_size: int):
    dataset_size = next(iter(dataset.values())).shape[0]

    indices = jax.random.randint(key, (batch_size, ), 0, dataset_size)
    return jax.tree_map(lambda x: x[indices], dataset)


@partial(jax.jit, static_argnames=["truncate_obs_for_actor_fn", "deterministic"])
def _infer_jit(key: Any, truncate_obs_for_actor_fn: Callable, trans: Sequence[TrainState], exec: TrainState, obs: jax.Array, deterministic: bool):
    # Sequence plan
    seq = [obs]
    for t in reversed(trans):
        pred = t.apply_fn(t.params, jnp.concatenate(seq, axis=-1), training=False).loc

        seq.append(pred)

    # Execute policy
    # Truncate obs
    seq = [truncate_obs_for_actor_fn(x) for x in seq]

    act = exec.apply_fn(exec.params, jnp.concatenate(seq, axis=-1), training=False)
    if deterministic:
        act = act.loc
    else:
        act = act.sample(seed=key)

    return act, seq


@partial(jax.jit, static_argnames=["truncate_obs_for_actor_fn", "idx"], donate_argnames=["trans", "exec"])
def _learn_jit(key: Any, truncate_obs_for_actor_fn: Callable, idx: int, trans: List[TrainState], exec: TrainState, batch: Dataset):
    # loss fn & grad
    def _mle_loss_fn(params: Params, dropout_key: Any, apply_fn: Callable, x: jax.Array, y: jax.Array):
        dist = apply_fn(params, x, training=True, rngs={"dropout": dropout_key})

        return -dist.log_prob(y).mean()

    _mle_loss_grad = jax.value_and_grad(_mle_loss_fn)

    metrics = {}

    # get seq (all noisy upper level predictions)
    sample_keys = jax.random.split(key, len(trans) - 1 - idx)

    seq = [batch["obs"]]
    for sample_key, next_stage_idx in zip(sample_keys, range(len(trans) - 1, idx, -1)):
        upper_trans = trans[next_stage_idx]
        upper_seq   = batch["seq"][:, next_stage_idx]

        noisy_seq   = upper_trans.apply_fn(upper_trans.params, upper_seq, method=MLPNormal.distribution).sample(seed=sample_key)
        seq.append(noisy_seq)

    if idx >= 0:
        # sequence predictor update
        target = batch["seq"][:, idx]

        # update
        t_loss, t_grads = _mle_loss_grad(trans[idx].params, key, trans[idx].apply_fn, jnp.concatenate(seq, axis=-1), target)
        trans[idx] = trans[idx].apply_gradients(grads=t_grads)

        metrics["t_loss/{}".format(idx)] = t_loss
    else:
        # exec update
        # truncate seq
        seq = [truncate_obs_for_actor_fn(x) for x in seq]
        target = batch["act"]

        exec_loss, exec_grads = _mle_loss_grad(exec.params, key, exec.apply_fn, jnp.concatenate(seq, axis=-1), target)
        exec = exec.apply_gradients(grads=exec_grads)

        metrics["exec_loss"] = exec_loss

    return metrics, trans, exec


def _create_optimizer(lr_schedule, lr, weight_decay, max_steps):
    if lr_schedule == "cosine":
        schedule_fn = optax.cosine_decay_schedule(-lr, max_steps)
        return optax.chain(optax.scale_by_adam(),
                           optax.add_decayed_weights(weight_decay) if weight_decay > 0 else optax.identity(),
                           optax.scale_by_schedule(schedule_fn))
    if lr_schedule == "constant":
        return optax.adam(learning_rate=lr)

    raise NotImplementedError()


class TSeqLearner:
    def __init__(
        self,
        seed,
        dataset,
        actor_data_mask,  # For action-free experiments
        truncate_obs_for_actor_fn,

        # [Hyperparameter] Sequence Learning
        scales: Sequence[int],

        deterministic: bool = False,
        normalize_obs: bool = False,

        # [Hyperparameter] Network
        hidden_dims: Sequence[int] = (1024, 1024),
        dropout:             float = 0.0,

        # [Hyperparameter] Optimization
        max_steps: Dict[int, int] = None,
        batch_size: int = 16384,

        lr: float = 1e-3,
        lr_schedule: str = "cosine",
        weight_decay: float = 0.0,  # 0 = off

        **kwargs
    ):
        # Config
        self.batch_size    = batch_size
        self.max_steps     = max_steps

        self.normalize_obs = normalize_obs
        self.deterministic = deterministic
        self.truncate_obs_for_actor_fn = truncate_obs_for_actor_fn

        # PRNG
        self.key       = jax.random.PRNGKey(seed)
        self.infer_key = jax.random.PRNGKey(seed + 1)

        # dataset
        self.dataset = _load_dataset(dataset, scales)

        if self.normalize_obs:
            self.obs_normalizer = Normalizer.init_state(self.dataset["obs"])
            self.dataset["obs"] = Normalizer.normalize(self.obs_normalizer, self.dataset["obs"])
            if "seq" in self.dataset:
                self.seq_normalizer = Normalizer.init_state(self.dataset["seq"])
                self.dataset["seq"] = Normalizer.normalize(self.seq_normalizer, self.dataset["seq"])

        # dataset info
        example_batch = _sample_dataset(jax.random.PRNGKey(0), self.dataset, self.batch_size)

        obs_dims = example_batch["obs"].shape[-1]
        actor_obs_dims = truncate_obs_for_actor_fn(example_batch["obs"]).shape[-1]
        actor_act_dims = example_batch["act"].shape[-1]

        # actor dataset
        self.actor_dataset = None
        if actor_data_mask is not None:
            actor_data_mask = jax.device_put(actor_data_mask)
            self.actor_dataset = jax.tree_map(lambda x: x[actor_data_mask], self.dataset)

        # trans models
        trans_model = MLPNormal(
            output_dims=obs_dims,
            hidden_dims=hidden_dims,
            dropout=dropout
        )

        num_trans = len(scales)
        models_input_dims = [obs_dims * i for i in range(num_trans, 0, -1)]

        self.key, *trans_keys = jax.random.split(self.key, 1 + num_trans)
        self.trans = [TrainState.create(
            apply_fn=trans_model.apply,
            params=trans_model.init(trans_keys[idx], jnp.zeros((self.batch_size, models_input_dims[idx])), training=False),
            tx=_create_optimizer(lr_schedule, lr, weight_decay, max_steps[idx] if max_steps is not None else 1)
        ) for idx in range(num_trans)]

        # exec model
        exec_model = MLPNormal(
            output_dims=actor_act_dims,
            hidden_dims=hidden_dims,
            dropout=dropout
        )

        self.key, exec_key = jax.random.split(self.key)
        self.exec = TrainState.create(
            apply_fn=exec_model.apply,
            params=exec_model.init(exec_key, jnp.zeros((self.batch_size, (num_trans + 1) * actor_obs_dims)), training=False),
            tx=_create_optimizer(lr_schedule, lr, weight_decay, max_steps[-1] if max_steps is not None else 1)
        )

    def infer(self, obs):
        # normalize obs
        if hasattr(self, "obs_normalizer"):
            obs_norm = Normalizer.normalize(self.obs_normalizer, obs)
        else:
            obs_norm = obs

        # infer
        self.infer_key, subkey = jax.random.split(self.infer_key)
        act, seq = _infer_jit(subkey, self.truncate_obs_for_actor_fn, self.trans, self.exec, obs_norm, self.deterministic)

        # denormalize plan
        # if len(plan):
        #     if hasattr(self, "seq_normalizer"):
        #         plan = Normalizer.denormalize(self.seq_normalizer, plan)
        #     # plan (delta) to global
        #     plan = plan + jnp.expand_dims(obs, 1)

        return act, seq

    def learn_batch(self, stage_idx):
        # sample batch
        self.key, dataset_key, train_key = jax.random.split(self.key, 3)

        if stage_idx < 0 and self.actor_dataset is not None:
            # sample actor dataset
            batch = _sample_dataset(dataset_key, self.actor_dataset, self.batch_size)
        else:
            batch = _sample_dataset(dataset_key, self.dataset, self.batch_size)

        # train
        metrics, self.trans, self.exec = _learn_jit(train_key, self.truncate_obs_for_actor_fn, stage_idx, self.trans, self.exec, batch)
        return metrics

    def load(self, dirname):
        def _load_model(name, model):
            with open(os.path.join(dirname, name), "rb") as f:
                return model.replace(params=flax.serialization.from_bytes(model.params, f.read()))

        for idx in range(len(self.trans)):
            self.trans[idx] = _load_model("trans_{}".format(idx), self.trans[idx])
        self.exec = _load_model("exec", self.exec)

    def save(self, dirname):
        def _save_model(name, model):
            with open(os.path.join(dirname, name), "wb") as f:
                f.write(flax.serialization.to_bytes(model.params))

        for idx, t in enumerate(self.trans):
            _save_model("trans_{}".format(idx), t)
        _save_model("exec", self.exec)

