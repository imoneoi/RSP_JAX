import os
import time
import json

import jax
import jax.numpy as jnp
import numpy as np

import wandb

from tqdm import tqdm
from omegaconf import OmegaConf
from filelock import FileLock

from tseq import TSeqLearner
from vector_env import DummyVectorEnv
from metric_aggregator import MetricAggregator

from envs import *


def evaluate_tseq(tseq, envs, render=False):
    bs = envs.num

    sum_reward         = np.zeros(bs)

    ep_return          = np.zeros(bs)
    ep_count           = np.zeros(bs, np.int32)
    
    obs = envs.reset()
    while True:
        obs_device = jax.device_put(obs)
        act, plan = jax.device_get(tseq.infer(obs_device))

        # step env
        obs, rew, done, _ = envs.step(act)
        if render:
            if len(plan):
                envs.envs[0].render_plan(plan[0], color_far=np.array([1., 0., 0., 1.]))
            envs.envs[0].render()

        # accumulate rewards
        sum_reward         += rew
        ep_return[done]         += sum_reward[done]
        ep_count[done]          += 1

        sum_reward[done]         = 0

        # terminate if all done
        if np.sum(ep_count >= 1) >= bs:
            break

    # divide by number of episodes
    ep_return         /= ep_count

    # Normalize returns
    ep_return         = 100 * envs.get_normalized_score(ep_return)

    return {
        "return":     np.mean(ep_return),
        "return_std": np.std(ep_return),
    }


def train_tseq(conf):
    # Train trajectory world models
    # load configurations
    conf = OmegaConf.merge({
        # seed
        "seed": 0,

        # logging
        "log_group": None,
        "log_metric_every": 100,  # avoid logging a lot data
        "log_eval_metric_save_name": None,
        
        # load model
        "load_model": None, # "models/antmaze-large-play-v2/TSeqV-23:45 02-10 2023/seed_0",

        # hyperparameter
        "max_steps": 20_000,
        "scales": [32],

        # task
        "task": "kitchen-mixed-v0",
        "actor_subsample": None,
        "actor_subsample_scale_step": False,

        "eval_envs": 100
    }, conf)

    # seed numpy
    np.random.seed(conf.seed)

    # env
    if is_antmaze_env(conf.task):
        print("ENV: AntMaze")

        create_env_fn = create_antmaze_env
        conf.deterministic = False
        conf.normalize_obs = False
    elif is_mujoco_adroit_env(conf.task):
        print("ENV: Mujoco/Adroit")

        create_env_fn = create_mujoco_adroit_env
        conf.deterministic = True
        conf.normalize_obs = True
    elif is_kitchen_env(conf.task):
        print("ENV: Kitchen")

        create_env_fn = create_kitchen_env
        conf.deterministic = True
        conf.normalize_obs = True
    else:
        raise NotImplementedError(f"Unknown task {conf.task}")

    eval_envs = DummyVectorEnv([create_env_fn(conf.task) for _ in range(conf.eval_envs)])
    env = eval_envs.envs[0]

    # dataset
    dataset = env.dataset()

    # max steps
    conf.max_steps = {idx: conf.max_steps for idx in range(-1, len(conf.scales))}

    # actor mask
    actor_data_mask = None
    if conf.actor_subsample:
        actor_data_mask = np.load(f"masks/{conf.actor_subsample}/{conf.task}.npy")
    
        if conf.actor_subsample_scale_step:
            conf.max_steps[-1] = int(conf.max_steps[-1] * float(conf.actor_subsample))

    # tseq
    tseq = TSeqLearner(dataset=dataset, actor_data_mask=actor_data_mask, truncate_obs_for_actor_fn=lambda x: x, **OmegaConf.to_container(conf))

    # load model
    if conf.load_model:
        tseq.load(conf.load_model)

        print(evaluate_tseq(tseq, eval_envs, render=True))

    # logging & checkpointing
    if conf.log_group is not None:
        conf.run_name = conf.log_group
        wandb.init(reinit=True, project="SP-" + conf.task, group=conf.run_name, name=str(conf.seed), config=OmegaConf.to_container(conf))
    else:
        conf.run_name = "TSeqV-" + time.strftime("%H:%M %m-%d %Y")
        wandb.init(reinit=True, project="SP-" + conf.task, name=conf.run_name, config=OmegaConf.to_container(conf))

    save_dirname = "models/{}/{}/seed_{}/".format(conf.task, conf.run_name, conf.seed)
    os.makedirs(os.path.dirname(save_dirname), exist_ok=True)

    # train
    for stage_idx in range(len(tseq.trans) - 1, -2, -1):
        # learn this stage
        metric_agg = MetricAggregator(conf.log_metric_every)
        for step in tqdm(range(1, tseq.max_steps[stage_idx] + 1)):
            # train
            metrics = tseq.learn_batch(stage_idx)

            # metric
            metric_agg.update(metrics)
            if not (step % conf.log_metric_every):
                wandb.log(metric_agg.pop())

    # eval
    tseq.save(save_dirname)

    metrics = evaluate_tseq(tseq, eval_envs)
    wandb.log(metrics)

    # save eval metrics
    if conf.log_eval_metric_save_name:
        metric_filename = os.path.join("metrics", f"{conf.log_eval_metric_save_name}.jsonl")
        os.makedirs(os.path.dirname(metric_filename), exist_ok=True)

        # write metrics
        lock = FileLock(metric_filename + ".lock")
        with lock.acquire():
            with open(metric_filename, "a") as f:
                json.dump({
                    "task": conf.task,
                    "metrics": metrics,
                    "conf": OmegaConf.to_container(conf)
                }, f)
                f.write("\n")


if __name__ == "__main__":
    train_tseq(OmegaConf.from_cli())
