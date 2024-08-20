import d4rl
import gym

import numpy as np

from d4rl.kitchen import kitchen_envs


def _subtask_status(env, obs: np.ndarray):
    qp_len = 9
    next_obj_obs = obs[qp_len:]

    result = np.zeros((len(env.TASK_ELEMENTS, )), dtype=np.bool_)
    for task_id, element in enumerate(env.TASK_ELEMENTS):
        element_idx = kitchen_envs.OBS_ELEMENT_INDICES[element]
        distance = np.linalg.norm(
            next_obj_obs[..., element_idx - qp_len] -
            kitchen_envs.OBS_ELEMENT_GOALS[element])
        result[task_id] = distance < kitchen_envs.BONUS_THRESH

    return result


class KitchenWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            float("-inf"), float("inf"),
            (env.observation_space.shape[-1] + len(env.TASK_ELEMENTS, ), )
        )

    def observation(self, obs):
        # Add goal to observation
        goal = np.ones((len(self.env.TASK_ELEMENTS, ), ), obs.dtype)

        return np.concatenate([obs[:30], goal], axis=-1)

    def dataset(self):
        dataset = self.env.get_dataset()

        # Remove goal state as all goal states in kitchen is the same.
        observations = dataset["observations"][:, :30]
        dones = dataset["timeouts"] | dataset["terminals"]
        N = len(dones)

        # Relabel goal states
        goals = np.zeros((N, len(self.env.TASK_ELEMENTS)), observations.dtype)
        last_goal = None

        assert dones[-1]
        for idx in range(N - 1, -1, -1):
            if dones[idx]:
                last_goal = _subtask_status(self.env, observations[idx])

            goals[idx] = last_goal

        # Check relabel
        assert (np.sum(goals, -1) == dataset["rewards"])[dones].all()

        # Result
        dataset["observations"] = np.concatenate([observations, goals], -1)
        dataset["real_episode_terminals"] = dones
        return dataset

    # Truncate obs for actor
    @staticmethod
    def truncate_obs_for_actor(obs):
        return obs


def is_kitchen_env(env_name):
    return env_name.startswith("kitchen-")


def create_kitchen_env(env_name):
    return KitchenWrapper(gym.make(env_name))
