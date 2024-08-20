import d4rl
import gym

import numpy as np


class AntmazeWrapper(gym.ObservationWrapper):
    @staticmethod
    def pos_dims():
        return 2

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            float("-inf"), float("inf"),
            (env.observation_space.shape[-1] + self.pos_dims(),)
        )

    def observation(self, obs):
        # Add goal to observation
        goal = self.env.unwrapped._goal

        return np.concatenate([obs, goal], axis=-1)

    def dataset(self):
        dataset = self.env.get_dataset()

        # Add real terminals to dataset
        # only "timeouts" is real terminals in
        dataset["real_episode_terminals"] = dataset["timeouts"]

        # Add goal to observations
        dataset["observations"] = np.concatenate([dataset["observations"], dataset["infos/goal"]], axis=-1)
        return dataset

    # Truncate obs for actor
    @staticmethod
    def truncate_obs_for_actor(obs):
        return obs[..., :-2]

    # Visualization
    def render(self):
        self.env.render()

        del self.env.viewer._markers[:]

    def render_plan(self, plan, color_far):
        if self.env.viewer is None:
            self.render()

        # color
        color_near = np.array([1., 1., 1., 1.])

        # add marker
        H, D = plan.shape
        for idx in range(H):
            p = (idx + 1) / H
            color = p * color_far + (1 - p) * color_near

            self.env.viewer.add_marker(pos=np.array([*self.get_pos_from_obs(plan[idx]), 1]), label="",
                                    type=2, size=np.array([0.1, 0.1, 0.1]), rgba=color
            )


def is_antmaze_env(env_name):
    return env_name.startswith("antmaze-")


def create_antmaze_env(env_name):
    return AntmazeWrapper(gym.make(env_name))


def test_antmaze_env(env_name):
    env = create_antmaze_env(env_name)

    # Assert 1: (online) pos & goal ok
    obs = env.reset()
    assert np.isclose(env.get_pos_from_obs(obs),  env.unwrapped.get_xy()).all()
    assert np.isclose(env.get_goal_from_obs(obs), env.unwrapped._goal).all()
