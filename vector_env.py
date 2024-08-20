import numpy as np


class DummyVectorEnv:
    def __init__(self, envs):
        self.envs = envs
        self.num = len(envs)

    @staticmethod
    def _concat_obs(obs_list):
        return np.concatenate([np.expand_dims(obs, 0) for obs in obs_list], axis=0)

    def reset(self):
        return self._concat_obs([env.reset() for env in self.envs])

    def step(self, act):
        obs_list  = []
        rew_list  = []
        done_list = []
        for env_id, env in enumerate(self.envs):
            obs, rew, done, _ = env.step(act[env_id])
            if done:
                obs = env.reset()

            obs_list.append(obs)
            rew_list.append(rew)
            done_list.append(done)

        return self._concat_obs(obs_list), np.array(rew_list), np.array(done_list), None

    def __getattr__(self, name):
        return getattr(self.envs[0], name)
