import d4rl
import gym

import numpy as np
import numba


# Cached datasets
DATASETS = {}


@numba.njit
def _calculate_maxstep_and_rtg(rew: np.ndarray, done: np.ndarray):
    # Calculate RTG according to RvS (https://arxiv.org/pdf/2112.10751.pdf)
    N,   = rew.shape

    mask = 1 - done.astype(rew.dtype)

    # calc rtg
    rtg = np.zeros_like(rew)
    rtg[N - 1] = rew[N - 1]
    for i in range(N - 2, -1, -1):
        rtg[i] = rew[i] + mask[i] * rtg[i + 1]

    # calc step
    step = np.zeros_like(rew, dtype=np.int32)
    step[0] = 1
    for i in range(1, N):
        step[i] = 1 + mask[i - 1] * step[i - 1]

    max_episode_steps = np.max(step)

    return max_episode_steps, rtg / (max_episode_steps - step + 1)


def _calculate_rtg_target(env_name, max_episode_steps: int):
    # RTG Table
    RTG_TABLE = {
        # ===== Mujoco
        # 110 for all medium-expert
        "halfcheetah-medium-expert-v2": 110,
        "hopper-medium-expert-v2":      110,
        "walker2d-medium-expert-v2":    110,

        # 90 for all medium(replay), 45 for halfcheetah medium(replay)
        "halfcheetah-medium-v2": 45,
        "hopper-medium-v2":      90,
        "walker2d-medium-v2":    90,

        "halfcheetah-medium-replay-v2": 45,
        "hopper-medium-replay-v2":      90,
        "walker2d-medium-replay-v2":    90,

        # ===== Adroit
        # 70 for all cloned
        "pen-cloned-v1":      70,
        "hammer-cloned-v1":   70,
        "door-cloned-v1":     70,
        "relocate-cloned-v1": 70,

        # 140 for all expert
        "pen-expert-v1":      140,
        "hammer-expert-v1":   140,
        "door-expert-v1":     140,
        "relocate-expert-v1": 140,
    }

    # Get rtg
    assert env_name in RTG_TABLE, f"Unknown environemnt {env_name}"
    norm_rtg = RTG_TABLE[env_name]

    # Convert rtg to average rtg
    rtg = d4rl.reverse_normalized_score(env_name, norm_rtg / 100.0)
    average_rtg = rtg / max_episode_steps

    print(f"Env {env_name} norm_RTG {norm_rtg:.2f} max_step {max_episode_steps} real_RTG {rtg:.2f} average_RTG {average_rtg:.2f}")
    return average_rtg


class MujocoAdroitWrapper(gym.ObservationWrapper):
    def __init__(self, name, env):
        super().__init__(env)
        self.name = name
        self.observation_space = gym.spaces.Box(
            float("-inf"), float("inf"),
            (env.observation_space.shape[-1] + 1,)
        )

        # rtg
        self.use_rtg = True
        if self.name == "door-expert-v1":
            self.use_rtg = False  # FIXME: the policy of door-expert is too monotonous. Do not use reward. BC mode instead.

        self._init_dataset()

    def _init_dataset(self):
        if self.name in DATASETS:
            return

        dataset = self.env.get_dataset()
        # Add real terminals to dataset
        # only "timeouts" is real terminals in
        dataset["real_episode_terminals"] = dataset["terminals"] | dataset["timeouts"]

        rtg_target = None
        if self.use_rtg:
            # Add RTG to dataset
            max_episode_steps, rtg = _calculate_maxstep_and_rtg(
                rew=dataset["rewards"],
                done=dataset["real_episode_terminals"]
            )
            dataset["observations"] = np.concatenate([dataset["observations"], np.expand_dims(rtg, -1)], axis=-1)

            # Get RTG target
            rtg_target = _calculate_rtg_target(self.name, max_episode_steps)

        # Append dataset
        DATASETS[self.name] = {"data": dataset, "rtg_target": rtg_target}

    def observation(self, obs):
        if self.use_rtg:
            # Add rtg to observation
            return np.concatenate([obs, np.expand_dims(DATASETS[self.name]["rtg_target"], -1)], axis=-1)
        
        return obs
    
    def dataset(self):
        return DATASETS[self.name]["data"]

    # Visualization
    def render_plan(self, plan, color_far):
        pass


def is_mujoco_adroit_env(env_name):
    for prefix in ["halfcheetah-", "walker2d-", "hopper-"] + ["pen-", "hammer-", "door-", "relocate-"]:
        if env_name.startswith(prefix):
            return True

    return False


def create_mujoco_adroit_env(env_name):
    return MujocoAdroitWrapper(env_name, gym.make(env_name))
