import gym
class MarioRewardWrapper(gym.Wrapper):
    """
    Shape reward to encourage moving right and discourage standing still.
    - Reward for delta x (horizontal progress)
    - Small time penalty
    - Bonus for finishing the level
    - Truncate episode if stuck near same x for too long
    """
    def __init__(self, env, stuck_steps_max=400):
        super().__init__(env)
        self._last_x = 0
        self._stuck_steps = 0
        self._stuck_steps_max = stuck_steps_max

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        # Support both obs or (obs, info)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}

        self._last_x = info.get("x_pos", 0)
        self._stuck_steps = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        x = info.get("x_pos", self._last_x)
        dx = x - self._last_x
        self._last_x = x

        # --- reward shaping ---
        shaped = 0.1 * dx            # reward for progress
        shaped += 0.01 * reward      # keep some of original reward
        shaped -= 0.01               # time penalty

        if terminated and info.get("flag_get", False):
            shaped += 50.0           # big bonus for finishing

        # --- "stuck" detection: no forward progress for too long ---
        if dx <= 0:
            self._stuck_steps += 1
        else:
            self._stuck_steps = 0

        if self._stuck_steps >= self._stuck_steps_max:
            truncated = True

        return obs, shaped, terminated, truncated, info
