# Environment for training a center-fed dipole antenna using NEC2 simulations.
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from nec_core import simulate_dipole


MAX_STEPS = 150

class DipoleEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, target_freq_mhz=2400.0, wire_radius_m=0.00015,
                 length_min=0.03, length_max=0.09, max_steps=MAX_STEPS):
        super().__init__()
        self.target_freq_mhz = target_freq_mhz
        self.wire_radius_m = wire_radius_m
        self.length_min = length_min
        self.length_max = length_max
        self.max_steps = max_steps

        self.action_space = spaces.Box(low=-0.001, high=0.001, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([self.length_min, 1.0], dtype=np.float32),
            high=np.array([self.length_max, 1000.0], dtype=np.float32),
        )

        self.length = None
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.length = float(self.np_random.uniform(self.length_min, self.length_max))
        result = simulate_dipole(self.length, self.target_freq_mhz, self.wire_radius_m)
        vswr = result["vswr"]
        return np.array([self.length, vswr], dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        delta = float(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))
        self.length = float(np.clip(self.length + delta, self.length_min, self.length_max))

        result = simulate_dipole(self.length, self.target_freq_mhz, self.wire_radius_m)
        vswr = result["vswr"]

        reward = -np.log(vswr)
        terminated = vswr < 1.05
        truncated = self.steps >= self.max_steps

        obs = np.array([self.length, vswr], dtype=np.float32)
        info = {"length": self.length, "vswr": vswr}
        return obs, reward, terminated, truncated, info