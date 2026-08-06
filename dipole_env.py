# Environment for training a center-fed dipole antenna using NEC2 simulations.
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from PyNEC import nec_context

def simulate_dipole_vswr(length_m, freq_mhz=100.0, wire_radius_m=0.001, z0=50.0):
    """Run one NEC2 simulation and return (impedance, vswr) for a center-fed dipole."""
    nec = nec_context()
    geo = nec.get_geometry()
    n_segs = 21
    half_len = length_m / 2.0
    geo.wire(1, n_segs, 0, 0, -half_len, 0, 0, half_len, wire_radius_m, 1.0, 1.0)
    nec.geometry_complete(0)

    nec.gn_card(-1, 0, 0, 0, 0, 0, 0, 0)
    feed_seg = (n_segs + 1) // 2
    nec.ex_card(0, 1, feed_seg, 0, 1.0, 0, 0, 0, 0, 0)
    nec.fr_card(0, 1, freq_mhz, 0)
    nec.xq_card(0)

    ipt = nec.get_input_parameters(0)
    impedance = ipt.get_impedance()[0]
    gamma = abs((impedance - z0) / (impedance + z0))
    vswr = (1 + gamma) / (1 - gamma) if gamma < 0.999 else 999.0
    return impedance, vswr

class DipoleEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, target_freq_mhz=100.0, wire_radius_m=0.001,
                 length_min=0.9, length_max=1.8, max_steps=25):
        super().__init__()
        self.target_freq_mhz = target_freq_mhz
        self.wire_radius_m = wire_radius_m
        self.length_min = length_min
        self.length_max = length_max
        self.max_steps = max_steps

        self.action_space = spaces.Box(low=-0.05, high=0.05, shape=(1,), dtype=np.float32)
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
        _, vswr = simulate_dipole_vswr(self.length, self.target_freq_mhz, self.wire_radius_m)
        return np.array([self.length, vswr], dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        delta = float(np.clip(action[0], -0.05, 0.05))
        self.length = float(np.clip(self.length + delta, self.length_min, self.length_max))

        _, vswr = simulate_dipole_vswr(self.length, self.target_freq_mhz, self.wire_radius_m)
        
        reward = -vswr
        terminated = vswr < 1.05
        truncated = self.steps >= self.max_steps

        obs = np.array([self.length, vswr], dtype=np.float32)
        info = {"length": self.length, "vswr": vswr}
        return obs, reward, terminated, truncated, info

if __name__ == "__main__":
    # Test draft simulation function
    z, vswr = simulate_dipole_vswr(1.5, freq_mhz=100.0)
    print(f"Test Run @ 1.5m -> Impedance: {z}, VSWR: {vswr:.4f}")