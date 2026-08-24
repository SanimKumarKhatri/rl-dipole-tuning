import gymnasium as gym
import numpy as np
from gymnasium import spaces
from dipole_env import simulate_dipole_vswr, MAX_STEPS

class DipoleEnvAblation(gym.Env):
    """
    Dipole antenna environment with configurable components for ablation studies.

    Configurable aspects:
    - reward_type: raw, log, quadratic, sqrt, distance, exponential
    - obs_keys: which observations to include
    - action_type: discrete, continuous, continuous_absolute
    - normalization: handled externally via VecNormalize or manual scaling
    """

    def __init__(self, config=None):
        super().__init__()

        self.config = config or {}
        self.target_freq_mhz = self.config.get("frequency_mhz", 100.0)
        self.wavelength = 300.0 / self.target_freq_mhz  # meters
        self.theoretical_length = 0.48 * self.wavelength
        self.length_min = self.config.get("length_min", 0.9)
        self.length_max = self.config.get("length_max", 1.8)
        self.max_steps = self.config.get("max_steps", MAX_STEPS)
        
        # Reward configuration
        self.reward_type = self.config.get("reward_type", "log")
        self.reward_fn = self._get_reward_fn()

        # Observation configuration
        self.obs_keys = self.config.get("obs_keys", ["length", "vswr"])

        # Action configuration
        self.action_type = self.config.get("action_type", "continuous")
        self._setup_action_space()

        # Observation space (dynamic based on obs_keys)
        self.observation_space = self._setup_observation_space()

        # State
        self.current_length = 1.0
        self.current_vswr = 100.0
        self.prev_vswr = None
        self.step_count = 0

        self.wire_radius_m = self.config.get("wire_radius_m", 0.001)

    def _get_reward_fn(self):
        reward_fns = {
            "raw": lambda v: -v,
            "log": lambda v: -np.log(v) if v > 1.0 else 0.0,
            "quadratic": lambda v: -(min(v, 50.0) ** 2),
            "sqrt": lambda v: -np.sqrt(v),
            "distance": lambda v: -((min(v, 50.0) - 1.0) ** 2),
            "exponential": lambda v: -np.exp(min(v - 1.0, 20.0)),
        }
        return reward_fns.get(self.reward_type, reward_fns["log"])

    def _setup_action_space(self):
        if self.action_type == "discrete":
            actions = self.config.get("actions", [-0.01, -0.005, 0.0, 0.005, 0.01])
            self.action_space = spaces.Discrete(len(actions))
            self.discrete_actions = actions
        elif self.action_type == "continuous_absolute":
            low = self.config.get("low", 0.5)
            high = self.config.get("high", 2.5)
            self.action_space = spaces.Box(
                low=np.array([low], dtype=np.float32), 
                high=np.array([high], dtype=np.float32), 
                dtype=np.float32
            )
        else:  
            low = self.config.get("low", -0.005)
            high = self.config.get("high", 0.005)
            self.action_space = spaces.Box(
                low=np.array([low], dtype=np.float32), 
                high=np.array([high], dtype=np.float32), 
                dtype=np.float32
            )

    def _setup_observation_space(self):
        obs_dim = len(self.obs_keys)
        low = np.zeros(obs_dim, dtype=np.float32)
        high = np.ones(obs_dim, dtype=np.float32) * 10.0

        for i, key in enumerate(self.obs_keys):
            if key == "length":
                low[i], high[i] = self.length_min, self.length_max
            elif key == "vswr":
                low[i], high[i] = 1.0, 1000.0
            elif key == "step_count":
                low[i], high[i] = 0.0, 150.0
            elif key == "vswr_gradient":
                low[i], high[i] = -1000.0, 1000.0

        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _run_nec(self):
        _, vswr = simulate_dipole_vswr(
            length_m = self.current_length,
            freq_mhz = self.target_freq_mhz,
            wire_radius_m = self.wire_radius_m,
        )
        return vswr

    def _get_observation(self):
        obs = []
        for key in self.obs_keys:
            if key == "length":
                obs.append(self.current_length)
            elif key == "vswr":
                obs.append(self.current_vswr)
            elif key == "step_count":
                obs.append(float(self.step_count))
            elif key == "vswr_gradient":
                if self.prev_vswr is not None:
                    grad = (self.current_vswr - self.prev_vswr) / 0.001
                else:
                    grad = 0.0
                obs.append(grad)
        return np.array(obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        init_type = self.config.get("init_type", "random")
        if init_type == "random":
            self.current_length = self.np_random.uniform(self.length_min, self.length_max)
        elif init_type == "near_theory":
            self.current_length = self.theoretical_length + self.np_random.uniform(-0.1, 0.1)
        else:  # fixed
            self.current_length = 1.0

        self.step_count = 0
        self.prev_vswr = None
        self.current_vswr = self._run_nec()

        return self._get_observation(), {}

    def step(self, action):
        if self.action_type == "discrete":
            delta = self.discrete_actions[int(action)]
            self.current_length += delta
        elif self.action_type == "continuous_absolute":
            self.current_length = float(action[0])
        else:
            delta = float(action[0])
            self.current_length += delta

        # Clip length
        self.current_length = np.clip(self.current_length, self.length_min, self.length_max)

        # Run NEC2
        self.prev_vswr = self.current_vswr
        self.current_vswr = self._run_nec()

        # Compute reward
        reward = self.reward_fn(self.current_vswr)

        # Check termination
        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.max_steps

        term_threshold = self.config.get("term_threshold", 1.05)
        if self.current_vswr < term_threshold:
            terminated = True

        info = {
            "length": self.current_length,
            "vswr": self.current_vswr,
            "theoretical_length": self.theoretical_length,
            "error_pct": abs(self.current_length - self.theoretical_length) / self.theoretical_length * 100,
        }

        return self._get_observation(), reward, terminated, truncated, info