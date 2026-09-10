"""
PPO evaluation wrapper, matching the same result interface as
optimizers.py's classical methods (golden_section_search,
reactance_bisection, scipy_bounded_search) - n_calls, converged,
best_length, best_vswr, history - so all four can be compared directly.

Loads the trained model and VecNormalize stats once; evaluate() can
then be called repeatedly with different seeds (different random
starting lengths) without reloading either.
"""
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from dipole_env import DipoleEnv


class PPOBaseline:
    def __init__(self, model_path, vecnorm_path, target_freq_mhz=2400.0,
                 wire_radius_m=0.00015, length_min=0.03, length_max=0.09,
                 max_steps=150):
        self.model = PPO.load(model_path)
        self.vecnorm_path = vecnorm_path
        self.target_freq_mhz = target_freq_mhz
        self.wire_radius_m = wire_radius_m
        self.length_min = length_min
        self.length_max = length_max
        self.max_steps = max_steps

    def _make_env(self):
        def _init():
            return Monitor(DipoleEnv(
                target_freq_mhz=self.target_freq_mhz,
                wire_radius_m=self.wire_radius_m,
                length_min=self.length_min,
                length_max=self.length_max,
            ))
        return _init

    def evaluate(self, seed, target_vswr=1.5):
        """
        Run one evaluation episode from a seeded random starting length.
        n_calls = environment steps taken, since each step makes exactly
        one nec_core.simulate_dipole call (via DipoleEnv.step/reset).
        """
        raw_env = DummyVecEnv([self._make_env()])
        env = VecNormalize.load(self.vecnorm_path, raw_env)
        env.training = False
        env.norm_reward = False
        env.seed(seed)

        obs = env.reset()
        history = []
        best_vswr = float("inf")
        best_length = None
        n_calls = 1  # reset() itself makes one simulate_dipole call
        # length/vswr come from info dict on step; reset's own vswr
        # isn't directly exposed via VecNormalize's obs (which is
        # normalized) - so we track from info on step() only, and
        # accept the small inaccuracy of not logging reset's own point.

        for step in range(self.max_steps):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            length = info[0]["length"]
            vswr = info[0]["vswr"]
            n_calls += 1
            history.append((length, vswr))

            if vswr < best_vswr:
                best_vswr, best_length = vswr, length

            if vswr < target_vswr or done[0]:
                break

        env.close()

        return {
            "method": "ppo",
            "n_calls": n_calls,
            "converged": best_vswr < target_vswr,
            "best_length": best_length,
            "best_vswr": best_vswr,
            "history": history,
        }