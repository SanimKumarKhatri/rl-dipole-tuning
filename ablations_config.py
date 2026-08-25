ABLATION_STUDIES = {
    "reward_function": {
        "description": "Test different reward formulations",
        "variants": {
            "raw_vswr": {"reward_type": "raw"},
            "log_vswr": {"reward_type": "log"},
            "distance_ideal": {"reward_type": "distance"},
            "exponential": {"reward_type": "exponential"},
        },
    },

    "observation_space": {
        "description": "Test different observation formulations",
        "variants": {
            "length_only": {"obs_keys": ["length"]},
            "vswr_only": {"obs_keys": ["vswr"]},
            "length_vswr": {"obs_keys": ["length", "vswr"]},
            "with_step": {"obs_keys": ["length", "vswr", "step_count"]},
            "with_gradient": {"obs_keys": ["length", "vswr", "vswr_gradient"]},
        },
    },

    "normalization": {
        "description": "Test normalization strategies",
        "variants": {
            "none": {"use_vecnormalize": False, "norm_obs": False, "norm_reward": False},
            "vecnormalize_full": {"use_vecnormalize": True, "norm_obs": True, "norm_reward": True},
            "vecnormalize_obs_only": {"use_vecnormalize": True, "norm_obs": True, "norm_reward": False},
            "vecnormalize_reward_only": {"use_vecnormalize": True, "norm_obs": False, "norm_reward": True},
        },
    },

    "action_space": {
        "description": "Test action space formulations",
        "variants": {
            "discrete_5": {"action_type": "discrete", "actions": [-0.005, -0.002, 0.0, 0.002, 0.005]},
            "continuous_narrow": {"action_type": "continuous", "low": -0.0005, "high": 0.0005},
            "continuous_default": {"action_type": "continuous", "low": -0.002, "high": 0.002},
            "continuous_wide": {"action_type": "continuous", "low": -0.005, "high": 0.005},
            "absolute": {"action_type": "continuous_absolute"},
        },
    },

    "algorithm": {
        "description": "Test different RL algorithms",
        "variants": {
            "ppo": {"algo": "PPO"},
            "a2c": {"algo": "A2C"},
            "sac": {"algo": "SAC"},
            "td3": {"algo": "TD3"},
        },
    },

    "network_architecture": {
        "description": "Test network sizes",
        "variants": {
            "small": {"net_arch": [32, 32]},
            "default": {"net_arch": [64, 64]},
            "medium": {"net_arch": [128, 128]},
            "large": {"net_arch": [256, 256]},
            "deep": {"net_arch": [64, 64, 64]},
            "bottleneck": {"net_arch": [128, 64]},
        },
    },
}

DEFAULT_CONFIG = {
    "frequency_mhz": 2400.0,
    "length_min": 0.03,
    "length_max": 0.09,
    "wire_radius_m": 0.00015,
    "low": -0.001,
    "high": 0.001,
    "max_steps": 30,
    "total_timesteps": 10_000,
    "n_steps": 128,
    "batch_size": 32,
    "learning_rate": 3e-4,
    "n_envs": 1,
    "n_seeds": 5,
    "eval_episodes": 20,
    "save_dir": "./ablation_results",
    "use_vecnormalize": True,
    "norm_obs": True,
    "norm_reward": False,
    "vswr_noise_std": 0.3,
    "init_type": "adversarial",
}