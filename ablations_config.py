ABLATION_STUDIES = {
    "reward_function": {
        "description": "Test different reward formulations",
        "variants": {
            "raw_vswr": {"reward_type": "raw"},
            "log_vswr": {"reward_type": "log"},
            "quadratic_vswr": {"reward_type": "quadratic"},
            "sqrt_vswr": {"reward_type": "sqrt"},
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
            "none": {"use_vecnormalize": False},
            "vecnormalize_full": {"use_vecnormalize": True, "norm_obs": True, "norm_reward": True},
            "vecnormalize_obs_only": {"use_vecnormalize": True, "norm_obs": True, "norm_reward": False},
            "vecnormalize_reward_only": {"use_vecnormalize": True, "norm_obs": False, "norm_reward": True},
        },
    },

    "action_space": {
        "description": "Test action space formulations",
        "variants": {
            "discrete_5": {"action_type": "discrete", "actions": [-0.01, -0.005, 0.0, 0.005, 0.01]},
            "continuous_narrow": {"action_type": "continuous", "low": -0.002, "high": 0.002},
            "continuous_default": {"action_type": "continuous", "low": -0.005, "high": 0.005},
            "continuous_wide": {"action_type": "continuous", "low": -0.01, "high": 0.01},
            "absolute": {"action_type": "continuous_absolute", "low": 0.9, "high": 1.8},
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
    "frequency_mhz": 100.0,
    "max_steps": 150,
    "total_timesteps": 150_000,
    "n_steps": 256,
    "batch_size": 64,
    "learning_rate": 3e-4,
    "n_envs": 1,
    "n_seeds": 3,
    "eval_episodes": 10,
    "save_dir": "./ablation_results",
    "use_vecnormalize": False,
}