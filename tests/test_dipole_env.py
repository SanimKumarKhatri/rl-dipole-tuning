"""
Contract tests for DipoleEnv - reset/step return shapes, observation
bounds, and basic determinism. Does not re-test nec_core's physics
(covered by test_nec_core.py) - just that the environment wraps it
correctly per the Gymnasium interface.
"""
import numpy as np
import pytest
from dipole_env import DipoleEnv, MAX_STEPS


def make_env():
    return DipoleEnv(target_freq_mhz=2400.0, wire_radius_m=0.00015,
                      length_min=0.03, length_max=0.09)


def test_reset_returns_correct_shape():
    env = make_env()
    obs, info = env.reset(seed=0)
    assert obs.shape == (2,)
    assert obs.dtype == np.float32
    assert info == {}


def test_reset_length_within_bounds():
    env = make_env()
    for seed in range(10):
        obs, _ = env.reset(seed=seed)
        length = obs[0]
        assert env.length_min <= length <= env.length_max


def test_reset_is_deterministic_given_seed():
    env1 = make_env()
    env2 = make_env()
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)
    assert obs1[0] == pytest.approx(obs2[0])


def test_step_returns_correct_shape():
    env = make_env()
    env.reset(seed=0)
    action = np.array([0.0005], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    assert obs.shape == (2,)
    assert isinstance(reward, float) or isinstance(reward, np.floating)
    assert isinstance(terminated, bool) or isinstance(terminated, np.bool_)
    assert isinstance(truncated, bool)
    assert "length" in info and "vswr" in info


def test_step_clips_length_to_bounds():
    env = make_env()
    env.reset(seed=0)
    env.length = env.length_max
    action = np.array([0.001], dtype=np.float32)  # push past upper bound
    obs, *_ = env.step(action)
    assert obs[0] <= env.length_max


def test_action_clipping_respects_action_space():
    env = make_env()
    env.reset(seed=0)
    length_before = env.length
    oversized_action = np.array([100.0], dtype=np.float32)
    obs, *_ = env.step(oversized_action)
    length_after = obs[0]
    max_possible_move = env.action_space.high[0]
    assert length_after <= length_before + max_possible_move + 1e-6


def test_episode_truncates_at_max_steps():
    env = make_env()
    env.reset(seed=0)
    action = np.array([0.0], dtype=np.float32)
    for _ in range(MAX_STEPS - 1):
        _, _, terminated, truncated, _ = env.step(action)
        if terminated:
            pytest.skip("Episode terminated early via VSWR threshold - "
                        "cannot test truncation boundary from this seed")
    _, _, terminated, truncated, _ = env.step(action)
    assert truncated