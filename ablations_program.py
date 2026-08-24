import os
import sys
import json
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from ablations_config import ABLATION_STUDIES, DEFAULT_CONFIG
from dipole_env_ablation import DipoleEnvAblation


def make_env(config):
    def _init():
        env = DipoleEnvAblation(config)
        env = Monitor(env)
        return env
    return _init


def evaluate_policy(model, env, n_eval_episodes=10):
    episode_lengths = []
    episode_rewards = []
    final_errors = []
    final_vswrs = []
    final_lengths = []

    for _ in range(n_eval_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            done = dones[0]
            info = infos[0]
            episode_reward += reward[0]
            episode_length += 1

        episode_lengths.append(episode_length)
        episode_rewards.append(episode_reward)
        final_errors.append(info["error_pct"])
        final_vswrs.append(info["vswr"])
        final_lengths.append(info["length"])

    return {
        "mean_length": float(np.mean(episode_lengths)),
        "std_length": float(np.std(episode_lengths)),
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_error_pct": float(np.mean(final_errors)),
        "std_error_pct": float(np.std(final_errors)),
        "mean_vswr": float(np.mean(final_vswrs)),
        "std_vswr": float(np.std(final_vswrs)),
        "mean_final_length": float(np.mean(final_lengths)),
        "std_final_length": float(np.std(final_lengths)),
    }


def run_single_experiment(ablation_name, variant_name, variant_config, 
                          seed=0, save_dir="./ablation_results"):
    env_config = {**DEFAULT_CONFIG, **variant_config}
    env_config["seed"] = seed

    exp_dir = os.path.join(save_dir, ablation_name, variant_name, f"seed{seed}")
    os.makedirs(exp_dir, exist_ok=True)

    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(env_config, f, indent=2)

    env = DummyVecEnv([make_env(env_config)])

    # Apply normalization if configured
    norm_config = env_config
    if norm_config.get("use_vecnormalize", False):
        env = VecNormalize(
            env,
            norm_obs=norm_config.get("norm_obs", True),
            norm_reward=norm_config.get("norm_reward", True),
            clip_obs=10.0,
        )

    # Get algorithm
    algo_name = variant_config.get("algo", "PPO")
    algo_map = {"PPO": PPO, "A2C": A2C, "SAC": SAC, "TD3": TD3}
    algo_class = algo_map.get(algo_name, PPO)

    # Network architecture
    net_arch = variant_config.get("net_arch", [64, 64])
    policy_kwargs = {"net_arch": net_arch}

    # Create model
    model = algo_class(
        "MlpPolicy",
        env,
        learning_rate=DEFAULT_CONFIG["learning_rate"],
        n_steps=DEFAULT_CONFIG["n_steps"],
        batch_size=DEFAULT_CONFIG["batch_size"],
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=seed,
        tensorboard_log=os.path.join(exp_dir, "tensorboard"),
    )

    # Train
    model.learn(total_timesteps=DEFAULT_CONFIG["total_timesteps"])

    # Save model
    model.save(os.path.join(exp_dir, "model"))
    if norm_config.get("use_vecnormalize", False):
        env.save(os.path.join(exp_dir, "vecnormalize.pkl"))

    # Evaluate
    eval_env = DummyVecEnv([make_env(env_config)])
    if norm_config.get("use_vecnormalize", False):
        eval_env = VecNormalize.load(
            os.path.join(exp_dir, "vecnormalize.pkl"), 
            eval_env
        )
        eval_env.training = False
        eval_env.norm_reward = False

    results = evaluate_policy(model, eval_env, n_eval_episodes=DEFAULT_CONFIG["eval_episodes"])

    # Save results
    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    env.close()
    eval_env.close()

    return results


def run_ablation_study(ablation_name, n_seeds=3, save_dir="./ablation_results"):
    """Run all variants of an ablation study."""

    if ablation_name not in ABLATION_STUDIES:
        print(f"Error: Unknown ablation '{ablation_name}'")
        print(f"Available: {list(ABLATION_STUDIES.keys())}")
        return None

    study = ABLATION_STUDIES[ablation_name]
    print(f"\n{'='*60}")
    print(f"ABLATION STUDY: {ablation_name}")
    print(f"Description: {study['description']}")
    print(f"Variants: {list(study['variants'].keys())}")
    print(f"Seeds per variant: {n_seeds}")
    print(f"{'='*60}\n")

    all_results = {}

    for variant_name, variant_config in study["variants"].items():
        print(f"\nVariant: {variant_name}")
        print(f"Config: {json.dumps(variant_config, indent=2)}")
        variant_results = []

        for seed in range(n_seeds):
            print(f"  Seed {seed}...", end=" ", flush=True)
            try:
                result = run_single_experiment(
                    ablation_name, variant_name, variant_config, 
                    seed=seed, save_dir=save_dir
                )
                variant_results.append(result)
                print(f"Error: {result['mean_error_pct']:.4f}% | VSWR: {result['mean_vswr']:.4f}")
            except Exception as e:
                print(f"FAILED: {e}")
                variant_results.append({"error": str(e)})

        # Aggregate across seeds (only successful runs)
        successful = [r for r in variant_results if "error" not in r]
        if successful:
            all_results[variant_name] = {
                "mean_error_pct": float(np.mean([r["mean_error_pct"] for r in successful])),
                "std_error_pct": float(np.std([r["mean_error_pct"] for r in successful])),
                "mean_vswr": float(np.mean([r["mean_vswr"] for r in successful])),
                "std_vswr": float(np.std([r["mean_vswr"] for r in successful])),
                "mean_reward": float(np.mean([r["mean_reward"] for r in successful])),
                "std_reward": float(np.std([r["mean_reward"] for r in successful])),
                "n_successful": len(successful),
                "seeds": variant_results,
            }
        else:
            all_results[variant_name] = {"error": "All seeds failed"}

    # Save aggregated results
    os.makedirs(save_dir, exist_ok=True)
    results_path = os.path.join(save_dir, f"{ablation_name}_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print(f"\n{'='*45}")
    print(f"SUMMARY: {ablation_name}")
    print(f"{'='*45}")
    print(f"{'Variant':<25} {'Error (%)':<15} {'VSWR':<12}")
    print(f"{'-'*45}")
    for variant, res in all_results.items():
        if "error" in res:
            print(f"{variant:<25} FAILED")
        else:
            print(f"{variant:<25} {res['mean_error_pct']:>7.4f}±{res['std_error_pct']:<5.4f} "
                  f"{res['mean_vswr']:>6.4f}±{res['std_vswr']:<4.4f} ")

    print(f"\nResults saved to: {results_path}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies for RL dipole antenna")
    parser.add_argument("--ablation", type=str, required=True,
                       help="Name of ablation study to run")
    parser.add_argument("--n_seeds", type=int, default=3,
                       help="Number of random seeds per variant")
    parser.add_argument("--save_dir", type=str, default="./ablation_results",
                       help="Directory to save results")

    args = parser.parse_args()

    results = run_ablation_study(
        args.ablation, 
        n_seeds=args.n_seeds,
        save_dir=args.save_dir
    )

    if results is None:
        sys.exit(1)


if __name__ == "__main__":
    main()