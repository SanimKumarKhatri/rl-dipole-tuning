import numpy as np
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor
from dipole_env import DipoleEnv, MAX_STEPS
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

TARGET_FREQ_MHZ = 2400.0
LENGTH_MIN = 0.03
LENGTH_MAX = 0.09
WIRE_RADIUS_M = 0.00015

def plot_agent_trajectory(steps, lengths, vswrs, theory_length, output_path="convergence_2_4ghz.png"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(steps, lengths, marker="o", markersize=4, linewidth=1.5, color="steelblue")
    ax1.axhline(y=theory_length, color="red", linestyle="--", linewidth=2,
                label=f"Theoretical L = {theory_length:.4f} m")
    ax1.set_ylabel("Length (m)", fontsize=12)
    ax1.set_title("Agent trajectory: length and VSWR per step (2.4 GHz)", fontsize=14)
    ax1.legend(loc="lower right", fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, vswrs, marker="o", markersize=4, linewidth=1.5, color="darkorange")
    ax2.axhline(y=1.05, color="green", linestyle="--", linewidth=2,
                label="Termination threshold (VSWR=1.05)")
    ax2.set_xlabel("Step", fontsize=12)
    ax2.set_ylabel("VSWR", fontsize=12)
    ax2.legend(loc="upper right", fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

def make_env():
    return Monitor(DipoleEnv(
        target_freq_mhz=TARGET_FREQ_MHZ,
        length_min= LENGTH_MIN,
        length_max= LENGTH_MAX,
        wire_radius_m= WIRE_RADIUS_M,
        ))

def main():
    train_env = DummyVecEnv([make_env])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    model = PPO("MlpPolicy", train_env, verbose=1, n_steps=256, batch_size=64, learning_rate=3e-4, tensorboard_log="tb_logs")

    print("Training PPO agent...")
    model.learn(total_timesteps=300000)

    model.save("dipole_ppo_model_2_4_ghz")
    train_env.save("vecnormalize_2_4_ghz.pkl")
    
    print("Evaluating training agent")
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize.load("vecnormalize_2_4_ghz.pkl", eval_env)
    eval_env.training = False
    eval_env.norm_reward = False  

    obs = eval_env.reset()
    best_length, best_vswr = None, np.inf

    steps, lengths, vswrs = [], [], []

    for step in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        step_info = info[0]
        print(f"length={step_info['length']:.4f} m  VSWWR={step_info['vswr']:.4f}")

        steps.append(step)
        lengths.append(step_info['length'])
        vswrs.append(step_info['vswr'])

        if step_info['vswr'] < best_vswr:
            best_length = step_info['length']
            best_vswr = step_info['vswr']
        if done[0]:
            break
    
    print(f"Best length found = {best_length:.4f} m with VSWR= {best_vswr:.4f} at {TARGET_FREQ_MHZ} MHz")

    wavelength = 300.0 / TARGET_FREQ_MHZ
    theory_length = 0.48*wavelength
    print(f"Theretical half-wave dipole length = {theory_length:.4f} m")

    print("Model saved to dipole_ppo_model_2_4_ghz.zip")

    plot_agent_trajectory(steps, lengths, vswrs, theory_length)

if __name__ == "__main__":
    main()