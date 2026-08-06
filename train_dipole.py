from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from dipole_env import DipoleEnv

TARGET_FREQ_MHZ = 100.0

def make_env():
    return Monitor(DipoleEnv(target_freq_mhz=TARGET_FREQ_MHZ))

def main():
    env = make_env()
    model = PPO("MlpPolicy", env, verbose=1, n_steps=356, batch_size=64, learning_rate=3e-4)

    print("Training PPO agent...")
    model.learn(total_timesteps=10000)

    print("Evaluating training agent")
    obs, _ = env.reset()
    for _ in range(25):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"length={info['length']:.4f} m  VSWWR={info['vswr']:.4f}")
        if terminated or truncated:
            break

    model.save("dipole_ppo_model")
    print("Model saved.")

if __name__ == "__main__":
    main()