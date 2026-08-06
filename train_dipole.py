from stable_baselines3 import PPO
from dipole_env import DipoleEnv

TARGET_FREQ_MHZ = 100.0

def main():
    env = DipoleEnv(target_freq_mhz=TARGET_FREQ_MHZ)

    model = PPO("MlpPolicy", env, verbose=1)

    print("Training PPO agent...")
    model.learn(total_timesteps=10000)

    model.save("dipole_ppo_model")
    print("Model saved.")

if __name__ == "__main__":
    main()