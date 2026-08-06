import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from dipole_env import DipoleEnv, simulate_dipole_vswr

TARGET_FREQ_MHZ = 100.0

def make_env():
    return Monitor(DipoleEnv(target_freq_mhz=TARGET_FREQ_MHZ))

def main():
    env = make_env()
    model = PPO("MlpPolicy", env, verbose=1, n_steps=256, batch_size=64, learning_rate=3e-4)

    print("Training PPO agent...")
    model.learn(total_timesteps=20000)

    print("Evaluating training agent")
    obs, _ = env.reset()
    best_length, best_vswr = None, 999.0
    for _ in range(25):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"length={info['length']:.4f} m  VSWWR={info['vswr']:.4f}")
        if info['vswr'] < best_vswr:
            best_length = info['length']
            best_vswr = info['vswr']
        if terminated or truncated:
            break
    
    print(f"Best length found = {best_length:.4f} m with VSWR= {best_vswr:.4f} at {TARGET_FREQ_MHZ} MHz")

    wavelength = 300.0 / TARGET_FREQ_MHZ
    theory_length = 0.48*wavelength
    print(f"Theretical half-wave dipole length = {theory_length:.4f} m, lambda * 0.48 at {TARGET_FREQ_MHZ} MHz")

    model.save("dipole_ppo_model")
    print("Model saved to dipole_ppo_model.zip")

if __name__ == "__main__":
    main()