import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from dipole_env import DipoleEnv, simulate_dipole_vswr, MAX_STEPS
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

TARGET_FREQ_MHZ = 100.0

def make_env():
    return Monitor(DipoleEnv(target_freq_mhz=TARGET_FREQ_MHZ))

def main():
    train_env = DummyVecEnv([make_env])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    model = PPO("MlpPolicy", train_env, verbose=1, n_steps=256, batch_size=64, learning_rate=3e-4)
    
    print("Training PPO agent...")
    model.learn(total_timesteps=150000)

    model.save("dipole_ppo_model")
    train_env.save("vecnormalize.pkl")
    
    print("Evaluating training agent")
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize.load("vecnormalize.pkl", eval_env)
    eval_env.training = False
    eval_env.norm_reward = False  

    obs = eval_env.reset()
    best_length, best_vswr = None, np.inf
    for _ in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        step_info = info[0]
        print(f"length={step_info['length']:.4f} m  VSWWR={step_info['vswr']:.4f}")
        if step_info['vswr'] < best_vswr:
            best_length = step_info['length']
            best_vswr = step_info['vswr']
        if done[0]:
            break
    
    print(f"Best length found = {best_length:.4f} m with VSWR= {best_vswr:.4f} at {TARGET_FREQ_MHZ} MHz")

    wavelength = 300.0 / TARGET_FREQ_MHZ
    theory_length = 0.48*wavelength
    print(f"Theretical half-wave dipole length = {theory_length:.4f} m")

    print("Model saved to dipole_ppo_model.zip")

if __name__ == "__main__":
    main()