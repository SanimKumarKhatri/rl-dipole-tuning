import pandas as pd
import matplotlib.pyplot as plt

def plot_reward_curve(csv_path, save_path="reward_curve.png"):
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["Step"], df["Value"], linewidth=1.5, color="steelblue")

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Training Reward (rollout/ep_rew_mean)")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved reward curve to {save_path}")

plot_reward_curve("PPO_1.csv")