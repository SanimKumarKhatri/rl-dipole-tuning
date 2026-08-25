import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def load_results(ablation_name, save_dir="./ablation_results"):
    results_path = os.path.join(save_dir, f"{ablation_name}_results.json")
    with open(results_path, "r") as f:
        return json.load(f)


def plot_bar_comparison(ablation_name, metric="mean_error_pct", save_dir="./ablation_results", 
                        output_dir="./ablation_plots"):
    results = load_results(ablation_name, save_dir)

    variants = []
    means = []
    stds = []

    for variant, res in results.items():
        if "error" in res:
            continue
        variants.append(variant)
        means.append(res[metric])
        stds.append(res.get(f"std_{metric.replace('mean_', '')}", 0))

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(variants))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8, 
                   color='steelblue', edgecolor='navy')

    ax.set_xlabel('Variant', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'Ablation Study: {ablation_name.replace("_", " ").title()}', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # Highlight best performer
    best_idx = np.argmin(means) if "error" in metric or "vswr" in metric else np.argmax(means)
    bars[best_idx].set_color('forestgreen')
    bars[best_idx].set_edgecolor('darkgreen')

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{ablation_name}_bar.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")
    return output_path


def plot_all_ablations(save_dir="./ablation_results", output_dir="./ablation_plots"):
    ablations = ["reward_function", "observation_space", "normalization", 
                 "action_space", "algorithm", "network_architecture"]

    for ablation in ablations:
        results_path = os.path.join(save_dir, f"{ablation}_results.json")
        if os.path.exists(results_path):
            print(f"\nPlotting: {ablation}")
            plot_bar_comparison(ablation, save_dir=save_dir, output_dir=output_dir)
        else:
            print(f"Skipping: {ablation} (results not found)")

def plot_learning_curves(ablation_name, save_dir="./ablation_results", output_dir="./ablation_plots"):
    study_dir = os.path.join(save_dir, ablation_name)
    variants = [d for d in os.listdir(study_dir) if os.path.isdir(os.path.join(study_dir, d))]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for variant in variants:
        all_returns = []
        for seed in range(5):
            curve_path = os.path.join(study_dir, variant, f"seed{seed}", "training_curves.npz")
            if os.path.exists(curve_path):
                data = np.load(curve_path)
                returns = data["returns"]
                # Smooth with moving average
                window = 10
                if len(returns) >= window:
                    smoothed = np.convolve(returns, np.ones(window)/window, mode='valid')
                    all_returns.append(smoothed)
        
        if all_returns:
            # Pad to same length
            max_len = max(len(r) for r in all_returns)
            padded = [np.pad(r, (0, max_len - len(r)), mode='edge') for r in all_returns]
            mean_ret = np.mean(padded, axis=0)
            std_ret = np.std(padded, axis=0)
            x = np.arange(len(mean_ret))
            ax.plot(x, mean_ret, label=variant)
            ax.fill_between(x, mean_ret - std_ret, mean_ret + std_ret, alpha=0.2)
    
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title(f"Learning Curves: {ablation_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{ablation_name}_curves.png"), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", type=str, help="Specific ablation to plot")
    parser.add_argument("--save_dir", type=str, default="./ablation_results")
    parser.add_argument("--output_dir", type=str, default="./ablation_plots")

    args = parser.parse_args()

    if args.ablation:
        plot_bar_comparison(args.ablation, save_dir=args.save_dir, output_dir=args.output_dir)
    else:
        plot_all_ablations(args.save_dir, args.output_dir)
        ablations = ["reward_function", "observation_space", "normalization", 
                 "action_space", "algorithm", "network_architecture"]
        for d in ablations:
            plot_learning_curves(d, args.save_dir, args.output_dir)