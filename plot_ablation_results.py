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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", type=str, help="Specific ablation to plot")
    parser.add_argument("--save_dir", type=str, default="./ablation_results")
    parser.add_argument("--output_dir", type=str, default="./ablation_plots")

    args = parser.parse_args()

    if args.ablation:
        if args.latex:
            generate_latex_table(args.ablation, args.save_dir)
        else:
            plot_bar_comparison(args.ablation, save_dir=args.save_dir, output_dir=args.output_dir)
    else:
        plot_all_ablations(args.save_dir, args.output_dir)