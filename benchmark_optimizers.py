"""
Compare classical optimization methods against the trained PPO agent
for dipole length tuning at 2400 MHz / 0.00015m wire radius.

Classical methods (golden-section, reactance-bisection, scipy bounded)
are deterministic given fixed bounds - reported as single runs.
PPO's evaluation depends on a random starting length - reported as
mean +/- std across 5 seeds. These are NOT directly comparable in
sample count: classical methods report cold-start calls only; PPO's
row reports inference-only calls, not the 300,000-timestep training
cost paid once, up front, before any of these evaluation runs.
See FINDINGS.md and this script's printed summary for that caveat.
"""
import json
import numpy as np
import matplotlib.pyplot as plt

from optimizers import golden_section_search, reactance_bisection, scipy_bounded_search
from ppo_baseline import PPOBaseline

FREQ_MHZ = 2400.0
WIRE_RADIUS_M = 0.00015
LENGTH_MIN = 0.03
LENGTH_MAX = 0.09
TARGET_VSWR = 1.5
PPO_TRAINING_TIMESTEPS = 300_000  # paid once, not per-evaluation - see docstring
N_PPO_SEEDS = 5


def run_classical_methods():
    results = {}

    gs = golden_section_search(FREQ_MHZ, WIRE_RADIUS_M, LENGTH_MIN, LENGTH_MAX,
                                target_vswr=TARGET_VSWR)
    results["golden_section"] = {
        "n_calls": int(gs["n_calls"]), "n_calls_std": 0.0,
        "best_vswr": float(gs["best_vswr"]), "best_vswr_std": 0.0,
        "converged": bool(gs["converged"]), "n_runs": 1,
    }

    rb = reactance_bisection(FREQ_MHZ, WIRE_RADIUS_M, LENGTH_MIN, LENGTH_MAX,
                              target_vswr=TARGET_VSWR)
    results["reactance_bisection"] = {
        "n_calls": int(rb["n_calls"]), "n_calls_std": 0.0,
        "best_vswr": float(rb["best_vswr"]), "best_vswr_std": 0.0,
        "converged": bool(rb["converged"]), "n_runs": 1,
    }

    sp = scipy_bounded_search(FREQ_MHZ, WIRE_RADIUS_M, LENGTH_MIN, LENGTH_MAX,
                               target_vswr=TARGET_VSWR)
    results["scipy_bounded"] = {
        "n_calls": int(sp["n_calls"]), "n_calls_std": 0.0,
        "best_vswr": float(sp["best_vswr"]), "best_vswr_std": 0.0,
        "converged": bool(sp["converged"]), "n_runs": 1,
    }

    return results


def run_ppo_method():
    ppo = PPOBaseline("dipole_ppo_model_2_4_ghz.zip", "vecnormalize_2_4_ghz.pkl",
                       target_freq_mhz=FREQ_MHZ, wire_radius_m=WIRE_RADIUS_M,
                       length_min=LENGTH_MIN, length_max=LENGTH_MAX)

    n_calls_list, vswr_list, converged_list = [], [], []
    for seed in range(N_PPO_SEEDS):
        result = ppo.evaluate(seed=seed, target_vswr=TARGET_VSWR)
        n_calls_list.append(result["n_calls"])
        vswr_list.append(result["best_vswr"])
        converged_list.append(result["converged"])

    return {
        "ppo_inference": {
            "n_calls": float(np.mean(n_calls_list)),
            "n_calls_std": float(np.std(n_calls_list)),
            "best_vswr": float(np.mean(vswr_list)),
            "best_vswr_std": float(np.std(vswr_list)),
            "converged": bool(all(converged_list)),
            "n_runs": N_PPO_SEEDS,
            "note": f"inference only - excludes {PPO_TRAINING_TIMESTEPS} training timesteps paid once",
        }
    }


def plot_comparison(results, output_path="benchmark_calls_bar.png"):
    methods = list(results.keys())
    means = [results[m]["n_calls"] for m in methods]
    stds = [results[m]["n_calls_std"] for m in methods]

    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8,
                   color="steelblue", edgecolor="navy")

    best_idx = int(np.argmin(means))
    bars[best_idx].set_color("forestgreen")
    bars[best_idx].set_edgecolor("darkgreen")

    ax.set_xlabel("Method")
    ax.set_ylabel("Calls to simulate_dipole (cold-start / inference)")
    ax.set_title(f"Convergence Cost Comparison at {FREQ_MHZ} MHz "
                 f"(target VSWR < {TARGET_VSWR})")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)

    plt.figtext(0.5, -0.05,
                f"Note: ppo_inference excludes {PPO_TRAINING_TIMESTEPS:,} training "
                f"timesteps paid once, up front.",
                ha="center", fontsize=9, style="italic")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    print(f"=== Benchmark: dipole length tuning at {FREQ_MHZ} MHz ===\n")

    results = {}
    results.update(run_classical_methods())
    results.update(run_ppo_method())

    print(f"{'Method':<22} {'Calls':<18} {'Best VSWR':<18} {'Converged'}")
    print("-" * 70)
    for method, r in results.items():
        calls_str = f"{r['n_calls']:.1f} +/- {r['n_calls_std']:.1f}" if r["n_runs"] > 1 else f"{r['n_calls']:.0f}"
        vswr_str = f"{r['best_vswr']:.4f} +/- {r['best_vswr_std']:.4f}" if r["n_runs"] > 1 else f"{r['best_vswr']:.4f}"
        print(f"{method:<22} {calls_str:<18} {vswr_str:<18} {r['converged']}")

    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: benchmark_results.json")

    plot_comparison(results)

    print(f"\nNote: ppo_inference reports inference-only calls (mean of "
          f"{N_PPO_SEEDS} seeded evaluations), excluding the "
          f"{PPO_TRAINING_TIMESTEPS:,}-timestep training cost paid once, up front, "
          f"before any evaluation shown here. Classical methods have zero such cost.")