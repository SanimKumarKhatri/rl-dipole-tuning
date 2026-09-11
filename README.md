# Reinforcement Learning-Based Resonant Frequency Tuning for Wire Dipoles using NEC2
A lightweight RL environment wrapper around NEC2 for automated dipole antenna resonant frequency optimization.

Using PPO (via Stable-Baselines3) to learn the optimal length of a center-fed half-wave dipole antenna, with NEC (PyNEC) as the electromagnetic simulator providing ground-truth impedance and VSWR at each step.

## Why RL for this?

The textbook answer for a half-wave dipole length is well known,  L $\approx$ 0.48 $\times$ $\lambda$, and a simple bisection or grid search over length would converge to a low-VSWR design just fine for this single-parameter case.

We used this as a deliberately small, well-understood problem to build and debug an RL loop against a real EM solver (NEC2) rather than an analytic reward function. The end goal is to extend this to antenna geometries where there isn't a closed-form target (multi-element arrays, loaded elements, non-uniform wire radius) where a search over evaluations against a real solver is closer to what tools like this would actually be useful for. Starting with a dipole, where we can sanity check the RL result against 0.48 $\lambda$, was mainly a way to validate that the following pipeline was correct before moving on a problem without a known-good answer.

```mermaid
flowchart LR
    id1(NEC2) -- (VSWR) --> id2(gym) -- (reward) --> id3(PPO)
    id3(PPO) -- (action) --> id2(gym) -- (length) --> id1(NEC)
```

## Environment (dipole_env.py)
* Observation: [dipole_length_m, current_vswr]
* Action: continuous length adjustment, $\Delta$ length $\epsilon$ [-0.005, 0.005] m per step (100 MHz) or [-0.001, 0.001] m per step (2.4 GHz). The smaller step size at 2.4 GHz is needed because the resonant length is ~24× smaller, a 5 mm step would overshoot the entire tuning range.
* Reward: `-log(VSWR)`: a log reward gives a much smoother gradient than raw `-VSWR`, which spikes hard for detuned lengths and made early training runs unstable.
* Termination: episode ends when `VSWR < 1.05`, or after 150 steps
* Simulator: each step runs a full NEC2 simulation (PyNEC) on a 21-segment wire model at 100 MHz, center-fed, over free space `gn_card(-1, 0, 0, 0, 0, 0, 0, 0)`

The environment is parameterized by target_freq_mhz, length_min, length_max, and the action-space bounds.

## Training (train_dipole.py)
* PPO, MlpPolicy, n_steps=256, batch_size=64
* Wrapped in VecNormalize (norm_obs=True, norm_reward=True), observation scale (length of 1 m vs. VSWR of 1–1000) is wildly mismatched otherwise, and the policy effectively ignored length as a signal without this.
* Normalization stats saved to `vecnormalize.pkl` (100 MHz) or `vecnormalize_2_4_ghz.pkl` (2.4 GHz) and reloaded in an eval environment (training=False), separate from the training environment.

## 100 MHz Results

At 100 MHz, the theoretical half-wave length is:

$\lambda$ = 300 / f_MHz = 3.0 m

L = 0.48 × $\lambda$ = 1.44 m

The trained agent converges to a length **L = 1.4338 m** with VSWR= 1.4305 at 100.0 MHz. This is 0.4306% deviation from the theoretical value. 

This falls short of the environment's intentionally strict termination threshold (VSWR < 1.05) within the fixed episode, but confirms the RL policy is learning to navigate towards true physical resonance rather than a local minimum.

### Training & Convergence

**Training reward over time:**
![Reward curve](reward_curve.png)

Mean episode reward (`rollout/ep_rew_mean`) over 150,000 training timesteps. Reward drops sharply in the first ~2,000 steps as the randomly-initialized policy explores, then climbs steadily and plateaus around -120 by ~50,000
timesteps, indicating stable convergence rather than continued instability.

**Agent trajectory (single evaluation episode):**
![Convergence](convergence.png)

Length and VSWR at each step of a deterministic evaluation episode, starting from a random initial length near 0.98 m. The agent moves length steadily toward the theoretical resonant length (dashed red line), with VSWR dropping from ~100 to near the termination threshold (dashed green line) within the first ~90 steps, then holding in a tight band around resonance for the remainder of the episode.

## 2.4 GHz Results
At 2.4 GHz, the theoretical half-wave length is:

$\lambda$ = 300 / f_MHz = 0.125 m

L = 0.48 × $\lambda$ = 0.0600 m

The trained agent converges to a length **L = 0.0592 m** with VSWR= 1.4447 at 2.4 GHz. This is 1.33% deviation from the theoretical value. 

This falls short of the environment's intentionally strict termination threshold (VSWR < 1.05) within the fixed episode, but confirms the RL policy is learning to navigate towards true physical resonance rather than a local minimum.

### Training & Convergence (2.4 GHz)
**Training reward over time:**
![](./reward_curve_2_4_ghz.png)
Mean episode reward (rollout/ep_rew_mean) over 300,000 training timesteps. Reward drops sharply in the first ~5,000 steps as the randomly-initialized policy explores, then climbs steadily and plateaus around -80 by ~30,000 timesteps, indicating stable convergence rather than continued instability.

**Agent trajectory (single evaluation episode):**
![](./convergence_2_4ghz.png)
Length and VSWR at each step of a deterministic evaluation episode, starting from a random initial length near 0.079 m. The agent moves length steadily toward the theoretical resonant length (dashed red line), with VSWR dropping from ~14 to near the termination threshold (dashed green line) within the first ~20 steps, then holding in a tight band around resonance for the remainder of the episode.

## Why does VSWR plateau above the 1.05 threshold?

Both the 100 MHz and 2.4 GHz results above converge to a stable VSWR (1.43 and 1.44 respectively) that consistently falls short of the 0.05 termination threshold, described above as the agent "navigating toward true physical resonance rather than a local minimum." This section explains the mechanism.

A dipole's feed impedance is `Z = R + jX`. Length controls the reactance `X`, driving it to zero at resonance - but it does not meaningfully change the resonant resistance `R`, which is set by the wire's length-to-radius ratio. For this wire (length/radius ~ 400), `R` at resonance is ~72-75 ohm. VSWR requires both `X ~ 0` and `R ~ 50` ohm simultaneously:

```
Gamma = (R - 50) / (R + 50) = 22 / 122 ~ 0.180
VSWR = (1 + Gamma) / (1 - Gamma) ~ 1.44
```

This matches the observed plateau closely. With length as the only free variable, no length exists that satisfies both conditions - **VSWR < 1.05 is very likely physically unreachable for this bare dipole geometry, independent of training time or algorithm.**

Verified five independent ways, all agreeing to 2-3 significant figures: golden-section search (1.4243), reactance-bisection root-finding on X=0 (1.4347), `scipy.optimize.minimize_scalar` (1.4243), a PPO policy trained for 20k timesteps (1.4245), and the full 300k-timestep run above (1.4266-1.4377 across runs). Achieving true VSWR < 1.05 on a bare dipole would need an actual matching technique (folded dipole, gamma match, L-network) - a different antenna design, not a different length.

## Classical Baseline Comparison

Given the problem is single-parameter (length) and the reward surface is smooth, we benchmarked three classical optimizers against the trained PPO policy at 2.4 GHz:

| Method | Calls to converge | Best VSWR |
|---|---|---|
| Golden-section search | 4 | 1.4484 |
| Reactance-bisection | 18 | 1.4347 |
| scipy.optimize (bounded Brent) | 12 | 1.4243 |
| PPO (inference only, 5 seeds) | 15.8 $\pm$ 9.1 | 1.4541 $\pm$ 0.029 |
| *PPO (training, one-time cost)* | *300,000* | - |

All four converge to the same physical floor. Classical methods match or beat PPO's per-evaluation call count with zero training investment. For this specific task, classical optimization is the more efficient tool - the RL loop's value here is methodological (validating the NEC2/gym/PPO pipeline against a known-good answer, per "Why RL for this?" above) rather than a performance win over simpler alternatives.

## Generalization Across Frequencies

Since the trained policy's only advantage over classical search would be handling new conditions without retraining, we evaluated the 2.4 GHz-trained model, unmodified, on frequencies it never trained on:

| Frequency | True resonant length | PPO best_length | Result |
|---|---|---|---|
| 2450 MHz (near training) | 0.05833 m | ~0.058 m | Mostly converges |
| 2000 MHz | 0.07043 m | ~0.060 m | Fails - defaults to training-frequency length |
| 1800 MHz | 0.07918 m | ~0.060 m | Fails - defaults to training-frequency length |

The policy does not generalize beyond a narrow band around its training frequency. At 2000 and 1800 MHz it converges to ~0.06 m regardless of the true target - the length correct *for 2.4 GHz*, not for the frequency it was actually evaluated against. This is expected given the observation space is `[length, VSWR]` with no frequency signal: the policy has no way to distinguish "VSWR is high because I'm at the wrong length for this frequency" from "VSWR is high because I'm at the wrong length, period." Generalizing across frequencies would require frequency in the observation space and training across a distribution of frequencies, not a fixed one - not yet implemented here.

## Ablation Study (2.4 GHz)

To understand which design choices matter at 2.4 GHz, controlled ablations was ran across five axes: reward function, observation space, action space, normalization, and algorithm. Each variant was trained with **5 random seeds**; values are mean $\pm$ standard deviation across seeds, evaluated over 20 deterministic episodes per seed.

### Reward Function

| Variant | Formula | Mean Error (%) | Mean VSWR |
|---|---|---|---|
| raw_vswr | -VSWR | 6.73 $\pm$ 2.67 | 3.00 $\pm$ 0.93 |
| log_vswr | -log(VSWR) | 7.33 $\pm$ 3.34 | 3.68 $\pm$ 1.53 |
| distance_ideal | -abs(L - 0.06) | 17.48 $\pm$ 3.34 | 27.84 $\pm$ 12.23 |
| exponential | -exp(VSWR) | 32.15 $\pm$ 14.71 | 107.71 $\pm$ 71.60 |

`raw_vswr` and `log_vswr` perform best, within a standard deviation of each other. The log transform was originally introduced to smooth early-training gradients, but the raw VSWR penalty works almost as well for this single-parameter problem. Distance-based and exponential rewards fail because they either ignore the EM simulator's true signal or explode numerically for detuned lengths, exponential's mean training return reaches roughly $-1.2\times10^{10}$, and its 14.71 point standard deviation (vs. 2.67-3.34 for the other three) shows it is not just worse on average but far less consistent across seeds.

![](./ablation_plots/reward_function_bar.png)

Learning curves: 
![](./ablation_plots/reward_function_curves.png)

### Observation Space

| Variant | Observation | Mean Error (%) | Mean VSWR |
|---|---|---|---|
| length_vswr | [length, VSWR] | 8.12 $\pm$ 4.08 | 4.15 $\pm$ 2.22 |
| length_only | [length] | 14.02 $\pm$ 18.10 | 34.29 $\pm$ 63.36 |
| vswr_only | [VSWR] | 23.41 $\pm$ 13.52 | 69.11 $\pm$ 77.11 |
| with_step | [length, VSWR, step_count] | 11.93 $\pm$ 8.49 | 5.42 $\pm$ 3.72 |
| with_gradient | [length, VSWR, d(VSWR)/dL] | 13.97 $\pm$ 8.87 | 28.32 $\pm$ 47.19 |

`length_vswr` (the default) is the clear winner, and also has the tightest spread across seeds. VSWR alone is insufficient because the policy cannot distinguish whether it is above or below resonance without length context, `length_only` and `vswr_only` both carry very high seed-to-seed variance ($\pm 18.10$ and $\pm 13.52$), suggesting these observation spaces are not just worse on average but unreliable run-to-run. Adding step count or a numerical gradient estimate does not help and adds noise.

![](./ablation_plots/observation_space_bar.png)

Learning curves: 
![](./ablation_plots/observation_space_curves.png)

`length_vswr` (green) and `with_gradient` (purple) climb fastest early on, but `with_gradient`'s higher-noise observation costs it accuracy by the end (13.97% final error vs. `length_vswr`'s 8.12%)m a faster-rising curve doesn't translate to a better final policy here. `vswr_only` (blue) lags significantly, confirming that length is a necessary signal for directional tuning.

### Normalization

| Variant | Mean Error (%) | Mean VSWR |
|---|---|---|
| vecnormalize_full | 5.25 $\pm$ 0.87 | 1.90 $\pm$ 0.24 |
| vecnormalize_obs_only | 7.10 $\pm$ 3.38 | 3.60 $\pm$ 1.67 |
| vecnormalize_reward_only | 16.47 $\pm$ 4.42 | 12.78 $\pm$ 2.47 |
| none | 23.32 $\pm$ 6.53 | 83.81 $\pm$ 75.81 |

Full VecNormalize is critical with 4.4 times lower error than no normalization, and a tighter spread ($\pm$0.87 vs. $\pm$6.53). Without observation normalization, the policy ignores length (scale ~0.06 m) relative to VSWR (scale ~1-14). Reward normalization alone actually is worse relative to observation normalization alone (16.47% vs. 7.10%), likely because it rescales the already well-behaved `-log(VSWR)` into a range that destabilizes the value function.

![](./ablation_plots/normalization_bar.png)

Learning curves: 
![](./ablation_plots/normalization_curves.png)

The ablations confirm that the original 2.4 GHz design ([length, VSWR] observation, `-log(VSWR)` reward, continuous $\pm$ 1 mm steps, full VecNormalize, PPO) is well-justified.

## Known Methodological Notes (ablation study)

Two additional findings surfaced during later analysis, relevant to reading the ablation tables above:

- **Seed variance is large at the ablation study's training budget** (10,000 timesteps, 30-step episodes, adversarial initialization). A controlled comparison (same 10k timesteps, 150-step episodes, random init) showed near-zero seed variance (std=0.009 vs 3.07), isolating the short episode length / adversarial reset combination, not PPO or the reward function, as the main driver of the spread reported above.
- **`exponential`'s reward is not actually bounded to a learnable range** - `min(v-1, 20)` caps the exponent, not the output, so `exp(20) ~ 4.85e8` is still reachable per step. Its mean training return (~-1.2e10, noted above) is 7+ orders of magnitude larger than `raw_vswr`/`log_vswr`'s. This scale mismatch likely explains most of `exponential`'s poor, high-variance results independent of whether distance-to-target shaping is conceptually sound.

## Setup
```bash
pip install -r requirements.txt
# 2.4 GHz model
python train_dipole.py
```
