# Reinforcement Learning-Based Resonant Frequency Tuning for Wire Dipoles using NEC2
A lightweight RL environment wrapper around NEC2 for automated dipole antenna resonant frequency optimization.

Using PPO (via Stable-Baselines3) to learn the optimal length of a center-fed half-wave dipole antenna, with NEC (PyNEC) as the electromagnetic simulator providing ground-truth impedance and VSWR at each step.

## Why RL for this?

The textbook answer for a half-wave dipole length is well known,  L $\approx$ 0.48 $\times$ $\lambda$, and a simple bisection or grid search over length would converge to a low-VSWR design just fine for this single-parameter case.

I used this as a deliberately small, well-understood problem to build and debug an RL loop against a real EM solver (NEC2) rather than an analytic reward function. The end goal is to extend this to antenna geometries where there isn't a closed-form target (multi-element arrays, loaded elements, non-uniform wire radius) where a search over evaluations against a real solver is closer to what tools like this would actually be useful for. Starting with a dipole, where I can sanity check the RL result against 0.48 $\lambda$, was mainly a way to validate that the following pipeline was correct before moving on a problem without a known-good answer.

```mermaid
flowchart LR
    id1(NEC2) -- (VSWR) --> id2(gym) -- (reward) --> id3(PPO)
    id3(PPO) -- (action) --> id2(gym) -- (length) --> id1(NEC)
```

## Environment (dipole_env.py)
* Observation: [dipole_length_m, current_vswr]
* Action: continuous length adjustment, $\Delta$ length $\epsilon$ [-0.005, 0.005] m per step
* Reward: `-log(VSWR)`: a log reward gives a much smoother gradient than raw `-VSWR`, which spikes hard for detuned lengths and made early training runs unstable.
* Termination: episode ends when `VSWR < 1.05`, or after 150 steps
* Simulator: each step runs a full NEC2 simulation (PyNEC) on a 21-segment wire model at 100 MHz, center-fed, over free space `gn_card(-1, 0, 0, 0, 0, 0, 0, 0)`

## Training (train_dipole.py)
* PPO, MlpPolicy, n_steps=256, batch_size=64
* Wrapped in VecNormalize (norm_obs=True, norm_reward=True), observation scale (length of 1 m vs. VSWR of 1–1000) is wildly mismatched otherwise, and the policy effectively ignored length as a signal without this.
* Normalization stats saved to `vecnormalize.pkl` and reloaded in an eval environment (training=False), separate from the training environment.

## Results

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

## Setup
```bash
pip install -r requirements.txt
python train_dipole.py
```