"""
Visualize results from the ground-plane-aware dipole simulator:
1. VSWR vs length (ground-truth sweep, as before)
2. Radiation pattern (elevation cut, phi=0) as a polar plot -- the classic
   "gain vs angle" antenna plot
3. Free-space vs ground-plane pattern comparison, overlaid
"""
import numpy as np
import matplotlib.pyplot as plt
from testing_nec import simulate_dipole

FREQ_MHZ = 100.0
WAVELENGTH = 300.0 / FREQ_MHZ
RESONANT_LENGTH = 0.48 * WAVELENGTH


def plot_vswr_sweep():
    fracs = np.linspace(0.44, 0.52, 25)
    vswrs = []
    for frac in fracs:
        r = simulate_dipole(frac * WAVELENGTH, FREQ_MHZ, ground=False)
        vswrs.append(min(r["vswr"], 20))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(fracs, vswrs, "-o", color="#2a6f97", markersize=4)
    ax.axvline(0.48, color="gray", linestyle="--", alpha=0.6, label="theoretical ~0.48*lambda")
    ax.set_xlabel("Dipole length (fraction of lambda)")
    ax.set_ylabel("VSWR")
    ax.set_title(f"VSWR vs length at {FREQ_MHZ} MHz (free space)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("vswr_sweep.png", dpi=150)
    plt.close(fig)
    print("Saved vswr_sweep.png")


def get_elevation_cut(gains_db, thetas, phis, phi_target=0.0):
    """Extract gain values along theta, at a single fixed phi (an 'elevation cut')."""
    phi_idx = int(np.argmin(np.abs(np.array(phis) - phi_target)))
    cut = [gains_db[i][phi_idx] for i in range(len(thetas))]
    # clip sentinel/garbage values for plotting
    cut = [max(g, -20) if (g is not None and g > -900 and g < 100) else -20 for g in cut]
    return np.array(thetas), np.array(cut)


def plot_radiation_pattern_comparison():
    r_free = simulate_dipole(RESONANT_LENGTH, FREQ_MHZ, ground=False)
    r_ground = simulate_dipole(RESONANT_LENGTH, FREQ_MHZ, ground=True)

    th_free, gain_free = get_elevation_cut(r_free["gains_db"], r_free["thetas"], r_free["phis"], phi_target=0.0)
    th_gnd, gain_gnd = get_elevation_cut(r_ground["gains_db"], r_ground["thetas"], r_ground["phis"], phi_target=0.0)

    # Convert theta (0=zenith, 90=horizon) to a polar angle where 0=horizon-right,
    # going up and over -- more intuitive for an antenna elevation pattern.
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})

    theta_rad_free = np.radians(90 - th_free)  # remap so 0=horizon, 90=zenith
    ax.plot(theta_rad_free, gain_free + 20, color="#2a6f97", linewidth=2,
            label=f"Free space (peak {r_free['max_gain_dbi']:.2f} dBi)")

    theta_rad_gnd = np.radians(90 - th_gnd)
    ax.plot(theta_rad_gnd, gain_gnd + 20, color="#d1495b", linewidth=2,
            label=f"Above real ground (peak {r_ground['max_gain_dbi']:.2f} dBi)")

    ax.set_theta_zero_location("E")   # 0 deg = horizon, pointing right
    ax.set_theta_direction(1)
    ax.set_thetamin(0)
    ax.set_thetamax(180)               # only need the upper half (0=horizon .. 180=other horizon)
    ax.set_rlabel_position(135)
    ax.set_title(f"Dipole elevation pattern at {FREQ_MHZ} MHz (phi=0 cut)\n"
                 f"radius = gain (dBi) + 20 offset, dashed circles unmarked = -20/0/10/20 dBi",
                 pad=20)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, -0.15))
    fig.tight_layout()
    fig.savefig("radiation_pattern_comparison.png", dpi=150)
    plt.close(fig)
    print("Saved radiation_pattern_comparison.png")


if __name__ == "__main__":
    plot_vswr_sweep()
    plot_radiation_pattern_comparison()