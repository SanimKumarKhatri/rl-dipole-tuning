import numpy as np
import matplotlib.pyplot as plt
from PyNEC import nec_context
from dipole_env import simulate_dipole_vswr

def get_radiation_pattern(length_m, freq_mhz=100.0, wire_radius_m=0.001):
    """Compute far-field E-plane pattern (elevation cut) for a center-fed dipole."""
    nec = nec_context()
    geo = nec.get_geometry()
    n_segs = 21
    half_len = length_m / 2.0
    geo.wire(1, n_segs, 0, 0, -half_len, 0, 0, half_len, wire_radius_m, 1.0, 1.0)
    nec.geometry_complete(0)

    nec.gn_card(-1, 0, 0, 0, 0, 0, 0, 0)
    feed_seg = (n_segs + 1) // 2
    nec.ex_card(0, 1, feed_seg, 0, 1.0, 0, 0, 0, 0, 0)
    nec.fr_card(0, 1, freq_mhz, 0)

    # rp_card(calc_mode, n_theta, n_phi, output_format, normalization,
    #          D/A, A/M, theta0, phi0, delta_theta, delta_phi, radial_dist, gain_norm)
    n_theta = 181  # 0-180 deg elevation, 1 deg steps
    nec.rp_card(0, n_theta, 1, 0, 0, 0, 0, 0.0, 0.0, 1.0, 0.0, 0, 0)

    rp = nec.get_radiation_pattern(0)
    theta_deg = rp.get_theta_angles()
    gain_db = rp.get_gain()[:, 0]  # phi=0 cut

    return theta_deg, gain_db

def plot_pattern(length_m, freq_mhz=100.0, save_path="radiation_pattern.png"):
    theta_deg, gain_db = get_radiation_pattern(length_m, freq_mhz)
    theta_rad = np.radians(theta_deg)

    # clip floor for display so nulls don't blow up the polar plot scale
    gain_db_clipped = np.clip(gain_db, -30, None)

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
    ax.plot(theta_rad, gain_db_clipped, linewidth=2)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(f"E-plane pattern — L={length_m:.3f} m @ {freq_mhz:.0f} MHz", pad=20)
    ax.set_rlabel_position(135)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved pattern plot to {save_path}")

if __name__ == "__main__":
    BEST_LENGTH = 1.4295
    TARGET_FREQ_MHZ = 100.0
    plot_pattern(BEST_LENGTH, TARGET_FREQ_MHZ)