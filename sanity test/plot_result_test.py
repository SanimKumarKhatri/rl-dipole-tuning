import matplotlib.pyplot as plt
import numpy as np
from testing_nec import simulate_dipole

FREQ_MHZ = 100.0
WAVELENGTH = 300.0 / FREQ_MHZ
RESONANT_LENGTH = 0.48 * WAVELENGTH

def extract_cut(r_data, target_phi=0.0):
    """Helper to safely extract a 1D elevation slice at phi=0 from the NEC simulation data."""
    gains = np.array(r_data["gains_db"])
    thetas = np.array(r_data["thetas"])
    phis = np.array(r_data["phis"])

    phi_idx = int(np.argmin(np.abs(phis - target_phi)))
    cut = gains[:, phi_idx]

    # Clean sentinels (-900 / invalid NEC data)
    cut = np.where((cut > -100) & (cut < 100), cut, -20)
    return thetas, cut

def plot_vswr():
    """Plot the 2D VSWR vs dipole length (fraction of wavelength) for a range of lengths."""
    fracs = np.linspace(0.44, 0.52, 25)
    vswrs = []
    for f in fracs:
        vswrs.append(min(simulate_dipole(f * WAVELENGTH, FREQ_MHZ, ground=False)["vswr"], 20))

    plt.figure(figsize=(6, 4))
    plt.plot(fracs, vswrs, "-o", color="#2a6f97", ms=4)
    plt.axvline(0.48, color="gray", linestyle="--", label="Theoretical (~0.48*lambda)")
    plt.xlabel("Length (Fraction of Lambda)")
    plt.ylabel("VSWR")
    plt.title(f"VSWR vs Dipole Length ({FREQ_MHZ} MHz)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("vswr_sweep.png", dpi=150)
    plt.close()

def plot_2d_polar_comparison(r_free, r_ground):
    """Plot the 2D radiation pattern comparison for free space vs above ground"""
    th_free, gain_free = extract_cut(r_free)
    th_gnd, gain_gnd = extract_cut(r_ground)

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})

    # Map zenith angles to horizon angles (0 = Horizon)
    ax.plot(
        np.radians(90 - th_free),
        gain_free,
        label=f"Free Space ({r_free['max_gain_dbi']:.1f} dBi)",)
    ax.plot(
        np.radians(90 - th_gnd),
        gain_gnd,
        label=f"Above Ground ({r_ground['max_gain_dbi']:.1f} dBi)",)

    ax.set_rorigin(-20)
    ax.set_rlim(-20, 10)

    ax.set_thetamin(0)
    ax.set_thetamax(180)  # Upper hemisphere view
    ax.set_theta_zero_location("E")
    ax.set_title("Elevation Pattern (phi = 0)", pad=15)
    ax.legend(loc="lower center")
    plt.tight_layout()
    plt.savefig("radiation_pattern_2d.png", dpi=150)
    plt.close()


def plot_3d_balloon(r_data, ax, title):
    """Plot 3d radiation pattern as a balloon plot"""
    gains = np.array(r_data["gains_db"])
    thetas = np.radians(r_data["thetas"])
    phis = np.radians(r_data["phis"])

    # Stitch phi boundary if missing 360 wrap
    if phis[-1] < 2 * np.pi - 0.1:
        phis = np.append(phis, 2 * np.pi)
        gains = np.hstack([gains, gains[:, [0]]])

    TH, PH = np.meshgrid(thetas, phis, indexing="ij")

    # Scale radius relative to peak gain
    peak = np.nanmax(np.where(gains > -100, gains, -100))
    radius = np.clip(gains - peak, -30, 0) + 30

    # Spherical to Cartesian
    X = radius * np.sin(TH) * np.cos(PH)
    Y = radius * np.sin(TH) * np.sin(PH)
    Z = radius * np.cos(TH)

    norm_colors = plt.cm.viridis((gains - (peak - 30)) / 30)
    ax.plot_surface(
        X,
        Y,
        Z,
        facecolors=norm_colors,
        rstride=1,
        cstride=1,
        antialiased=True,
    )
    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])


def plot_3d_comparison(r_free, r_ground):
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    plot_3d_balloon(r_free, ax1, f"Free Space ({r_free['max_gain_dbi']:.1f} dBi)")
    plot_3d_balloon(r_ground, ax2, f"Above Ground ({r_ground['max_gain_dbi']:.1f} dBi)")

    plt.savefig("radiation_pattern_3d.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    r_free = simulate_dipole(RESONANT_LENGTH, FREQ_MHZ, ground=False, return_pattern=True)
    r_ground = simulate_dipole(RESONANT_LENGTH, FREQ_MHZ, ground=True, return_pattern=True)

    print(f"Free Space:  Z={r_free.get('impedance', 0):.1f} Ω | VSWR={r_free.get('vswr', 0):.2f} | Peak Gain={r_free.get('max_gain_dbi', 0.0):.2f} dBi")
    print(f"Real Ground: Z={r_ground.get('impedance', 0):.1f} Ω | VSWR={r_ground.get('vswr', 0):.2f} | Peak Gain={r_ground.get('max_gain_dbi', 0.0):.2f} dBi")

    plot_vswr()
    plot_2d_polar_comparison(r_free, r_ground)
    plot_3d_comparison(r_free, r_ground)
    print("All plots generated successfully!")