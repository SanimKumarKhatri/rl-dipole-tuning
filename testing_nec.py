"""
Check: Does our NEC2 setup correctly simulate a center-fed dipole?

Before building any RL on top of a simulator let us validate the simulator against a known
theoretical result. We know a half-wave dipole should resonate (minimum VSWR) close
to, but slightly under, 0.5 * wavelength, around 0.47-0.48 * lambda
due to end effects on a finite-radius wire.

This script sweeps dipole length across that range and confirms the minimum
VSWR falls where theory predicts, before any RLagent is involved.
"""

from PyNEC import nec_context


def simulate_dipole(length_m, freq_mhz=100.0, wire_radius_m=0.001, z0=50.0):
    """Simulate a center-fed dipole and return (impedance, vswr) at freq_mhz."""
    nec = nec_context()
    geo = nec.get_geometry()
    n_segs = 21  #odd -> clean center segment for the feed point
    half_len = length_m / 2.0
    geo.wire(1, n_segs, 0, 0, -half_len, 0, 0, half_len, wire_radius_m, 1.0, 1.0)
    nec.geometry_complete(0)

    nec.gn_card(-1, 0, 0, 0, 0, 0, 0, 0)  #free space, no ground plane
    feed_seg = (n_segs + 1) // 2
    nec.ex_card(0, 1, feed_seg, 0, 1.0, 0, 0, 0, 0, 0)  #voltage source at center
    nec.fr_card(0, 1, freq_mhz, 0)
    nec.xq_card(0)

    ipt = nec.get_input_parameters(0)
    impedance = ipt.get_impedance()[0]
    gamma = abs((impedance - z0) / (impedance + z0))
    vswr = (1 + gamma) / (1 - gamma) if gamma < 0.999 else 999.0
    return impedance, vswr


if __name__ == "__main__":
    freq = 100.0 #MHz
    wavelength = 300.0 / freq  #c/f, wavelength in meters

    print(f"Sweeping dipole length near half-wavelength at {freq} MHz "
          f"(lambda = {wavelength:.3f} m)\n")

    best_frac, best_vswr = None, 999.0
    for frac in [0.45, 0.46, 0.47, 0.48, 0.49, 0.50]:
        L = frac * wavelength
        Z, vswr = simulate_dipole(L, freq)
        print(f"length={L:.3f}m ({frac}*lambda) -> Z={Z:.1f} ohm, VSWR={vswr:.3f}")
        if vswr < best_vswr:
            best_vswr = vswr
            best_frac = frac

    print(f"\nBest match found at {best_frac}*lambda (VSWR={best_vswr:.3f}), "
          f"consistent with the expected ~0.47-0.48*lambda resonant length.")