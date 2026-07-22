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
import math

def simulate_dipole(length_m, freq_mhz=100.0, wire_radius_m=0.001, z0=50.0, height_m=None, ground=False):
    """
    Simulate a center-fed dipole.

    height_m: if set (and ground=True), the dipole is placed horizontally
              at this height above ground instead of at the origin.
    ground:   False -> free space (as before)
              True  -> real, finite ground plane (average earth: eps_r=13, sigma=0.005 S/m)
 
    Returns: dict with impedance, vswr, max_gain_dbi, gain_direction (theta, phi)
    """
    nec = nec_context()
    geo = nec.get_geometry()
    
    n_segs = 21  #odd -> clean center segment for the feed point
    half_len = length_m / 2.0
    
    if ground:
        # Horizontal dipole at height_m above ground, running along x-axis.
        # z must be > 0 and non-zero, or NEC treats it as touching ground.
        z = height_m if height_m else 0.25 * (300.0 / freq_mhz)  # default: quarter-wave up
        geo.wire(1, n_segs, -half_len, 0, z, half_len, 0, z, wire_radius_m, 1.0, 1.0)
    else:
        # Vertical dipole centered at origin, free space (same as before)
        geo.wire(1, n_segs, 0, 0, -half_len, 0, 0, half_len, wire_radius_m, 1.0, 1.0)

    nec.geometry_complete(0)

    if ground:
        # gn_card(ground_type, n_radials, epse, sig, ...)
        # ground_type=0 -> finite ground, Sommerfeld/Norton approximation (accurate, slower)
        # n_radials=0   -> no radial-wire ground screen (just flat lossy ground)
        # epse=13.0     -> relative permittivity (average earth)
        # sig=0.005     -> conductivity in S/m (average earth)
        nec.gn_card(0, 0, 13.0, 0.005, 0, 0, 0, 0)
    else:
        nec.gn_card(-1, 0, 0, 0, 0, 0, 0, 0)  # free space, as before

    feed_seg = (n_segs + 1) // 2
    nec.ex_card(0, 1, feed_seg, 0, 1.0, 0, 0, 0, 0, 0)  #voltage source at center
    nec.fr_card(0, 1, freq_mhz, 0)

    # request a radiation pattern before executing ---
    # rp_card(calc_mode, n_theta, n_phi,
    #         output_format, normalization, D/A conversion, average,
    #         theta0, phi0, delta_theta, delta_phi, radial_distance, gain_norm)
    #
    # calc_mode=0     -> normal mode (full 3D pattern over the ranges given)
    # n_theta, n_phi  -> how many angle steps to compute (resolution)
    # theta0=0        -> start at zenith (0 deg from +z axis), theta=90 is the horizon
    # phi0=0          -> start at 0 deg azimuth
    # delta_theta/phi -> step size in degrees
    #
    # IMPORTANT: with real ground present, NEC2 only computes the upper half-space
    # (theta 0-90 deg, i.e. above the horizon) -- points "underground" (theta>90)
    # are physically meaningless and come back as sentinel/garbage values, so we
    # must not sweep past 90 deg when ground=True.
    if ground:
        n_theta, d_theta = 19, 5.0   # theta: 0 to 90 deg in 5 deg steps
    else:
        n_theta, d_theta = 37, 5.0   # theta: 0 to 180 deg (full sphere, free space)
    n_phi, d_phi = 37, 10.0          # phi: 0 to 360 deg in 10 deg steps
    nec.rp_card(0, n_theta, n_phi, 0, 5, 0, 0, 0.0, 0.0, d_theta, d_phi, 0, 0)

    nec.xq_card(0)

    ipt = nec.get_input_parameters(0)
    impedance = ipt.get_impedance()[0]
    gamma = abs((impedance - z0) / (impedance + z0))
    vswr = (1 + gamma) / (1 - gamma) if gamma < 0.999 else 999.0

    # --- NEW: pull the radiation pattern and find peak gain ---
    rp = nec.get_radiation_pattern(0)
    gains_db = rp.get_gain()          # 2D array [theta_index][phi_index], in dBi
    thetas = rp.get_theta_angles()    # degrees
    phis = rp.get_phi_angles()        # degrees
 
    # -999.99 is NEC2's sentinel for "not computed / invalid point" (e.g. exactly
    # on the horizon in some configurations); skip it and anything non-finite.
    max_gain = -999.0
    max_theta, max_phi = None, None
    for i, th in enumerate(thetas):
        for j, ph in enumerate(phis):
            g = gains_db[i][j]
            if not math.isfinite(g) or g <= -999.0 or g > 100.0:
                continue
            if g > max_gain:
                max_gain, max_theta, max_phi = g, th, ph

    return {
        "impedance": impedance,
        "vswr": vswr,
        "max_gain_dbi": max_gain,
        "gain_direction": (max_theta, max_phi),
        "gains_db": gains_db,
        "thetas": thetas,
        "phis": phis,
    }


if __name__ == "__main__":
    freq = 100.0 #MHz
    wavelength = 300.0 / freq  #c/f, wavelength in meters

    print("=== Free-space dipole ===")
    result = simulate_dipole(0.48 * wavelength, freq, ground=False)
    print(f"Z={result['impedance']:.1f} ohm, VSWR={result['vswr']:.3f}, "
          f"peak gain={result['max_gain_dbi']:.2f} dBi "
          f"at theta={result['gain_direction'][0]:.0f}, phi={result['gain_direction'][1]:.0f}")
 
    print("\n=== Horizontal dipole, quarter-wave above real ground ===")
    result_g = simulate_dipole(0.48 * wavelength, freq, ground=True)
    print(f"Z={result_g['impedance']:.1f} ohm, VSWR={result_g['vswr']:.3f}, "
          f"peak gain={result_g['max_gain_dbi']:.2f} dBi "
          f"at theta={result_g['gain_direction'][0]:.0f}, phi={result_g['gain_direction'][1]:.0f}")