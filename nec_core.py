"""
Shared NEC2 dipole simulation core.
Single source of truth for both the sanity-check validation (sanity test/testing_nec.py)
and the RL environment (dipole_env.py). Any change here must be re-validated via
sanity test/testing_nec.py before being trusted by training code.
"""
import math
from PyNEC import nec_context

MIN_SEG_TO_RADIUS_RATIO = 8.0  # thin-wire kernel validity floor; warn below this


def compute_n_segs(length_m, wire_radius_m, target_ratio=15.0, min_segs=9, max_segs=41):
    """
    Choose an odd segment count so segment_length / wire_radius stays comfortably
    above the thin-wire kernel's validity floor, without over-segmenting.
    Odd count keeps a clean center segment for the feed point.
    """
    n = round((length_m / target_ratio) / wire_radius_m) if wire_radius_m > 0 else min_segs
    n = max(min_segs, min(max_segs, n))
    if n % 2 == 0:
        n += 1
    return n


def simulate_dipole(length_m, freq_mhz=100.0, wire_radius_m=0.001, z0=50.0,
                     ground=False, height_m=None, return_pattern=False,
                     n_segs=None):
    """
    Simulate a center-fed dipole in NEC2.

    ground=False -> free space, vertical dipole at origin
    ground=True  -> horizontal dipole at height_m over real ground (eps_r=13, sigma=0.005 S/m)
    return_pattern=False -> skip rp_card/xq_card pattern extraction (fast path for RL loops)
    n_segs=None   -> auto-computed via compute_n_segs(); pass explicit int to override

    Returns dict with impedance, vswr, and (if return_pattern) gain/pattern fields.
    """
    half_len = length_m / 2.0

    if n_segs is None:
        n_segs = compute_n_segs(length_m, wire_radius_m)
    ratio = (length_m / n_segs) / wire_radius_m if wire_radius_m > 0 else float("inf")
    if ratio < MIN_SEG_TO_RADIUS_RATIO:
        raise ValueError(
            f"segment_length/wire_radius = {ratio:.2f} is below the thin-wire validity "
            f"floor of {MIN_SEG_TO_RADIUS_RATIO}. length_m={length_m}, "
            f"wire_radius_m={wire_radius_m}, n_segs={n_segs}. Increase n_segs or radius."
        )

    nec = nec_context()
    geo = nec.get_geometry()

    if ground:
        z = height_m if height_m else 0.25 * (300.0 / freq_mhz)
        geo.wire(1, n_segs, -half_len, 0, z, half_len, 0, z, wire_radius_m, 1.0, 1.0)
    else:
        geo.wire(1, n_segs, 0, 0, -half_len, 0, 0, half_len, wire_radius_m, 1.0, 1.0)

    nec.geometry_complete(0)

    if ground:
        nec.gn_card(0, 0, 13.0, 0.005, 0, 0, 0, 0)
    else:
        nec.gn_card(-1, 0, 0, 0, 0, 0, 0, 0)

    feed_seg = (n_segs + 1) // 2
    nec.ex_card(0, 1, feed_seg, 0, 1.0, 0, 0, 0, 0, 0)
    nec.fr_card(0, 1, freq_mhz, 0)

    if return_pattern:
        if ground:
            n_theta, d_theta = 19, 5.0
        else:
            n_theta, d_theta = 37, 5.0
        n_phi, d_phi = 37, 10.0
        nec.rp_card(0, n_theta, n_phi, 0, 5, 0, 0, 0.0, 0.0, d_theta, d_phi, 0, 0)

    nec.xq_card(0)

    ipt = nec.get_input_parameters(0)
    impedance = ipt.get_impedance()[0]
    gamma = abs((impedance - z0) / (impedance + z0))
    vswr = (1 + gamma) / (1 - gamma) if gamma < 0.999 else 999.0

    result = {"impedance": impedance, "vswr": vswr}

    if return_pattern:
        rp = nec.get_radiation_pattern(0)
        gains_db = rp.get_gain()
        thetas = rp.get_theta_angles()
        phis = rp.get_phi_angles()

        max_gain, max_theta, max_phi = -999.0, None, None
        for i, th in enumerate(thetas):
            for j, ph in enumerate(phis):
                g = gains_db[i][j]
                if not math.isfinite(g) or g <= -999.0 or g > 100.0:
                    continue
                if g > max_gain:
                    max_gain, max_theta, max_phi = g, th, ph

        result.update({
            "max_gain_dbi": max_gain,
            "gain_direction": (max_theta, max_phi),
            "gains_db": gains_db,
            "thetas": thetas,
            "phis": phis,
        })

    return result