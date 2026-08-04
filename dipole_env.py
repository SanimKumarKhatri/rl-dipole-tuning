# Environment for training a center-fed dipole antenna using NEC2 simulations.
import numpy as np
from PyNEC import nec_context

def simulate_dipole_vswr(length_m, freq_mhz=100.0, wire_radius_m=0.001, z0=50.0):
    """Run one NEC2 simulation and return (impedance, vswr) for a center-fed dipole."""
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
    nec.xq_card(0)

    ipt = nec.get_input_parameters(0)
    impedance = ipt.get_impedance()[0]
    gamma = abs((impedance - z0) / (impedance + z0))
    vswr = (1 + gamma) / (1 - gamma) if gamma < 0.999 else 999.0
    return impedance, vswr

if __name__ == "__main__":
    # Test draft simulation function
    z, vswr = simulate_dipole_vswr(1.5, freq_mhz=100.0)
    print(f"Test Run @ 1.5m -> Impedance: {z}, VSWR: {vswr:.4f}")