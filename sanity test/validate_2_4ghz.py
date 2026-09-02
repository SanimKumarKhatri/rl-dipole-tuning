"""
Validate the unified NEC core at the RL environment's actual operating point:
2400 MHz, wire_radius=0.00015m (vs. the original sanity check's 100 MHz, 0.001m).

Reuses plot_result_test.py's plotting functions with different parameters,
rather than duplicating them. Same
fractional-of-wavelength sweep methodology (0.44-0.52 lambda) as the
original 100 MHz validation, for direct comparability between the two.
"""
from testing_nec import simulate_dipole
from plot_result_test import plot_vswr, plot_2d_polar_comparison, plot_3d_comparison

FREQ_MHZ = 2400.0
WIRE_RADIUS_M = 0.00015
WAVELENGTH = 300.0 / FREQ_MHZ
RESONANT_LENGTH = 0.48 * WAVELENGTH

if __name__ == "__main__":
    r_free = simulate_dipole(RESONANT_LENGTH, FREQ_MHZ, WIRE_RADIUS_M,
                              ground=False, return_pattern=True)
    r_ground = simulate_dipole(RESONANT_LENGTH, FREQ_MHZ, WIRE_RADIUS_M,
                                ground=True, return_pattern=True)

    print(f"=== Validation at {FREQ_MHZ} MHz, wire_radius={WIRE_RADIUS_M} m (RL operating point) ===")
    print(f"Free Space:  Z={r_free.get('impedance', 0):.1f} Ω | VSWR={r_free.get('vswr', 0):.2f} | Peak Gain={r_free.get('max_gain_dbi', 0.0):.2f} dBi")
    print(f"Real Ground: Z={r_ground.get('impedance', 0):.1f} Ω | VSWR={r_ground.get('vswr', 0):.2f} | Peak Gain={r_ground.get('max_gain_dbi', 0.0):.2f} dBi")

    plot_vswr(freq_mhz=FREQ_MHZ, wire_radius_m=WIRE_RADIUS_M,
              output_path="vswr_sweep_2_4ghz.png")
    plot_2d_polar_comparison(r_free, r_ground, output_path="radiation_pattern_2d_2_4ghz.png")
    plot_3d_comparison(r_free, r_ground, output_path="radiation_pattern_3d_2_4ghz.png")

    print("All 2.4 GHz validation plots generated successfully!")