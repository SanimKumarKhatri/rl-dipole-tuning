"""
Check: Does our NEC2 setup correctly simulate a center-fed dipole?

Before building any RL on top of a simulator let us validate the simulator against a known
theoretical result. We know a half-wave dipole should resonate (minimum VSWR) close
to, but slightly under, 0.5 * wavelength, around 0.47-0.48 * lambda
due to end effects on a finite-radius wire.

This script sweeps dipole length across that range and confirms the minimum
VSWR falls where theory predicts, before any RLagent is involved.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nec_core import simulate_dipole

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