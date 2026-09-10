"""
Physics-correctness tests for nec_core.simulate_dipole.

These encode the manual spot-checks done throughout PR1/PR2/PR5:
known impedance/VSWR/gain values at specific, previously-verified
points. If these fail after a future change to nec_core.py, that
change altered the underlying physics and needs re-validation via
sanity test/testing_nec.py before being trusted.
"""
import pytest
from nec_core import simulate_dipole, compute_n_segs, MIN_SEG_TO_RADIUS_RATIO


def test_free_space_resonance_100mhz():
    """Matches testing_nec.py's original validated free-space case."""
    result = simulate_dipole(0.48 * 3.0, freq_mhz=100.0, wire_radius_m=0.001,
                              ground=False, return_pattern=True)
    assert result["impedance"].real == pytest.approx(72.2, abs=1.0)
    assert result["vswr"] == pytest.approx(1.445, abs=0.05)
    assert result["max_gain_dbi"] == pytest.approx(2.14, abs=0.1)


def test_ground_case_100mhz():
    """Matches testing_nec.py's original validated ground case."""
    result = simulate_dipole(0.48 * 3.0, freq_mhz=100.0, wire_radius_m=0.001,
                              ground=True, return_pattern=True)
    assert result["vswr"] == pytest.approx(1.732, abs=0.05)
    assert result["max_gain_dbi"] == pytest.approx(5.72, abs=0.2)


def test_resonance_2_4ghz_rl_operating_point():
    """
    Matches validate_2_4ghz.py's confirmed result at the RL environment's
    actual operating parameters - the PR2 validation gap this closed.
    """
    result = simulate_dipole(0.06, freq_mhz=2400.0, wire_radius_m=0.00015)
    assert result["impedance"].real == pytest.approx(75.4, abs=2.0)
    assert result["vswr"] == pytest.approx(1.59, abs=0.1)


def test_vswr_floor_near_2_4ghz_resonance():
    """
    Encodes the FINDINGS.md conclusion: VSWR cannot go below ~1.4 for
    this geometry, no matter how close to resonance. This is a sanity
    bound, not a tight assertion - if this ever fails LOW (vswr < 1.35),
    something about the wire geometry or NEC setup has changed and the
    floor finding needs re-verification, not silent acceptance.
    """
    result = simulate_dipole(0.0588, freq_mhz=2400.0, wire_radius_m=0.00015)
    assert result["vswr"] > 1.35
    assert result["vswr"] < 1.55


def test_short_dipole_is_badly_mismatched():
    """A dipole far from resonance should show very poor VSWR."""
    result = simulate_dipole(0.03, freq_mhz=2400.0, wire_radius_m=0.00015)
    assert result["vswr"] > 50


def test_thin_wire_ratio_check_raises_below_floor():
    """MIN_SEG_TO_RADIUS_RATIO enforcement should reject unsafe geometry.

    n_segs=27 forces segment_length/wire_radius below the 8.0 floor:
    (0.03/27)/0.00015 = 7.4, which should raise.
    """
    with pytest.raises(ValueError):
        simulate_dipole(0.03, freq_mhz=2400.0, wire_radius_m=0.00015, n_segs=27)


def test_compute_n_segs_is_odd():
    """Feed segment requires an odd segment count."""
    for length in [0.03, 0.06, 0.09, 1.44]:
        n = compute_n_segs(length, wire_radius_m=0.00015)
        assert n % 2 == 1