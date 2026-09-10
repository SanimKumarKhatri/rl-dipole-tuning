"""
Tests for optimizers.py - confirms all three classical methods find
the physically-verified VSWR floor (~1.42-1.45, see FINDINGS.md and
test_nec_core.py's test_vswr_floor_near_2_4ghz_resonance) at the RL
environment's operating point.
"""
import pytest
from optimizers import golden_section_search, reactance_bisection, scipy_bounded_search

FREQ_MHZ = 2400.0
WIRE_RADIUS_M = 0.00015
LENGTH_MIN = 0.03
LENGTH_MAX = 0.09


def test_golden_section_finds_floor():
    result = golden_section_search(FREQ_MHZ, WIRE_RADIUS_M, LENGTH_MIN, LENGTH_MAX,
                                    target_vswr=1.5)
    assert result["converged"]
    assert result["best_vswr"] < 1.5
    assert result["best_vswr"] > 1.35
    assert result["n_calls"] < 20


def test_golden_section_does_not_converge_at_unreachable_target():
    """Encodes the FINDINGS.md conclusion that 1.05 is unreachable
    for this geometry - if this ever converges, the floor has changed
    and FINDINGS.md needs revisiting."""
    result = golden_section_search(FREQ_MHZ, WIRE_RADIUS_M, LENGTH_MIN, LENGTH_MAX,
                                    target_vswr=1.05, max_calls=50)
    assert not result["converged"]


def test_reactance_bisection_finds_floor():
    result = reactance_bisection(FREQ_MHZ, WIRE_RADIUS_M, LENGTH_MIN, LENGTH_MAX,
                                  target_vswr=1.5)
    assert result["converged"]
    assert result["best_vswr"] < 1.5
    assert result["best_vswr"] > 1.35


def test_reactance_bisection_raises_without_sign_change():
    """Both bounds on the same side of resonance - no root to find."""
    with pytest.raises(ValueError):
        reactance_bisection(FREQ_MHZ, WIRE_RADIUS_M, 0.03, 0.04, target_vswr=1.5)


def test_scipy_bounded_finds_floor():
    result = scipy_bounded_search(FREQ_MHZ, WIRE_RADIUS_M, LENGTH_MIN, LENGTH_MAX,
                                   target_vswr=1.5)
    assert result["converged"]
    assert result["best_vswr"] < 1.5
    assert result["best_vswr"] > 1.35


def test_all_three_methods_agree_on_length():
    """All three should find essentially the same resonant length,
    since they're solving the same underlying physics."""
    gs = golden_section_search(FREQ_MHZ, WIRE_RADIUS_M, LENGTH_MIN, LENGTH_MAX, target_vswr=1.5)
    rb = reactance_bisection(FREQ_MHZ, WIRE_RADIUS_M, LENGTH_MIN, LENGTH_MAX, target_vswr=1.5)
    sp = scipy_bounded_search(FREQ_MHZ, WIRE_RADIUS_M, LENGTH_MIN, LENGTH_MAX, target_vswr=1.5)

    lengths = [gs["best_length"], rb["best_length"], sp["best_length"]]
    assert max(lengths) - min(lengths) < 0.005  # within 5mm of each other