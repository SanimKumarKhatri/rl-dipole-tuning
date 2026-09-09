"""
Benchmark harness for comparing dipole length optimization methods.

Every method (classical or RL) is measured identically: same target,
same convergence threshold, same call-counting, same result format.
This lets PR5 add new methods later (bisection, scipy, PPO, DQN+Optuna)
without changing how any existing method is measured.
"""
from nec_core import simulate_dipole
from scipy.optimize import minimize_scalar


def golden_section_search(freq_mhz, wire_radius_m, length_min, length_max,
                           target_vswr=1.05, max_calls=100, tol=1e-5):
    """
    Find the dipole length minimizing VSWR using golden-section search.

    Assumes VSWR(length) is unimodal over [length_min, length_max] - i.e.,
    exactly one minimum, no other local dips. This holds for a dipole
    swept through a single half-wave resonance, which is our case.

    Returns a dict: n_calls, converged, best_length, best_vswr, history.
    history is a list of (length, vswr) pairs in the order they were
    evaluated - useful later for plotting convergence, same style as
    train_dipole.py's trajectory plot.
    """
    phi_inv = (5 ** 0.5 - 1) / 2  # 1/golden ratio, approx 0.618

    def vswr_at(length_m):
        result = simulate_dipole(length_m, freq_mhz, wire_radius_m)
        return result["vswr"]

    a, b = length_min, length_max
    history = []
    n_calls = 0

    # initial two interior points
    c = b - phi_inv * (b - a)
    d = a + phi_inv * (b - a)
    fc = vswr_at(c); n_calls += 1
    fd = vswr_at(d); n_calls += 1
    history.append((c, fc))
    history.append((d, fd))

    best_length = c if fc < fd else d
    best_vswr = min(fc, fd)

    while (b - a) > tol and n_calls < max_calls:
        if fc < fd:
            # minimum is in [a, d] - discard the right side
            b = d
            d, fd = c, fc
            c = b - phi_inv * (b - a)
            fc = vswr_at(c); n_calls += 1
            history.append((c, fc))
        else:
            # minimum is in [c, b] - discard the left side
            a = c
            c, fc = d, fd
            d = a + phi_inv * (b - a)
            fd = vswr_at(d); n_calls += 1
            history.append((d, fd))

        if fc < best_vswr:
            best_length, best_vswr = c, fc
        if fd < best_vswr:
            best_length, best_vswr = d, fd

        if best_vswr < target_vswr:
            break

    return {
        "method": "golden_section",
        "n_calls": n_calls,
        "converged": best_vswr < target_vswr,
        "best_length": best_length,
        "best_vswr": best_vswr,
        "history": history,
    }

def reactance_bisection(freq_mhz, wire_radius_m, length_min, length_max,
                         target_vswr=1.5, max_calls=100, tol=1e-6):
    """
    Find the resonant dipole length (reactance X = 0) using bisection.

    Unlike golden_section_search (which minimizes VSWR directly), this
    exploits the fact that reactance X(length) is monotonic through
    resonance: negative (too short) on one side, positive (too long)
    on the other. Bisection finds the zero-crossing directly.

    Note: this finds resonance (X=0), which is necessary but not
    sufficient for low VSWR - see FINDINGS.md for why R at resonance
    (~72-75 ohm) sets a VSWR floor independent of length. target_vswr
    here is used only to report convergence against the harness's
    common interface; it does not change what bisection is solving for.

    Returns a dict: n_calls, converged, best_length, best_vswr, history.
    Raises ValueError if X does not change sign across [length_min, length_max] -
    bisection's correctness depends on a sign change existing in range.
    """
    def reactance_and_vswr_at(length_m):
        result = simulate_dipole(length_m, freq_mhz, wire_radius_m)
        return result["impedance"].imag, result["vswr"]

    a, b = length_min, length_max
    history = []
    n_calls = 0

    xa, vswr_a = reactance_and_vswr_at(a); n_calls += 1
    xb, vswr_b = reactance_and_vswr_at(b); n_calls += 1
    history.append((a, vswr_a))
    history.append((b, vswr_b))

    if xa == 0:
        return {"method": "reactance_bisection", "n_calls": n_calls,
                "converged": vswr_a < target_vswr, "best_length": a,
                "best_vswr": vswr_a, "history": history}
    if xb == 0:
        return {"method": "reactance_bisection", "n_calls": n_calls,
                "converged": vswr_b < target_vswr, "best_length": b,
                "best_vswr": vswr_b, "history": history}

    if (xa > 0) == (xb > 0):
        raise ValueError(
            f"Reactance does not change sign across [{length_min}, {length_max}] "
            f"at {freq_mhz} MHz (X({length_min})={xa:.2f}, X({length_max})={xb:.2f}). "
            f"Bisection requires a sign change - no guaranteed resonance in this range."
        )

    best_length = a if abs(xa) < abs(xb) else b
    best_vswr = vswr_a if abs(xa) < abs(xb) else vswr_b

    while (b - a) > tol and n_calls < max_calls:
        mid = (a + b) / 2.0
        xm, vswr_m = reactance_and_vswr_at(mid); n_calls += 1
        history.append((mid, vswr_m))

        if vswr_m < best_vswr:
            best_length, best_vswr = mid, vswr_m

        if xm == 0:
            best_length, best_vswr = mid, vswr_m
            break

        if (xm > 0) == (xa > 0):
            a, xa = mid, xm
        else:
            b, xb = mid, xm

    return {
        "method": "reactance_bisection",
        "n_calls": n_calls,
        "converged": best_vswr < target_vswr,
        "best_length": best_length,
        "best_vswr": best_vswr,
        "history": history,
    }

def scipy_bounded_search(freq_mhz, wire_radius_m, length_min, length_max,
                          target_vswr=1.5, max_calls=100, tol=1e-6):
    """
    Find the dipole length minimizing VSWR using scipy's bounded Brent
    method (scipy.optimize.minimize_scalar, method='bounded').

    Included as a third classical baseline using a standard, independently
    maintained library implementation, rather than relying only on the
    two methods implemented above (golden_section_search,
    reactance_bisection) - lets the benchmark's "classical methods
    converge quickly" claim rest on more than our own code.

    Returns a dict: n_calls, converged, best_length, best_vswr, history.
    """
    history = []
    call_count = [0]  # mutable, so the closure below can increment it

    def vswr_at(length_m):
        result = simulate_dipole(length_m, freq_mhz, wire_radius_m)
        vswr = result["vswr"]
        call_count[0] += 1
        history.append((length_m, vswr))
        return vswr

    result = minimize_scalar(
        vswr_at,
        bounds=(length_min, length_max),
        method="bounded",
        options={"xatol": tol, "maxiter": max_calls},
    )

    best_length = result.x
    best_vswr = result.fun

    return {
        "method": "scipy_bounded",
        "n_calls": call_count[0],
        "converged": best_vswr < target_vswr,
        "best_length": best_length,
        "best_vswr": best_vswr,
        "history": history,
    }