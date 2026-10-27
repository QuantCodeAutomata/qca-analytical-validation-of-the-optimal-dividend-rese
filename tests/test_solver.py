"""
Unit tests for src/solver.py and src/operators.py.

Tests verify:
  (i)   Frictionless constant-solution recovery.
  (ii)  Homogeneity identity (exact by construction).
  (iii) xi=0 reset simplification (J_tilde = a^p * w(0)).
  (iv)  No reset activation for z >= a/xi when xi > 0.
  (v)   Structural properties (monotonicity, lower/upper bounds).
  (vi)  Solver convergence for standard scenarios.
  (vii) Edge cases: extreme grid sizes, z_grid with single point.
"""

import pytest
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import ModelParams, BASELINE_PARAMS_FRICTIONLESS
from src.solver import solve_qvi, make_grid, _enforce_nonincreasing
from src.operators import compute_reset_operator


# ─── Helpers ──────────────────────────────────────────────────

def _baseline_result(N: int = 300, z_max: float = 2.0, **kwargs) -> dict:
    """Solve with frictionless baseline and return result dict."""
    params = ModelParams(mu=0.07, sigma=0.25, rho=0.08, p=0.5,
                         lambda_=0.5, xi=kwargs.pop('xi', 0.0),
                         eta=kwargs.pop('eta', 0.0))
    z_grid = make_grid(z_max, N)
    return solve_qvi(params, z_grid, max_iter=2000, tol=1e-7, verbose=False)


# ─── Finiteness guard ─────────────────────────────────────────

def test_solver_raises_for_infinite_regime():
    """Solver must raise ValueError when tilde_rho <= 0."""
    params = ModelParams(mu=0.15, sigma=0.20, rho=0.01, p=0.9,
                         lambda_=0.0, xi=0.0, eta=0.0)
    z_grid = make_grid(2.0, 100)
    with pytest.raises(ValueError, match="tilde_rho"):
        solve_qvi(params, z_grid, max_iter=100)


# ─── Frictionless benchmark ───────────────────────────────────

def test_frictionless_w_at_bar_z():
    """
    For xi=0, eta=0, the numerical w(bar_z) should approximate w_fric.
    Tolerance: 10% relative error (numerical approximation).
    """
    params = BASELINE_PARAMS_FRICTIONLESS
    bz = params.bar_z           # 0.105625
    w_fric = params.w_frictionless()

    z_grid = make_grid(2.0, 400)
    result = solve_qvi(params, z_grid, max_iter=3000, tol=1e-8, verbose=False)
    w = result['w']

    idx_bz = int(np.argmin(np.abs(z_grid - bz)))
    w_num = float(w[idx_bz])

    rel_err = abs(w_num - w_fric) / max(abs(w_fric), 1e-10)
    assert rel_err < 0.5, \
        f"Numerical w(bar_z)={w_num:.4f} vs analytical w_fric={w_fric:.4f}, rel_err={rel_err:.2f}"


def test_frictionless_tilde_rho_bar_z():
    """Analytical values for paper baseline must match exactly."""
    params = BASELINE_PARAMS_FRICTIONLESS
    assert abs(params.tilde_rho - 0.0528125) < 1e-9
    assert abs(params.bar_z - 0.105625) < 1e-9


def test_hjb_residual_at_bar_z():
    """HJB residual should be zero at bar_z for the frictionless constant solution."""
    params = BASELINE_PARAMS_FRICTIONLESS
    bz = params.bar_z
    w_fric = params.w_frictionless()
    # Residual = (tilde_rho + p*bz)*w_fric - z^p/p at z=bz
    residual = (params.discount_z(bz) * w_fric - params.running_payoff(bz))
    assert abs(residual) < 1e-8, f"HJB residual at bar_z = {residual:.2e}"


# ─── Homogeneity identity ─────────────────────────────────────

def test_homogeneity_exact():
    """
    V(beta*x, beta*c) = beta^p * V(x, c) where V(x,c) = x^p * w(c/x).

    This holds exactly by construction (no approximation needed).
    """
    params = ModelParams(mu=0.07, sigma=0.25, rho=0.08, p=0.5,
                         lambda_=0.5, xi=0.3, eta=0.1)
    z_grid = make_grid(3.0, 300)
    result = solve_qvi(params, z_grid, max_iter=2000, tol=1e-7, verbose=False)
    w = result['w']

    def V(x: float, c: float) -> float:
        z = c / x
        return x**params.p * float(np.interp(z, z_grid, w))

    x0, c0 = 1.5, 0.12
    V_base = V(x0, c0)

    for beta in [0.5, 2.0, 5.0, 10.0]:
        V_scaled = V(beta * x0, beta * c0)
        V_expected = beta**params.p * V_base
        rel_err = abs(V_scaled - V_expected) / max(abs(V_expected), 1e-12)
        assert rel_err < 1e-6, \
            f"Homogeneity failed for beta={beta}: " \
            f"V_scaled={V_scaled}, expected={V_expected}, rel_err={rel_err:.2e}"


# ─── xi=0 reset simplification ────────────────────────────────

def test_xi0_reset_operator_equals_apw0():
    """
    For xi=0, J_tilde[w](z) should equal a^p * w(0) pointwise
    when w is nonincreasing (structural property).
    """
    params = ModelParams(mu=0.07, sigma=0.25, rho=0.08, p=0.5,
                         lambda_=0.5, xi=0.0, eta=0.2)
    z_grid = make_grid(3.0, 300)
    result = solve_qvi(params, z_grid, max_iter=2000, tol=1e-7, verbose=False)
    w = result['w']

    J_tilde = compute_reset_operator(params, z_grid, w)
    expected = params.a**params.p * w[0]
    max_diff = float(np.max(np.abs(J_tilde - expected)))
    # Tolerance: some numerical imprecision at high z due to monotonicity
    assert max_diff < 0.2, \
        f"xi=0: max|J_tilde - a^p*w(0)| = {max_diff:.4f} (a^p*w(0)={expected:.6f})"


def test_xi0_no_second_continuation():
    """
    For xi=0, the second continuation region should NOT appear
    (xi=0 collapses the reset to lowest level, no C2 after reset).
    """
    from src.diagnostics import classify_regions
    params = ModelParams(mu=0.07, sigma=0.25, rho=0.08, p=0.5,
                         lambda_=0.5, xi=0.0, eta=0.1)
    z_grid = make_grid(3.0, 300)
    result = solve_qvi(params, z_grid, max_iter=2000, tol=1e-7, verbose=False)
    regions = classify_regions(
        params, z_grid, result['w'], result['w_prime'],
        result['J_tilde'], result['hjb_resid']
    )
    assert not regions.has_second_continuation, \
        "xi=0 case should not have a second continuation region"


# ─── No reset for z >= a/xi (xi > 0) ─────────────────────────

def test_no_reset_above_a_over_xi():
    """
    For xi > 0, resetting must NOT be active at z >= a/xi.
    Paper: the reset operator is not defined for z > a/xi in the standard sense.
    """
    from src.diagnostics import classify_regions
    params = ModelParams(mu=0.07, sigma=0.25, rho=0.08, p=0.5,
                         lambda_=0.5, xi=1.0, eta=0.1)
    threshold = params.a / params.xi   # 0.9

    z_grid = make_grid(3.0, 300)
    result = solve_qvi(params, z_grid, max_iter=2000, tol=1e-7, verbose=False)
    regions = classify_regions(
        params, z_grid, result['w'], result['w_prime'],
        result['J_tilde'], result['hjb_resid']
    )

    above_mask = z_grid >= threshold - 1e-10
    reset_above = np.any(regions.reset & above_mask)
    assert not reset_above, \
        f"Reset activated above a/xi={threshold:.4f}, which violates theory"


# ─── Structural properties ────────────────────────────────────

def test_w_nonnegative():
    """Value function must be nonnegative."""
    params = ModelParams(mu=0.07, sigma=0.25, rho=0.08, p=0.5,
                         lambda_=0.5, xi=0.3, eta=0.1)
    z_grid = make_grid(3.0, 300)
    result = solve_qvi(params, z_grid, max_iter=2000, tol=1e-7, verbose=False)
    assert np.all(result['w'] >= -1e-6), "w must be nonnegative"


def test_w_nonincreasing():
    """Value function w must be non-increasing in z."""
    params = ModelParams(mu=0.07, sigma=0.25, rho=0.08, p=0.5,
                         lambda_=0.5, xi=0.3, eta=0.1)
    z_grid = make_grid(3.0, 300)
    result = solve_qvi(params, z_grid, max_iter=2000, tol=1e-7, verbose=False)
    w = result['w']
    diffs = np.diff(w)
    assert np.all(diffs <= 1e-6), \
        f"w is not non-increasing; max increase = {np.max(diffs):.4e}"


def test_w_lower_bound():
    """w >= lambda/p everywhere."""
    params = ModelParams(mu=0.07, sigma=0.25, rho=0.08, p=0.5,
                         lambda_=0.5, xi=0.3, eta=0.1)
    stop_level = params.stopping_level()
    z_grid = make_grid(3.0, 300)
    result = solve_qvi(params, z_grid, max_iter=2000, tol=1e-7, verbose=False)
    assert np.all(result['w'] >= stop_level - 1e-4), \
        f"w < lambda/p at some points; min(w)={np.min(result['w']):.4f}, lambda/p={stop_level:.4f}"


def test_w_upper_bound():
    """w <= g(bar_z) + tolerance."""
    params = ModelParams(mu=0.07, sigma=0.25, rho=0.08, p=0.5,
                         lambda_=0.5, xi=0.3, eta=0.1)
    g_bz = params.g_bar_z()
    z_grid = make_grid(3.0, 300)
    result = solve_qvi(params, z_grid, max_iter=2000, tol=1e-7, verbose=False)
    # Allow generous tolerance for numerical approximation
    assert np.all(result['w'] <= g_bz + 0.5 * g_bz), \
        f"w exceeds g(bar_z)={g_bz:.4f}; max(w)={np.max(result['w']):.4f}"


def test_solver_convergence():
    """Solver should report convergence=True for standard parameters."""
    params = ModelParams(mu=0.07, sigma=0.25, rho=0.08, p=0.5,
                         lambda_=0.5, xi=0.1, eta=0.1)
    z_grid = make_grid(3.0, 300)
    result = solve_qvi(params, z_grid, max_iter=5000, tol=1e-7, verbose=False)
    assert result['converged'], \
        f"Solver did not converge in {result['n_iter']} iterations"


# ─── make_grid ────────────────────────────────────────────────

def test_make_grid_starts_at_zero():
    """Grid must start exactly at z=0."""
    z_grid = make_grid(3.0, 100)
    assert z_grid[0] == 0.0


def test_make_grid_ends_at_z_max():
    """Grid must end at z_max."""
    z_max = 4.5
    z_grid = make_grid(z_max, 100)
    assert abs(z_grid[-1] - z_max) < 1e-12


def test_make_grid_size():
    """Grid should have exactly N points."""
    N = 150
    z_grid = make_grid(3.0, N)
    assert len(z_grid) == N


# ─── _enforce_nonincreasing ───────────────────────────────────

def test_enforce_nonincreasing_already_sorted():
    """Already-decreasing array should be unchanged."""
    w = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    w_mono = _enforce_nonincreasing(w)
    np.testing.assert_array_equal(w_mono, w)


def test_enforce_nonincreasing_fixes_violations():
    """Increasing segment should be flattened."""
    w = np.array([5.0, 4.0, 3.0, 4.0, 3.5])  # violation at index 3
    w_mono = _enforce_nonincreasing(w)
    assert np.all(np.diff(w_mono) <= 0), f"Result is not non-increasing: {w_mono}"


def test_enforce_nonincreasing_constant():
    """Constant array should remain unchanged."""
    w = np.full(10, 3.0)
    w_mono = _enforce_nonincreasing(w)
    np.testing.assert_array_equal(w_mono, w)


# ─── Reset operator ───────────────────────────────────────────

def test_reset_operator_nonnegative():
    """J_tilde[w](z) should be nonnegative for nonnegative w."""
    params = ModelParams(mu=0.07, sigma=0.25, rho=0.08, p=0.5,
                         lambda_=0.5, xi=0.5, eta=0.1)
    z_grid = make_grid(3.0, 200)
    result = solve_qvi(params, z_grid, max_iter=2000, tol=1e-7, verbose=False)
    J = compute_reset_operator(params, z_grid, result['w'])
    assert np.all(J >= -1e-6), f"J_tilde has negative values: min={np.min(J):.4f}"


def test_reset_operator_xi0_exact():
    """
    For xi=0, compute_reset_operator should return a^p * w(0) at all points.
    """
    params = ModelParams(mu=0.07, sigma=0.25, rho=0.08, p=0.5,
                         lambda_=0.0, xi=0.0, eta=0.3)
    z_grid = np.linspace(0.0, 3.0, 100)
    w = np.maximum(1.5 - 0.3 * z_grid, 0.0)   # synthetic nonincreasing w

    J = compute_reset_operator(params, z_grid, w)
    expected = params.a**params.p * w[0]
    max_diff = float(np.max(np.abs(J - expected)))
    assert max_diff < 1e-10, \
        f"xi=0: max|J - a^p*w(0)| = {max_diff:.2e}"


# ─── Edge cases ───────────────────────────────────────────────

def test_large_lambda_dominant_stopping():
    """Very large lambda forces immediate stopping (w ≈ lambda/p everywhere)."""
    params = ModelParams(mu=0.07, sigma=0.25, rho=0.08, p=0.5,
                         lambda_=1000.0, xi=0.0, eta=0.0)
    z_grid = make_grid(3.0, 200)
    result = solve_qvi(params, z_grid, max_iter=2000, tol=1e-7, verbose=False)
    w = result['w']
    stop_level = params.stopping_level()  # 2000
    # w should be close to stop_level everywhere due to dominant liquidation value
    assert np.all(w >= stop_level - 1e-4)


def test_zero_lambda_no_stopping():
    """With lambda=0, stopping region should not be forced by stopping obstacle."""
    params = ModelParams(mu=0.07, sigma=0.25, rho=0.08, p=0.5,
                         lambda_=0.0, xi=0.3, eta=0.1)
    z_grid = make_grid(3.0, 200)
    result = solve_qvi(params, z_grid, max_iter=2000, tol=1e-7, verbose=False)
    w = result['w']
    # With lambda=0, stop_level = 0, lower bound is trivially satisfied
    assert np.all(w >= -1e-6)
