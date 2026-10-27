"""
Unit tests for src/model.py.

Tests verify:
  (i)   Finiteness classification from tilde_rho sign.
  (ii)  Exact benchmark values for the paper baseline.
  (iii) Derived constants: a, tilde_rho, bar_z.
  (iv)  Parameter validation (invalid inputs raise ValueError).
  (v)   g(z) upper bound properties.
  (vi)  Operator evaluation (drift, diffusion, discount, payoff).
"""

import pytest
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import ModelParams, BASELINE_PARAMS_FRICTIONLESS


# ─── Finiteness condition ──────────────────────────────────────

def test_finiteness_paper_baseline():
    """Paper baseline must give tilde_rho = 0.0528125 > 0 → finite."""
    params = ModelParams(mu=0.07, sigma=0.25, rho=0.08, p=0.5,
                         lambda_=0.5, xi=0.0, eta=0.0)
    assert abs(params.tilde_rho - 0.0528125) < 1e-9, \
        f"Expected 0.0528125, got {params.tilde_rho}"
    assert params.is_finite()


def test_finiteness_low_rho_infinite():
    """Very low rho can lead to tilde_rho <= 0 (infinite regime)."""
    params = ModelParams(mu=0.15, sigma=0.20, rho=0.01, p=0.9,
                         lambda_=0.0, xi=0.0, eta=0.0)
    # tilde_rho = 0.01 - 0.15*0.9 + 0.5*0.04*0.9*0.1 = 0.01 - 0.135 + 0.0018 = -0.1232
    assert params.tilde_rho < 0
    assert not params.is_finite()


def test_finiteness_positive():
    """Standard parameters with rho > mu*p give finite tilde_rho."""
    params = ModelParams(mu=0.05, sigma=0.20, rho=0.10, p=0.5,
                         lambda_=1.0, xi=0.5, eta=0.1)
    assert params.tilde_rho > 0
    assert params.is_finite()


def test_finiteness_threshold_exact():
    """Test exactly at the boundary: rho = mu*p - 0.5*sigma^2*p*(1-p) → tilde_rho=0."""
    mu, sigma, p = 0.10, 0.20, 0.5
    rho_boundary = mu * p - 0.5 * sigma**2 * p * (1 - p)
    params = ModelParams(mu=mu, sigma=sigma, rho=rho_boundary + 1e-9, p=p,
                         lambda_=0.5, xi=0.0, eta=0.0)
    # Slightly above boundary → finite
    assert params.tilde_rho > 0


# ─── Paper benchmark constants ────────────────────────────────

def test_bar_z_paper_baseline():
    """bar_z = tilde_rho/(1-p) = 0.0528125/0.5 = 0.105625 for paper baseline."""
    params = BASELINE_PARAMS_FRICTIONLESS
    assert abs(params.bar_z - 0.105625) < 1e-9, \
        f"Expected 0.105625, got {params.bar_z}"


def test_bar_z_raises_when_infinite():
    """bar_z should raise ValueError when tilde_rho <= 0."""
    params = ModelParams(mu=0.15, sigma=0.20, rho=0.01, p=0.9,
                         lambda_=0.0, xi=0.0, eta=0.0)
    with pytest.raises(ValueError):
        _ = params.bar_z


def test_w_frictionless_paper_baseline():
    """w_fric = (1/p)*bar_z^(p-1). For baseline ≈ 6.15."""
    params = BASELINE_PARAMS_FRICTIONLESS
    bz = params.bar_z   # 0.105625
    p = params.p        # 0.5
    expected = (1.0 / p) * bz ** (p - 1.0)
    assert abs(params.w_frictionless() - expected) < 1e-10
    # Sanity: should be around 6 for these parameters
    assert 4.0 < params.w_frictionless() < 10.0


# ─── Derived constants ────────────────────────────────────────

def test_a_property():
    """a = 1 - eta."""
    params = ModelParams(mu=0.07, sigma=0.25, rho=0.08, p=0.5,
                         lambda_=0.5, xi=0.5, eta=0.1)
    assert abs(params.a - 0.9) < 1e-12


def test_tilde_rho_formula():
    """tilde_rho = rho - mu*p + 0.5*sigma^2*p*(1-p)."""
    mu, sigma, rho, p = 0.08, 0.30, 0.12, 0.6
    params = ModelParams(mu=mu, sigma=sigma, rho=rho, p=p,
                         lambda_=0.0, xi=0.0, eta=0.0)
    expected = rho - mu * p + 0.5 * sigma**2 * p * (1 - p)
    assert abs(params.tilde_rho - expected) < 1e-12


# ─── Parameter validation ─────────────────────────────────────

def test_invalid_mu_zero():
    with pytest.raises(ValueError):
        ModelParams(mu=0.0, sigma=0.25, rho=0.08, p=0.5, lambda_=0, xi=0, eta=0)


def test_invalid_sigma_negative():
    with pytest.raises(ValueError):
        ModelParams(mu=0.07, sigma=-0.1, rho=0.08, p=0.5, lambda_=0, xi=0, eta=0)


def test_invalid_p_zero():
    with pytest.raises(ValueError):
        ModelParams(mu=0.07, sigma=0.25, rho=0.08, p=0.0, lambda_=0, xi=0, eta=0)


def test_invalid_p_one():
    with pytest.raises(ValueError):
        ModelParams(mu=0.07, sigma=0.25, rho=0.08, p=1.0, lambda_=0, xi=0, eta=0)


def test_invalid_eta_one():
    with pytest.raises(ValueError):
        ModelParams(mu=0.07, sigma=0.25, rho=0.08, p=0.5, lambda_=0, xi=0, eta=1.0)


def test_invalid_lambda_negative():
    with pytest.raises(ValueError):
        ModelParams(mu=0.07, sigma=0.25, rho=0.08, p=0.5, lambda_=-1.0, xi=0, eta=0)


def test_invalid_xi_negative():
    with pytest.raises(ValueError):
        ModelParams(mu=0.07, sigma=0.25, rho=0.08, p=0.5, lambda_=0, xi=-0.1, eta=0)


# ─── g(z) upper bound ─────────────────────────────────────────

def test_g_positive_for_positive_z():
    """g(z) should be positive for z > 0."""
    params = BASELINE_PARAMS_FRICTIONLESS
    z = np.linspace(0.01, 2.0, 50)
    g_vals = params.g(z)
    assert np.all(g_vals > 0)


def test_g_decreasing():
    """g(z) = z^p/(p*tilde_rho + p^2*z) is not necessarily monotone, but
    g(z) / z^p is decreasing (denominator grows linearly in z)."""
    params = BASELINE_PARAMS_FRICTIONLESS
    z = np.linspace(0.01, 2.0, 50)
    g_vals = params.g(z)
    ratio = g_vals / (z**params.p)
    diffs = np.diff(ratio)
    assert np.all(diffs < 0), "g(z)/z^p should be decreasing"


def test_g_at_bar_z():
    """g(bar_z) = bar_z^p / (p*tilde_rho + p^2*bar_z). Check formula."""
    params = BASELINE_PARAMS_FRICTIONLESS
    bz = params.bar_z
    expected = bz**params.p / (params.p * params.tilde_rho + params.p**2 * bz)
    assert abs(params.g_bar_z() - expected) < 1e-12


# ─── Operator evaluations ─────────────────────────────────────

def test_drift_z_at_zero():
    """drift_z(0) = 0 since all terms multiply z."""
    params = BASELINE_PARAMS_FRICTIONLESS
    assert params.drift_z(0.0) == 0.0


def test_diffusion_z_at_zero():
    """diffusion_z(0) = 0 since it is proportional to z^2."""
    params = BASELINE_PARAMS_FRICTIONLESS
    assert params.diffusion_z(0.0) == 0.0


def test_discount_z_positive():
    """discount_z(z) = tilde_rho + p*z > 0 for z >= 0 when tilde_rho > 0."""
    params = BASELINE_PARAMS_FRICTIONLESS
    z = np.linspace(0.0, 2.0, 20)
    assert np.all(params.discount_z(z) > 0)


def test_running_payoff_at_zero():
    """running_payoff(0) should return 0 (0^p/p = 0 for p>0)."""
    params = BASELINE_PARAMS_FRICTIONLESS
    assert params.running_payoff(0.0) == 0.0


def test_running_payoff_positive():
    """running_payoff(z) > 0 for z > 0."""
    params = BASELINE_PARAMS_FRICTIONLESS
    z = np.linspace(0.01, 2.0, 20)
    assert np.all(params.running_payoff(z) > 0)


def test_stopping_level():
    """stopping_level() = lambda/p."""
    params = ModelParams(mu=0.07, sigma=0.25, rho=0.08, p=0.5,
                         lambda_=2.0, xi=0.0, eta=0.0)
    assert abs(params.stopping_level() - 2.0 / 0.5) < 1e-12


# ─── Edge cases ───────────────────────────────────────────────

def test_small_p():
    """Very small p should still work."""
    params = ModelParams(mu=0.07, sigma=0.25, rho=0.08, p=0.01,
                         lambda_=0.5, xi=0.0, eta=0.0)
    assert params.is_finite()
    assert params.tilde_rho > 0


def test_large_sigma():
    """High volatility increases tilde_rho through sigma^2 term."""
    params_low_sig = ModelParams(mu=0.07, sigma=0.10, rho=0.08, p=0.5,
                                 lambda_=0, xi=0, eta=0)
    params_high_sig = ModelParams(mu=0.07, sigma=0.50, rho=0.08, p=0.5,
                                  lambda_=0, xi=0, eta=0)
    assert params_high_sig.tilde_rho > params_low_sig.tilde_rho


def test_vector_input_for_g():
    """g(z) should accept numpy array input."""
    params = BASELINE_PARAMS_FRICTIONLESS
    z = np.array([0.01, 0.05, 0.1, 0.5, 1.0])
    g = params.g(z)
    assert g.shape == z.shape
    assert np.all(np.isfinite(g))
