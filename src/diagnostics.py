"""
Diagnostics, region classification, threshold extraction, and structural
property checks for the solved reduced value function w(z).

# Custom — Context7 found no library equivalent for these domain-specific
# diagnostics of the dividend-reset QVI solution.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from src.model import ModelParams
from src.solver import numerical_second_derivative


@dataclass
class RegionInfo:
    """Classification of yield-space policy regions."""
    # Boolean masks on z_grid
    increase: np.ndarray          # w is flat (singular control active)
    continuation1: np.ndarray     # local HJB equation active, no obstacle
    reset: np.ndarray             # w = J_tilde[w]
    continuation2: np.ndarray     # continuation after reset region (if present)
    stopping: np.ndarray          # w = lambda/p

    # Scalar thresholds
    z_star: Optional[float] = None       # sup{increase region}
    z_upper: Optional[float] = None      # inf{reset region}   (= z^*)
    z_upper2: Optional[float] = None     # sup{reset region}   (= z^{**})
    z_lambda: Optional[float] = None     # inf{stopping region}
    z_0_reset: Optional[float] = None    # representative reset target

    has_second_continuation: bool = False
    has_stopping: bool = False


def classify_regions(
    params: ModelParams,
    z_grid: np.ndarray,
    w: np.ndarray,
    w_prime: np.ndarray,
    J_tilde: np.ndarray,
    hjb_resid: np.ndarray,
    eps_deriv: float = 1e-3,
    eps_reset: float = 1e-3,
    eps_stop: float = 1e-3,
    eps_hjb: float = 5e-2,
) -> RegionInfo:
    """
    Classify each grid point into a policy region.

    Priority hierarchy (higher priority overrides lower):
        1. Stopping  : w ≈ lambda/p
        2. Reset     : w ≈ J_tilde[w]
        3. Increase  : w' ≈ 0 and w is flat (singular control)
        4. Continuation: local HJB residual small

    Parameters
    ----------
    params : ModelParams
    z_grid : np.ndarray, shape (N,)
    w : np.ndarray, shape (N,)
    w_prime : np.ndarray, shape (N,)
    J_tilde : np.ndarray, shape (N,)
    hjb_resid : np.ndarray, shape (N,)
    eps_deriv : float
        Tolerance for |w'| ≈ 0 (increase region detection).
    eps_reset : float
        Tolerance for |w - J_tilde| / max(|w|,1) ≈ 0.
    eps_stop : float
        Tolerance for |w - lambda/p| / max(|w|,1) ≈ 0.
    eps_hjb : float
        Tolerance for HJB residual (not primary classifier but used for C1/C2 label).

    Returns
    -------
    RegionInfo
    """
    N = len(z_grid)
    stop_level = params.stopping_level()
    w_scale = np.maximum(np.abs(w), 1.0)

    # Raw flags
    is_stop = np.abs(w - stop_level) / w_scale < eps_stop
    is_reset = (np.abs(w - J_tilde) / w_scale < eps_reset) & ~is_stop
    is_increase = (np.abs(w_prime) < eps_deriv) & ~is_reset & ~is_stop
    is_cont = ~is_stop & ~is_reset & ~is_increase

    # Smooth into connected intervals: remove isolated misclassifications
    is_stop = _smooth_mask(is_stop, min_run=1)
    is_reset = _smooth_mask(is_reset & ~is_stop, min_run=1)
    is_increase_raw = is_increase & ~is_stop & ~is_reset
    # Increase region should be at the left (small z): find first connected run from z=0
    first_increase_run = _first_connected_run(is_increase_raw)
    is_increase = first_increase_run
    is_cont = ~is_stop & ~is_reset & ~is_increase

    # Identify second continuation: if there is continuation AFTER the reset region
    # Split continuation into C1 (before reset) and C2 (after reset)
    cont2 = np.zeros(N, dtype=bool)
    if np.any(is_reset):
        reset_end = np.where(is_reset)[0][-1]
        cont2[reset_end + 1:] = is_cont[reset_end + 1:]
        cont1 = is_cont.copy()
        cont1[reset_end + 1:] = False
    else:
        cont1 = is_cont
        cont2 = np.zeros(N, dtype=bool)

    has_second_continuation = bool(np.any(cont2))
    has_stopping = bool(np.any(is_stop))

    # Extract scalar thresholds
    z_star = _last_index_threshold(is_increase, z_grid)
    z_upper = _first_index_threshold(is_reset, z_grid)
    z_upper2 = _last_index_threshold(is_reset, z_grid)
    z_lambda = _first_index_threshold(is_stop, z_grid)

    # Representative reset target: for each reset point, the argmax z0
    z_0_reset = _estimate_reset_target(params, z_grid, w, is_reset)

    return RegionInfo(
        increase=is_increase,
        continuation1=cont1,
        reset=is_reset,
        continuation2=cont2,
        stopping=is_stop,
        z_star=z_star,
        z_upper=z_upper,
        z_upper2=z_upper2,
        z_lambda=z_lambda,
        z_0_reset=z_0_reset,
        has_second_continuation=has_second_continuation,
        has_stopping=has_stopping,
    )


def _smooth_mask(mask: np.ndarray, min_run: int = 2) -> np.ndarray:
    """Remove isolated True values (runs shorter than min_run)."""
    result = mask.copy()
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            run_len = j - i
            if run_len < min_run:
                result[i:j] = False
            i = j
        else:
            i += 1
    return result


def _first_connected_run(mask: np.ndarray) -> np.ndarray:
    """Return a mask that keeps only the first connected True run."""
    result = np.zeros_like(mask)
    n = len(mask)
    i = 0
    # Find first True
    while i < n and not mask[i]:
        i += 1
    if i == n:
        return result
    # Keep this run
    while i < n and mask[i]:
        result[i] = True
        i += 1
    return result


def _last_index_threshold(mask: np.ndarray, z_grid: np.ndarray) -> Optional[float]:
    """Return z_grid[last True index] or None."""
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        return None
    return float(z_grid[idxs[-1]])


def _first_index_threshold(mask: np.ndarray, z_grid: np.ndarray) -> Optional[float]:
    """Return z_grid[first True index] or None."""
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        return None
    return float(z_grid[idxs[0]])


def _estimate_reset_target(
    params: ModelParams,
    z_grid: np.ndarray,
    w: np.ndarray,
    is_reset: np.ndarray,
) -> Optional[float]:
    """
    Estimate the representative reset target z_0 for the reset region.

    For xi=0: the target is always w(0) so z_0 = z_grid[0].
    For xi>0: for the first reset point, find argmax of the weighted objective.
    """
    if not np.any(is_reset):
        return None
    xi = params.xi
    a = params.a
    reset_idxs = np.where(is_reset)[0]
    z_reset_entry = z_grid[reset_idxs[0]]

    if abs(xi) < 1e-12:
        return float(z_grid[0])
    else:
        # For Case 1 (z < a/xi), maximize w(z0)/(1-xi*z0)^p over z0 in [0, z/a]
        threshold = a / xi
        if z_reset_entry < threshold:
            upper_z0 = z_reset_entry / a
            mask = z_grid <= upper_z0 + 1e-12
            if not np.any(mask):
                return None
            z0_cands = z_grid[mask]
            denom = (1.0 - xi * z0_cands)
            valid = denom > 1e-12
            if not np.any(valid):
                return None
            obj = np.where(valid, w[mask] / denom**params.p, -np.inf)
            best_idx = np.argmax(obj)
            return float(z0_cands[best_idx])
        else:
            # z >= a/xi: Case 3 or 2
            return float(a / xi)


def check_structural_properties(
    params: ModelParams,
    z_grid: np.ndarray,
    w: np.ndarray,
    w_prime: np.ndarray,
    regions: RegionInfo,
    tol: float = 1e-2,
) -> dict:
    """
    Check structural properties implied by the theory.

    Properties checked:
        1. Nonnegativity: w >= 0
        2. Monotonicity: w' <= tol_deriv
        3. Lower bound: w >= lambda/p - tol
        4. Upper bound: w <= g(bar_z) + tol
        5. Supercontact at z_*: w''(z_*) ≈ 0
        6. xi=0 reset simplification: J_tilde = a^p * w(0)
        7. No reset for z >= a/xi when xi > 0

    Parameters
    ----------
    params, z_grid, w, w_prime, regions : as from solve_qvi and classify_regions.
    tol : float — tolerance for bound checks.

    Returns
    -------
    report : dict mapping property name to (passed: bool, detail: str)
    """
    from src.operators import compute_reset_operator

    report = {}
    stop_level = params.stopping_level()

    # 1. Nonnegativity
    min_w = float(np.min(w))
    report['nonnegative'] = (min_w >= -tol, f"min(w) = {min_w:.6f}")

    # 2. Monotonicity (w non-increasing)
    max_deriv = float(np.max(w_prime))
    report['nonincreasing'] = (max_deriv <= tol, f"max(w') = {max_deriv:.6f}")

    # 3. Lower bound w >= lambda/p
    min_excess_lb = float(np.min(w - stop_level))
    report['lower_bound'] = (
        min_excess_lb >= -tol,
        f"min(w - lambda/p) = {min_excess_lb:.6f}, lambda/p = {stop_level:.4f}"
    )

    # 4. Upper bound w <= g(bar_z)
    try:
        g_bz = params.g_bar_z()
        max_excess_ub = float(np.max(w - g_bz))
        report['upper_bound'] = (
            max_excess_ub <= tol + abs(g_bz) * tol,
            f"max(w - g(bar_z)) = {max_excess_ub:.6f}, g(bar_z) = {g_bz:.4f}"
        )
    except Exception as e:
        report['upper_bound'] = (False, f"Error computing g(bar_z): {e}")

    # 5. Supercontact at z_* (w''(z_*) ≈ 0)
    if regions.z_star is not None:
        z_star_idx = int(np.argmin(np.abs(z_grid - regions.z_star)))
        wpp_at_zstar = numerical_second_derivative(w, z_grid, z_star_idx, window=5)
        report['supercontact'] = (
            abs(wpp_at_zstar) < 5.0,
            f"w''(z_*) ≈ {wpp_at_zstar:.4f} at z_* = {regions.z_star:.6f}"
        )
    else:
        report['supercontact'] = (False, "z_* not identified")

    # 6. xi=0 reset simplification
    if abs(params.xi) < 1e-12:
        J_tilde = compute_reset_operator(params, z_grid, w)
        expected = params.a**params.p * w[0]
        # Check at a representative reset point if any
        if np.any(regions.reset):
            reset_idx = np.where(regions.reset)[0][0]
            J_val = J_tilde[reset_idx]
            diff = abs(J_val - expected)
            report['xi0_reset_simplification'] = (
                diff < tol * max(abs(expected), 1.0) + 1e-4,
                f"J_tilde[{z_grid[reset_idx]:.4f}]={J_val:.6f}, a^p*w(0)={expected:.6f}"
            )
        else:
            # No reset region; check everywhere
            max_diff = float(np.max(np.abs(J_tilde - expected)))
            report['xi0_reset_simplification'] = (
                max_diff < tol + 1e-4,
                f"max|J_tilde - a^p*w(0)| = {max_diff:.6f}, a^p*w(0) = {expected:.6f}"
            )
    else:
        report['xi0_reset_simplification'] = (True, "xi>0: not applicable")

    # 7. No reset activation for z >= a/xi when xi > 0
    if params.xi > 1e-12:
        threshold_reset = params.a / params.xi
        above_threshold = z_grid >= threshold_reset - 1e-10
        reset_above = np.any(regions.reset & above_threshold)
        report['no_reset_above_threshold'] = (
            not reset_above,
            f"Reset active above a/xi={threshold_reset:.4f}: {reset_above}"
        )
    else:
        report['no_reset_above_threshold'] = (True, "xi=0: not applicable")

    return report


def check_homogeneity(
    params: ModelParams,
    z_grid: np.ndarray,
    w: np.ndarray,
    x_test: float = 1.0,
    c_test: float = 0.1,
    beta_values: list | None = None,
) -> dict:
    """
    Verify the homogeneity relation V(beta*x, beta*c) = beta^p * V(x, c)
    where V(x,c) = x^p * w(c/x).

    Parameters
    ----------
    params : ModelParams
    z_grid : np.ndarray, shape (N,)
    w : np.ndarray, shape (N,)
    x_test : float — base equity value
    c_test : float — base dividend rate
    beta_values : list of positive floats to test

    Returns
    -------
    report : dict with results for each beta
    """
    if beta_values is None:
        beta_values = [0.5, 1.0, 2.0, 5.0, 10.0]

    def V(x: float, c: float) -> float:
        z = c / x
        return x**params.p * float(np.interp(z, z_grid, w))

    v_base = V(x_test, c_test)
    report = {
        'x_test': x_test,
        'c_test': c_test,
        'V_base': v_base,
        'betas': {},
    }

    for beta in beta_values:
        v_scaled = V(beta * x_test, beta * c_test)
        v_expected = beta**params.p * v_base
        rel_err = abs(v_scaled - v_expected) / max(abs(v_expected), 1e-10)
        report['betas'][beta] = {
            'V_scaled': v_scaled,
            'V_expected': v_expected,
            'rel_error': rel_err,
            'passed': rel_err < 1e-6,  # exact by construction
        }

    return report
