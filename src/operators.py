"""
Finite-difference operators for the reduced HJB-QVI.

Implements:
  - build_fd_matrix : tridiagonal finite-difference matrix for L^z
  - compute_reset_operator : nonlocal J_tilde[w](z) for both xi>0 and xi=0 cases
  - apply_local_operator : evaluate L^z w pointwise from stored w array

# Custom — Context7 found no library equivalent for this domain-specific nonlocal
# impulse-control reset operator.  scipy.sparse is used for the tridiagonal system.
"""

from __future__ import annotations
import numpy as np
from scipy import sparse, interpolate
from src.model import ModelParams


def build_fd_matrix(
    params: ModelParams,
    z_grid: np.ndarray,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """
    Build the tridiagonal finite-difference matrix A and right-hand-side
    vector rhs for the reduced continuation ODE:

        L^z w(z) - (tilde_rho + p*z)*w(z) + z^p/p = 0

    Discretised with upwind first-order differences for the drift term and
    centred second-order differences for the diffusion term.

    At z=0 we use a one-sided (forward) scheme; the left Dirichlet boundary
    condition is w(0) = constant (handled in the solver via pinning or obstacle).

    Parameters
    ----------
    params : ModelParams
        Model parameters.
    z_grid : np.ndarray
        1D grid of z values, shape (N,).

    Returns
    -------
    A : scipy.sparse.csr_matrix, shape (N, N)
        The finite-difference operator matrix (A w = rhs in continuation region).
    rhs : np.ndarray, shape (N,)
        Right-hand side: rhs[i] = -z_i^p / p.
    """
    N = len(z_grid)
    h = np.diff(z_grid)  # N-1 spacings (uniform or non-uniform)

    # Coefficients at each interior point
    sigma2 = params.sigma**2
    data_diag = np.zeros(N)
    data_upper = np.zeros(N - 1)
    data_lower = np.zeros(N - 1)

    for i in range(N):
        z = z_grid[i]
        diff_coef = 0.5 * sigma2 * z**2      # diffusion coefficient
        drft_coef = params.drift_z(z)        # drift coefficient

        if i == 0:
            # Forward one-sided: z=0 boundary
            if N > 1:
                h_f = h[0]
                # w'(0) ~ (w[1]-w[0])/h_f,  w''(0) ~ handled implicitly
                # At z=0 diffusion vanishes; pure drift upwind
                # drift at z=0 is 0 regardless; use identity (boundary pinned)
                data_diag[0] = 1.0  # will be overridden by boundary condition handling
        elif i == N - 1:
            # Backward one-sided at right boundary
            h_b = h[-1]
            # w'(z_N) ~ (w[N-1]-w[N-2])/h_b,  w''(z_N) ~ 0 (Neumann)
            # Upwinding for drift
            if drft_coef >= 0:
                data_diag[i] -= drft_coef / h_b
                data_lower[i - 1] += drft_coef / h_b
            else:
                # At right boundary with negative drift, use right-biased; treat as 0 flux
                data_diag[i] -= abs(drft_coef) / h_b
                data_lower[i - 1] += abs(drft_coef) / h_b
            # Diffusion: use backward diff w'' ~ 2*(w[i]-w[i-1])/h_b^2 (approx)
            data_diag[i] += -2.0 * diff_coef / h_b**2
            data_lower[i - 1] += 2.0 * diff_coef / h_b**2
            # Discount
            data_diag[i] -= params.discount_z(z)
        else:
            # Interior: centred diffusion, upwind drift
            h_b = h[i - 1]
            h_f = h[i]
            h_avg = 0.5 * (h_b + h_f)

            # Centred second-order for diffusion (uniform-grid specialisation)
            data_diag[i] += -diff_coef * 2.0 / (h_b * h_f)
            data_upper[i] += diff_coef * 2.0 / (h_f * (h_b + h_f))
            data_lower[i - 1] += diff_coef * 2.0 / (h_b * (h_b + h_f))

            # Upwind for drift
            if drft_coef >= 0:
                # Upwind: use backward difference
                data_diag[i] -= drft_coef / h_b
                data_lower[i - 1] += drft_coef / h_b
            else:
                # Upwind: use forward difference
                data_diag[i] += drft_coef / h_f   # drft_coef < 0, so subtracting
                data_upper[i] -= drft_coef / h_f   # positive contribution

            # Discount coefficient
            data_diag[i] -= params.discount_z(z)

    # Assemble sparse matrix
    diagonals = [data_lower, data_diag, data_upper]
    offsets = [-1, 0, 1]
    A = sparse.diags(diagonals, offsets, shape=(N, N), format="csr")

    # Right-hand side: -z^p/p
    rhs = -params.running_payoff(z_grid)

    return A, rhs


def compute_reset_operator(
    params: ModelParams,
    z_grid: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """
    Compute the nonlocal reduced reset operator J_tilde[w](z) at every grid
    point using linear interpolation of w over the admissible reset-target grid.

    Three-branch formula for xi > 0 (paper Section 4):
      Case 1: 0 <= z < a/xi  →  J_tilde = (a - xi*z)^p * max_{0<=z0<=z/a} w(z0)/(1-xi*z0)^p
      Case 2: z = a/xi       →  J_tilde = a^p * w(1/xi)
      Case 3: z > a/xi       →  J_tilde = (xi*z - a)^p * max_{z0>=z/a} w(z0)/(xi*z0-1)^p

    For xi = 0 (paper Section 4, special case):
      J_tilde[w](z) = a^p * max_{0 <= z0 <= z/a} w(z0)
    which simplifies to a^p * w(0) when w is nonincreasing (structural property).

    Parameters
    ----------
    params : ModelParams
        Model parameters.
    z_grid : np.ndarray
        1D grid of z values, shape (N,).
    w : np.ndarray
        Current value function on z_grid, shape (N,).

    Returns
    -------
    J_tilde : np.ndarray
        Reset operator values, shape (N,).

    Implementation notes
    --------------------
    - Linear (monotone) interpolation is used inside the nonlocal operator.
    - Grid search over admissible z0 values is performed on the z_grid itself.
    - Boundary: z_grid[0] = 0 defines the lowest achievable yield after reset.
    """
    N = len(z_grid)
    a = params.a
    xi = params.xi
    J_tilde = np.zeros(N)

    # Build a linear interpolant of w for queries at z0 targets
    # Using numpy interp for monotone 1D interpolation (equivalent to linear)
    w_interp = lambda z0: np.interp(z0, z_grid, w)

    if abs(xi) < 1e-12:
        # ---- xi = 0 case ----
        # J_tilde[w](z) = a^p * max_{0<=z0<=z/a} w(z0)
        # For nonincreasing w, this equals a^p * w(0)
        a_p = a**params.p
        for i, z in enumerate(z_grid):
            upper_z0 = z / a if a > 0 else 0.0
            # Candidate targets on grid up to upper_z0
            mask = z_grid <= upper_z0 + 1e-14
            if not np.any(mask):
                J_tilde[i] = 0.0
            else:
                w_candidates = w[mask]
                J_tilde[i] = a_p * np.max(w_candidates)
    else:
        # ---- xi > 0 case ----
        threshold = a / xi   # = a/xi divides the domain

        for i, z in enumerate(z_grid):
            if z < threshold - 1e-12:
                # Case 1: z < a/xi
                # Admissible z0 in [0, z/a]; weight w(z0)/(1-xi*z0)^p
                upper_z0 = z / a
                mask = z_grid <= upper_z0 + 1e-14
                if not np.any(mask):
                    J_tilde[i] = 0.0
                    continue
                z0_cands = z_grid[mask]
                denom = (1.0 - xi * z0_cands)
                # Only valid where denom > 0 (z0 < 1/xi)
                valid = denom > 1e-14
                if not np.any(valid):
                    J_tilde[i] = 0.0
                    continue
                obj = np.where(valid, w[mask] / denom**params.p, -np.inf)
                J_tilde[i] = (a - xi * z)**params.p * np.max(obj)

            elif abs(z - threshold) <= 1e-12:
                # Case 2: z = a/xi exactly
                # J_tilde = a^p * w(1/xi)
                target_z0 = 1.0 / xi
                J_tilde[i] = a**params.p * w_interp(target_z0)

            else:
                # Case 3: z > a/xi
                # Admissible z0 in [z/a, infinity); weight w(z0)/(xi*z0-1)^p
                lower_z0 = z / a
                mask = z_grid >= lower_z0 - 1e-14
                if not np.any(mask):
                    J_tilde[i] = 0.0
                    continue
                z0_cands = z_grid[mask]
                denom = xi * z0_cands - 1.0
                valid = denom > 1e-14
                if not np.any(valid):
                    J_tilde[i] = 0.0
                    continue
                obj = np.where(valid, w[mask] / denom**params.p, -np.inf)
                best = np.max(obj)
                if best <= -1e14:
                    J_tilde[i] = 0.0
                else:
                    J_tilde[i] = (xi * z - a)**params.p * best

    return J_tilde


def apply_local_operator(
    params: ModelParams,
    z_grid: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """
    Pointwise evaluation of L^z w(z) - (tilde_rho + p*z)*w(z) + z^p/p
    using centred differences (for diagnostic / residual checking purposes).

    Parameters
    ----------
    params : ModelParams
    z_grid : np.ndarray, shape (N,)
    w : np.ndarray, shape (N,)

    Returns
    -------
    residual : np.ndarray, shape (N,)
        HJB residual at each grid point.
    """
    N = len(z_grid)
    residual = np.zeros(N)
    sigma2 = params.sigma**2

    for i in range(N):
        z = z_grid[i]
        diff_c = 0.5 * sigma2 * z**2
        drift_c = params.drift_z(z)
        disc_c = params.discount_z(z)
        pay_c = params.running_payoff(z)

        if i == 0:
            # Forward differences
            if N > 1:
                h = z_grid[1] - z_grid[0]
                wp = (w[1] - w[0]) / h
                wpp = 0.0  # boundary: assume w''(0)=0 for diagnostic
            else:
                wp = wpp = 0.0
        elif i == N - 1:
            h = z_grid[-1] - z_grid[-2]
            wp = (w[-1] - w[-2]) / h
            wpp = 0.0
        else:
            h_b = z_grid[i] - z_grid[i - 1]
            h_f = z_grid[i + 1] - z_grid[i]
            wp = (w[i + 1] - w[i - 1]) / (h_b + h_f)
            wpp = 2.0 * (w[i + 1] / h_f - w[i] * (1.0 / h_b + 1.0 / h_f) + w[i - 1] / h_b) / (h_b + h_f)

        lw = drift_c * wp + diff_c * wpp
        residual[i] = lw - disc_c * w[i] + pay_c

    return residual
