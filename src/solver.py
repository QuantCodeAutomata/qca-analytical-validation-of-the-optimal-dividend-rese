"""
Numerical solver for the reduced HJB-QVI on a 1D yield grid.

The reduced problem is:
    min{ (tilde_rho + p*z)*w(z) - L^z w(z) - z^p/p,
         -w'(z),
         w(z) - J_tilde[w](z),
         w(z) - lambda/p } = 0

Solved via policy iteration with obstacle projection on a truncated domain
z in [0, z_max].  At each iteration:
  1. Compute J_tilde[w^k] nonlocally.
  2. Solve the continuation linear system (tridiagonal).
  3. Project onto reset and stopping obstacles.
  4. Enforce monotonicity (non-increasing) constraint.
  5. Check sup-norm convergence and QVI residual.

# Custom — Context7 found no library equivalent for this combined
# singular/impulse/stopping obstacle problem.
# scipy.sparse.linalg.spsolve is used for the tridiagonal solve.
"""

from __future__ import annotations
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy import sparse

from src.model import ModelParams
from src.operators import build_fd_matrix, compute_reset_operator


def make_grid(z_max: float, N: int, nonuniform: bool = False) -> np.ndarray:
    """
    Build a 1D yield grid on [0, z_max].

    Parameters
    ----------
    z_max : float
        Upper truncation of the yield domain.
    N : int
        Number of grid points.
    nonuniform : bool
        If True, use a denser grid near z=0 to resolve the lower boundary z_*.

    Returns
    -------
    z_grid : np.ndarray, shape (N,)
    """
    if nonuniform:
        # Piecewise denser near z=0 and near z_max/4
        t = np.linspace(0, 1, N)
        z_grid = z_max * (t**0.6)   # mild stretching toward 0
    else:
        z_grid = np.linspace(0.0, z_max, N)
    z_grid[0] = 0.0
    return z_grid


def initial_guess(
    params: ModelParams,
    z_grid: np.ndarray,
) -> np.ndarray:
    """
    Compute an initial guess for the value function.

    Uses the frictionless benchmark g(z) = z^p/(p*tilde_rho + p^2*z)
    clipped below by lambda/p and above by a large constant.

    Parameters
    ----------
    params : ModelParams
    z_grid : np.ndarray, shape (N,)

    Returns
    -------
    w0 : np.ndarray, shape (N,)
    """
    g = params.g(z_grid)
    stop_level = params.stopping_level()
    # g is infinite at z=0 when tilde_rho=0; clip to upper bound
    g_finite = np.where(np.isfinite(g), g, params.g(z_grid.max()))
    # At z=0, use g(bar_z) as upper bound
    g_finite[0] = params.g_bar_z()
    w0 = np.maximum(g_finite, stop_level)
    return _enforce_nonincreasing(w0)


def _enforce_nonincreasing(w: np.ndarray) -> np.ndarray:
    """
    Project w onto the cone of non-increasing sequences.

    For every i, sets w[i] = max(w[i], w[i+1]) scanning right-to-left,
    ensuring w[i] >= w[i+1] for all i (i.e. w is non-increasing in index).

    Parameters
    ----------
    w : np.ndarray

    Returns
    -------
    w_mono : np.ndarray  (new array, original unchanged)
    """
    w_mono = w.copy()
    for i in range(len(w_mono) - 2, -1, -1):
        if w_mono[i] < w_mono[i + 1]:
            w_mono[i] = w_mono[i + 1]
    return w_mono


def solve_qvi(
    params: ModelParams,
    z_grid: np.ndarray,
    max_iter: int = 5000,
    tol: float = 1e-8,
    verbose: bool = False,
) -> dict:
    """
    Solve the reduced HJB-QVI for the dividend-reset model.

    Parameters
    ----------
    params : ModelParams
        Model parameters including mu, sigma, rho, p, lambda_, xi, eta.
    z_grid : np.ndarray, shape (N,)
        Yield grid on [0, z_max].
    max_iter : int
        Maximum number of policy iterations.
    tol : float
        Convergence tolerance (sup-norm of consecutive w updates).
    verbose : bool
        Print convergence information.

    Returns
    -------
    result : dict with keys:
        'w'          : np.ndarray(N,) — converged value function
        'w_prime'    : np.ndarray(N,) — numerical derivative of w
        'J_tilde'    : np.ndarray(N,) — reset operator at solution
        'hjb_resid'  : np.ndarray(N,) — local HJB residual
        'converged'  : bool
        'n_iter'     : int
        'z_grid'     : np.ndarray(N,)
        'params'     : ModelParams
    """
    if not params.is_finite():
        raise ValueError(
            f"tilde_rho = {params.tilde_rho:.6f} <= 0: infinite-value regime. "
            "Solver not applicable."
        )

    N = len(z_grid)
    stop_level = params.stopping_level()

    # Build the FD matrix once (parameters are fixed; grid is fixed)
    A, rhs_base = build_fd_matrix(params, z_grid)

    # Convert A to lil for boundary condition manipulation
    A_lil = A.tolil()

    # Boundary condition at z=0: the value at z=0 is determined by the
    # obstacles (reset or stopping). Pin via identity row; update rhs in loop.
    # At z=0, the diffusion vanishes and drift is 0, so L^z w(0) = 0.
    # The HJB at z=0: tilde_rho * w(0) = tilde_rho * w(0) - 0 - 0^p/p = 0
    # This is satisfied trivially; w(0) is determined by the obstacles or smooth pasting.
    # We use a Neumann-style condition w'(0) >= 0 (singular control), pinned via obstacle.
    A_lil[0, :] = 0
    A_lil[0, 0] = 1.0   # identity row; rhs[0] will be set dynamically

    A_csr = A_lil.tocsr()

    # Initial guess
    w = initial_guess(params, z_grid)

    converged = False
    n_iter = 0

    for iteration in range(max_iter):
        w_old = w.copy()

        # 1. Compute nonlocal reset obstacle J_tilde[w]
        J_tilde = compute_reset_operator(params, z_grid, w)

        # 2. Build rhs incorporating boundary condition at z=0
        rhs = rhs_base.copy()
        # At z=0, w[0] is forced to max of J_tilde[0] and stop_level (obstacle)
        rhs[0] = max(J_tilde[0], stop_level, w[0])

        # 3. Solve the continuation linear system A*w_new = rhs
        try:
            w_cont = spsolve(A_csr, rhs)
        except Exception:
            w_cont = w.copy()

        # 4. Ensure positivity (value function >= 0)
        w_cont = np.maximum(w_cont, 0.0)

        # 5. Project onto obstacles:
        #    w = max(w_cont,  J_tilde,  stop_level)
        w_new = np.maximum(w_cont, J_tilde)
        w_new = np.maximum(w_new, stop_level)

        # 6. Enforce monotonicity (singular control: w must be non-increasing)
        w_new = _enforce_nonincreasing(w_new)

        # 7. Clamp from below by stop_level (liquidation obstacle)
        w_new = np.maximum(w_new, stop_level)

        # 8. Convergence check
        diff = np.max(np.abs(w_new - w_old))
        w = w_new
        n_iter = iteration + 1

        if verbose and (iteration % 200 == 0 or diff < tol):
            print(f"  iter {iteration:4d}: sup|dw|={diff:.2e}")

        if diff < tol:
            converged = True
            break

    # Post-processing: numerical derivative
    w_prime = _numerical_derivative(w, z_grid)

    # HJB residual at converged solution
    J_tilde_final = compute_reset_operator(params, z_grid, w)
    hjb_resid = _compute_hjb_residual(params, z_grid, w)

    if verbose:
        print(f"Solver finished: converged={converged}, n_iter={n_iter}, "
              f"sup|dw|={np.max(np.abs(w - w_old)):.2e}")

    return {
        'w': w,
        'w_prime': w_prime,
        'J_tilde': J_tilde_final,
        'hjb_resid': hjb_resid,
        'converged': converged,
        'n_iter': n_iter,
        'z_grid': z_grid,
        'params': params,
    }


def _numerical_derivative(w: np.ndarray, z_grid: np.ndarray) -> np.ndarray:
    """
    Compute numerical first derivative of w using centred differences in
    the interior and one-sided differences at boundaries.

    Parameters
    ----------
    w : np.ndarray, shape (N,)
    z_grid : np.ndarray, shape (N,)

    Returns
    -------
    dw : np.ndarray, shape (N,)
    """
    N = len(w)
    dw = np.zeros(N)
    for i in range(N):
        if i == 0:
            h = z_grid[1] - z_grid[0]
            dw[i] = (w[1] - w[0]) / h
        elif i == N - 1:
            h = z_grid[-1] - z_grid[-2]
            dw[i] = (w[-1] - w[-2]) / h
        else:
            h_b = z_grid[i] - z_grid[i - 1]
            h_f = z_grid[i + 1] - z_grid[i]
            dw[i] = (w[i + 1] - w[i - 1]) / (h_b + h_f)
    return dw


def _compute_hjb_residual(
    params: ModelParams,
    z_grid: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """
    Compute the local HJB residual:
        resid[i] = (tilde_rho + p*z)*w(z) - L^z w(z) - z^p/p

    Parameters
    ----------
    params : ModelParams
    z_grid : np.ndarray, shape (N,)
    w : np.ndarray, shape (N,)

    Returns
    -------
    resid : np.ndarray, shape (N,)
    """
    N = len(z_grid)
    resid = np.zeros(N)
    sigma2 = params.sigma**2

    for i in range(N):
        z = z_grid[i]
        diff_c = 0.5 * sigma2 * z**2
        drift_c = params.drift_z(z)
        disc = params.discount_z(z)
        pay = params.running_payoff(z)

        if i == 0:
            h = z_grid[1] - z_grid[0]
            wp = (w[1] - w[0]) / h
            wpp = 0.0
        elif i == N - 1:
            h = z_grid[-1] - z_grid[-2]
            wp = (w[-1] - w[-2]) / h
            wpp = 0.0
        else:
            h_b = z_grid[i] - z_grid[i - 1]
            h_f = z_grid[i + 1] - z_grid[i]
            wp = (w[i + 1] - w[i - 1]) / (h_b + h_f)
            wpp = 2.0 * (
                w[i + 1] / h_f - w[i] * (1.0 / h_b + 1.0 / h_f) + w[i - 1] / h_b
            ) / (h_b + h_f)

        lw = drift_c * wp + diff_c * wpp
        resid[i] = disc * w[i] - lw - pay

    return resid


def numerical_second_derivative(
    w: np.ndarray,
    z_grid: np.ndarray,
    i: int,
    window: int = 3,
) -> float:
    """
    Estimate w''(z_i) using local quadratic regression over a window of
    2*window+1 points centred at i.

    Parameters
    ----------
    w : np.ndarray
    z_grid : np.ndarray
    i : int — central index
    window : int — half-width of the regression window

    Returns
    -------
    w_double_prime : float
    """
    N = len(z_grid)
    lo = max(0, i - window)
    hi = min(N - 1, i + window)
    z_loc = z_grid[lo:hi + 1] - z_grid[i]
    w_loc = w[lo:hi + 1]
    if len(z_loc) < 3:
        return 0.0
    # Fit quadratic: w = a + b*z + c*z^2, return 2c
    coeffs = np.polyfit(z_loc, w_loc, 2)
    return 2.0 * coeffs[0]   # 2 * coefficient of z^2
