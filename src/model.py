"""
Model parameters and analytical quantities for the continuous-time optimal
dividend-reset control model.

The controlled equity process is:
    dX_t = (mu*X_t - C_t)*dt + sigma*X_t*dW_t - dP_t

where P_t = sum_{s<=t} [eta*X_{s-} + xi*Delta_C_s^-] * 1_{Delta_C_s^- > 0}

The reduced value function w(z) satisfies V(x, c) = x^p * w(c/x)  for z = c/x.

References:
    Paper Section 2-5 on the HJB-QVI and homogeneity reduction.
    # Custom — Context7 found no library equivalent for this domain-specific model.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ModelParams:
    """
    Parameter container for the dividend-reset optimal control model.

    Attributes
    ----------
    mu : float
        Drift rate of equity process (mu > 0).
    sigma : float
        Volatility of equity process (sigma > 0).
    rho : float
        Discount rate (rho > 0).
    p : float
        Power utility exponent, 0 < p < 1.
    lambda_ : float
        Liquidation utility weight (lambda >= 0).
    xi : float
        Proportional reset friction on dividend reduction (xi >= 0).
    eta : float
        Proportional reset friction on equity (0 <= eta < 1).
    """
    mu: float
    sigma: float
    rho: float
    p: float
    lambda_: float
    xi: float
    eta: float

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.mu <= 0:
            raise ValueError(f"mu must be > 0, got {self.mu}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {self.sigma}")
        if self.rho <= 0:
            raise ValueError(f"rho must be > 0, got {self.rho}")
        if not (0 < self.p < 1):
            raise ValueError(f"p must be in (0,1), got {self.p}")
        if self.lambda_ < 0:
            raise ValueError(f"lambda_ must be >= 0, got {self.lambda_}")
        if self.xi < 0:
            raise ValueError(f"xi must be >= 0, got {self.xi}")
        if not (0 <= self.eta < 1):
            raise ValueError(f"eta must be in [0,1), got {self.eta}")

    @property
    def a(self) -> float:
        """Retained fraction of equity after reset: a = 1 - eta."""
        return 1.0 - self.eta

    @property
    def tilde_rho(self) -> float:
        """
        Effective discount rate: tilde_rho = rho - mu*p + 0.5*sigma^2*p*(1-p).

        Finiteness condition: tilde_rho > 0.
        """
        return self.rho - self.mu * self.p + 0.5 * self.sigma**2 * self.p * (1.0 - self.p)

    @property
    def bar_z(self) -> float:
        """
        Frictionless optimal yield: bar_z = tilde_rho / (1 - p).

        Only meaningful when the value function is finite (tilde_rho > 0).
        """
        if self.tilde_rho <= 0:
            raise ValueError("bar_z is undefined when tilde_rho <= 0 (infinite-value regime).")
        return self.tilde_rho / (1.0 - self.p)

    def is_finite(self) -> bool:
        """Return True iff the value function is finite (tilde_rho > 0)."""
        return self.tilde_rho > 0

    def w_frictionless(self) -> float:
        """
        Reduced value function constant in the frictionless case (xi=0, eta=0).

        w_fric = (1/p) * bar_z^(p-1)
        """
        bz = self.bar_z
        return (1.0 / self.p) * bz ** (self.p - 1.0)

    def g(self, z: np.ndarray | float) -> np.ndarray | float:
        """
        Upper bound function g(z) = z^p / (p*tilde_rho + p^2*z).

        Parameters
        ----------
        z : array-like or float
            Dividend yield state.

        Returns
        -------
        array-like or float
            Upper bound values.
        """
        z = np.asarray(z, dtype=float)
        denom = self.p * self.tilde_rho + self.p**2 * z
        # Avoid division by zero at z=0 when tilde_rho=0 (not expected in finite regime)
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(denom > 0, z**self.p / denom, np.inf)
        return result

    def g_bar_z(self) -> float:
        """Upper bound evaluated at bar_z: g(bar_z)."""
        bz = self.bar_z
        return self.g(bz)

    def drift_z(self, z: np.ndarray | float) -> np.ndarray | float:
        """
        Reduced drift coefficient for the local operator L^z:
            drift_z(z) = (-mu + z + sigma^2*(1-p)) * z

        Parameters
        ----------
        z : array-like or float

        Returns
        -------
        array-like or float
        """
        z = np.asarray(z, dtype=float)
        return (-self.mu + z + self.sigma**2 * (1.0 - self.p)) * z

    def diffusion_z(self, z: np.ndarray | float) -> np.ndarray | float:
        """
        Reduced diffusion coefficient for L^z:
            diffusion_z(z) = 0.5 * sigma^2 * z^2

        Parameters
        ----------
        z : array-like or float

        Returns
        -------
        array-like or float
        """
        z = np.asarray(z, dtype=float)
        return 0.5 * self.sigma**2 * z**2

    def discount_z(self, z: np.ndarray | float) -> np.ndarray | float:
        """
        Reduced discount coefficient: tilde_rho + p*z.

        Parameters
        ----------
        z : array-like or float

        Returns
        -------
        array-like or float
        """
        z = np.asarray(z, dtype=float)
        return self.tilde_rho + self.p * z

    def running_payoff(self, z: np.ndarray | float) -> np.ndarray | float:
        """
        Reduced running payoff: z^p / p.

        Parameters
        ----------
        z : array-like or float

        Returns
        -------
        array-like or float
        """
        z = np.asarray(z, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(z > 0, z**self.p / self.p, 0.0)

    def stopping_level(self) -> float:
        """
        Stopping obstacle level: lambda / p.

        Returns
        -------
        float
        """
        return self.lambda_ / self.p


BASELINE_PARAMS_FIG2_ETA01 = ModelParams(
    mu=0.07, sigma=0.25, rho=0.08, p=0.5, lambda_=0.5, xi=1.0, eta=0.1
)
BASELINE_PARAMS_FIG2_ETA0 = ModelParams(
    mu=0.07, sigma=0.25, rho=0.08, p=0.5, lambda_=0.5, xi=1.0, eta=1e-6
)
BASELINE_PARAMS_FIG3 = ModelParams(
    mu=0.07, sigma=0.25, rho=0.08, p=0.5, lambda_=0.5, xi=0.0, eta=0.1
)
BASELINE_PARAMS_FIG4 = ModelParams(
    mu=0.07, sigma=0.25, rho=0.08, p=0.5, lambda_=0.5, xi=0.1, eta=0.1
)
BASELINE_PARAMS_FRICTIONLESS = ModelParams(
    mu=0.07, sigma=0.25, rho=0.08, p=0.5, lambda_=0.5, xi=0.0, eta=0.0
)
