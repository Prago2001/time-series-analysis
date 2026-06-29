"""ARMA(p, q) model estimated by conditional maximum likelihood (CSS)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from scipy.optimize import minimize

from .utils import to_1d_array


@dataclass
class ARMAResults:
    """Container for a fitted :class:`ARMA` model."""

    p: int
    q: int
    mu: float
    ar_params: np.ndarray
    ma_params: np.ndarray
    sigma2: float
    loglik: float
    nobs: int
    resid: np.ndarray = field(repr=False)

    @property
    def nparams(self) -> int:
        return self.p + self.q + 2  # ar, ma, mu, sigma2

    @property
    def aic(self) -> float:
        return 2 * self.nparams - 2 * self.loglik

    @property
    def bic(self) -> float:
        return self.nparams * np.log(self.nobs) - 2 * self.loglik

    def summary(self) -> str:
        lines = [
            f"ARMA({self.p},{self.q}) — conditional MLE",
            "=" * 44,
            f"nobs={self.nobs}  loglik={self.loglik:.3f}  "
            f"AIC={self.aic:.2f}  BIC={self.bic:.2f}",
            f"{'mu (const)':<14}{self.mu: .6f}",
        ]
        for i, v in enumerate(self.ar_params, 1):
            lines.append(f"{'ar.L' + str(i):<14}{v: .6f}")
        for j, v in enumerate(self.ma_params, 1):
            lines.append(f"{'ma.L' + str(j):<14}{v: .6f}")
        lines.append(f"{'sigma2':<14}{self.sigma2: .6f}")
        return "\n".join(lines)


class ARMA:
    """Autoregressive moving-average model ``ARMA(p, q)``.

    The process is

    .. math::
        x_t - \\mu = \\sum_{i=1}^p \\phi_i (x_{t-i} - \\mu)
                     + \\varepsilon_t + \\sum_{j=1}^q \\theta_j \\varepsilon_{t-j},

    with :math:`\\varepsilon_t \\sim \\mathcal{N}(0, \\sigma^2)`.  Parameters are
    estimated by minimising the conditional sum of squared residuals, which is
    equivalent to conditional Gaussian maximum likelihood.

    Set ``q=0`` for a pure AR(p) model and ``p=0`` for a pure MA(q) model.
    """

    def __init__(self, p: int = 1, q: int = 0):
        if p < 0 or q < 0:
            raise ValueError("p and q must be non-negative integers.")
        self.p = int(p)
        self.q = int(q)
        self.results_: ARMAResults | None = None
        self._endog: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    def _residuals(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        p, q = self.p, self.q
        mu = params[0]
        phi = params[1 : 1 + p]
        theta = params[1 + p : 1 + p + q]
        n = x.size
        y = x - mu
        e = np.zeros(n)
        start = p
        for t in range(start, n):
            ar = np.dot(phi, y[t - p : t][::-1]) if p else 0.0
            ma = 0.0
            for j in range(1, q + 1):
                if t - j >= 0:
                    ma += theta[j - 1] * e[t - j]
            e[t] = y[t] - ar - ma
        return e[start:]

    def _neg_loglik(self, params: np.ndarray, x: np.ndarray) -> float:
        e = self._residuals(params, x)
        n = e.size
        sigma2 = np.mean(e**2)
        if sigma2 <= 0 or not np.isfinite(sigma2):
            return 1e12
        return 0.5 * n * (np.log(2 * np.pi * sigma2) + 1.0)

    def fit(self, x) -> ARMAResults:
        """Estimate the model parameters from observed series ``x``."""
        x = to_1d_array(x)
        if x.size <= self.p + self.q + 1:
            raise ValueError("Series too short for the requested ARMA order.")
        self._endog = x

        p, q = self.p, self.q
        init = np.zeros(1 + p + q)
        init[0] = x.mean()
        if p:
            init[1] = 0.1
        # Keep AR/MA coefficients in a generous box to encourage stable optima.
        bounds = [(None, None)] + [(-0.999, 0.999)] * (p + q)
        res = minimize(
            self._neg_loglik,
            init,
            args=(x,),
            method="L-BFGS-B",
            bounds=bounds,
        )
        params = res.x
        e = self._residuals(params, x)
        sigma2 = float(np.mean(e**2))
        loglik = -self._neg_loglik(params, x)
        self.results_ = ARMAResults(
            p=p,
            q=q,
            mu=float(params[0]),
            ar_params=params[1 : 1 + p].copy(),
            ma_params=params[1 + p : 1 + p + q].copy(),
            sigma2=sigma2,
            loglik=float(loglik),
            nobs=e.size,
            resid=e,
        )
        return self.results_

    # ------------------------------------------------------------------ #
    def _ma_infinity(self, h: int) -> np.ndarray:
        """psi-weights of the MA(inf) representation up to lag ``h-1``."""
        r = self.results_
        phi, theta = r.ar_params, r.ma_params
        psi = np.zeros(h)
        psi[0] = 1.0
        for j in range(1, h):
            val = theta[j - 1] if j - 1 < len(theta) else 0.0
            for i in range(1, min(j, len(phi)) + 1):
                val += phi[i - 1] * psi[j - i]
            psi[j] = val
        return psi

    def forecast(self, h: int = 1, alpha: float = 0.05):
        """Forecast ``h`` steps ahead with ``(1-alpha)`` prediction intervals.

        Returns a dict with ``mean``, ``se``, ``lower`` and ``upper`` arrays.
        """
        if self.results_ is None:
            raise RuntimeError("Call fit() before forecast().")
        r = self.results_
        x = self._endog
        p, q = self.p, self.q
        mu = r.mu
        phi, theta = r.ar_params, r.ma_params

        y = list(x - mu)
        e = list(r.resid)
        e = [0.0] * (len(y) - len(e)) + e  # pad to align with y

        preds = []
        for _ in range(h):
            ar = np.dot(phi, np.array(y[-p:])[::-1]) if p else 0.0
            ma = np.dot(theta, np.array(e[-q:])[::-1]) if q else 0.0
            yhat = ar + ma
            y.append(yhat)
            e.append(0.0)  # future shocks have zero expectation
            preds.append(yhat + mu)

        preds = np.array(preds)
        psi = self._ma_infinity(h)
        var = r.sigma2 * np.cumsum(psi**2)
        se = np.sqrt(var)
        z = stats.norm.ppf(1 - alpha / 2)
        return {
            "mean": preds,
            "se": se,
            "lower": preds - z * se,
            "upper": preds + z * se,
        }

    def simulate(self, n: int, burn: int = 200, seed: int | None = None) -> np.ndarray:
        """Simulate a path of length ``n`` from the *fitted* parameters."""
        if self.results_ is None:
            raise RuntimeError("Call fit() before simulate().")
        r = self.results_
        rng = np.random.default_rng(seed)
        p, q = self.p, self.q
        phi, theta = r.ar_params, r.ma_params
        total = n + burn
        e = rng.normal(0, np.sqrt(r.sigma2), total)
        y = np.zeros(total)
        for t in range(total):
            ar = np.dot(phi, y[max(0, t - p):t][::-1]) if p and t else 0.0
            ma = np.dot(theta, e[max(0, t - q):t][::-1]) if q and t else 0.0
            y[t] = ar + ma + e[t]
        return y[burn:] + r.mu
