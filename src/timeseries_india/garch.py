"""GARCH(p, q) volatility model with a constant mean, Gaussian quasi-MLE."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .utils import to_1d_array


@dataclass
class GARCHResults:
    p: int
    q: int
    mu: float
    omega: float
    alpha: np.ndarray  # ARCH coefficients (q of them)
    beta: np.ndarray   # GARCH coefficients (p of them)
    loglik: float
    nobs: int
    conditional_vol: np.ndarray = field(repr=False)
    resid: np.ndarray = field(repr=False)

    @property
    def nparams(self) -> int:
        return 2 + self.p + self.q  # mu, omega, alphas, betas

    @property
    def persistence(self) -> float:
        return float(self.alpha.sum() + self.beta.sum())

    @property
    def unconditional_vol(self) -> float:
        denom = 1.0 - self.persistence
        return float(np.sqrt(self.omega / denom)) if denom > 0 else float("nan")

    @property
    def aic(self) -> float:
        return 2 * self.nparams - 2 * self.loglik

    @property
    def bic(self) -> float:
        return self.nparams * np.log(self.nobs) - 2 * self.loglik

    def summary(self) -> str:
        lines = [
            f"GARCH({self.p},{self.q}) — Gaussian MLE",
            "=" * 44,
            f"nobs={self.nobs}  loglik={self.loglik:.3f}  "
            f"AIC={self.aic:.2f}  BIC={self.bic:.2f}",
            f"persistence={self.persistence:.4f}  "
            f"uncond.vol={self.unconditional_vol:.6f}",
            f"{'mu':<12}{self.mu: .6f}",
            f"{'omega':<12}{self.omega: .8f}",
        ]
        for i, v in enumerate(self.alpha, 1):
            lines.append(f"{'alpha[' + str(i) + ']':<12}{v: .6f}")
        for j, v in enumerate(self.beta, 1):
            lines.append(f"{'beta[' + str(j) + ']':<12}{v: .6f}")
        return "\n".join(lines)


class GARCH:
    """Generalised ARCH model ``GARCH(p, q)`` for conditional variance.

    With returns :math:`r_t = \\mu + \\varepsilon_t`,
    :math:`\\varepsilon_t = \\sigma_t z_t`, :math:`z_t \\sim \\mathcal{N}(0,1)`:

    .. math::
        \\sigma_t^2 = \\omega + \\sum_{i=1}^q \\alpha_i \\varepsilon_{t-i}^2
                      + \\sum_{j=1}^p \\beta_j \\sigma_{t-j}^2.

    Here ``p`` is the GARCH (lagged-variance) order and ``q`` the ARCH
    (lagged-shock) order, matching the conventional ``GARCH(p, q)`` notation.
    Setting ``p=0`` yields an ``ARCH(q)`` model.
    """

    def __init__(self, p: int = 1, q: int = 1):
        if p < 0 or q < 1:
            raise ValueError("Require q >= 1 and p >= 0.")
        self.p = int(p)
        self.q = int(q)
        self.results_: GARCHResults | None = None

    # ------------------------------------------------------------------ #
    def _filter(self, omega, alpha, beta, eps2):
        """Recursively compute the conditional-variance series."""
        n = eps2.size
        sigma2 = np.empty(n)
        uncond = np.mean(eps2)  # variance targeting for the pre-sample
        m = max(self.p, self.q)
        sigma2[:m] = uncond
        for t in range(m, n):
            arch = np.dot(alpha, eps2[t - self.q : t][::-1]) if self.q else 0.0
            garch = np.dot(beta, sigma2[t - self.p : t][::-1]) if self.p else 0.0
            sigma2[t] = omega + arch + garch
        return sigma2

    def _unpack(self, params):
        mu = params[0]
        omega = params[1]
        alpha = params[2 : 2 + self.q]
        beta = params[2 + self.q : 2 + self.q + self.p]
        return mu, omega, alpha, beta

    def _neg_loglik(self, params, r):
        mu, omega, alpha, beta = self._unpack(params)
        if omega <= 0 or np.any(alpha < 0) or np.any(beta < 0):
            return 1e12
        if alpha.sum() + beta.sum() >= 0.99999:
            return 1e12
        eps = r - mu
        sigma2 = self._filter(omega, alpha, beta, eps**2)
        if np.any(sigma2 <= 0) or not np.all(np.isfinite(sigma2)):
            return 1e12
        ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + eps**2 / sigma2)
        return -ll

    def fit(self, x) -> GARCHResults:
        """Estimate parameters from a (mean-stationary) return series ``x``."""
        r = to_1d_array(x)
        if r.size <= self.p + self.q + 2:
            raise ValueError("Series too short for the requested GARCH order.")

        var = np.var(r)
        init = np.concatenate(
            [
                [np.mean(r)],
                [var * 0.1],
                np.full(self.q, 0.05),
                np.full(self.p, 0.90 / max(self.p, 1)) if self.p else np.array([]),
            ]
        )
        bounds = (
            [(None, None), (1e-12, None)]
            + [(0.0, 1.0)] * self.q
            + [(0.0, 1.0)] * self.p
        )
        res = minimize(
            self._neg_loglik,
            init,
            args=(r,),
            method="L-BFGS-B",
            bounds=bounds,
        )
        mu, omega, alpha, beta = self._unpack(res.x)
        eps = r - mu
        sigma2 = self._filter(omega, alpha, beta, eps**2)
        self.results_ = GARCHResults(
            p=self.p,
            q=self.q,
            mu=float(mu),
            omega=float(omega),
            alpha=np.asarray(alpha, float),
            beta=np.asarray(beta, float),
            loglik=float(-res.fun),
            nobs=r.size,
            conditional_vol=np.sqrt(sigma2),
            resid=eps,
        )
        self._eps = eps
        self._sigma2 = sigma2
        return self.results_

    # ------------------------------------------------------------------ #
    def forecast(self, h: int = 1) -> dict:
        """Forecast conditional variance/volatility ``h`` steps ahead.

        Returns a dict with ``variance`` and ``volatility`` arrays of length ``h``.
        """
        if self.results_ is None:
            raise RuntimeError("Call fit() before forecast().")
        r = self.results_
        eps2 = self._eps**2
        sigma2 = self._sigma2

        fvar = []
        last_eps2 = list(eps2[-r.q:]) if r.q else []
        last_s2 = list(sigma2[-r.p:]) if r.p else []
        for step in range(1, h + 1):
            # For multi-step horizons E[eps^2_{t+k}] = sigma^2_{t+k}, so the
            # appended forecasts feed both the ARCH and GARCH recursions.
            arch = sum(r.alpha[i] * last_eps2[-1 - i] for i in range(r.q)) if r.q else 0.0
            garch = sum(r.beta[j] * last_s2[-1 - j] for j in range(r.p)) if r.p else 0.0
            s2 = r.omega + arch + garch
            fvar.append(s2)
            last_eps2.append(s2)   # future squared shocks ≈ their variance
            last_s2.append(s2)
            if len(last_eps2) > r.q and r.q:
                last_eps2 = last_eps2[-r.q:]
            if len(last_s2) > r.p and r.p:
                last_s2 = last_s2[-r.p:]
        fvar = np.array(fvar)
        return {"variance": fvar, "volatility": np.sqrt(fvar)}

    def simulate(self, n: int, burn: int = 500, seed: int | None = None):
        """Simulate returns and conditional volatility from fitted parameters."""
        if self.results_ is None:
            raise RuntimeError("Call fit() before simulate().")
        r = self.results_
        rng = np.random.default_rng(seed)
        total = n + burn
        eps = np.zeros(total)
        sigma2 = np.zeros(total)
        uncond = r.unconditional_vol**2
        m = max(r.p, r.q)
        sigma2[:m] = uncond
        eps[:m] = rng.normal(0, np.sqrt(uncond), m)
        for t in range(m, total):
            arch = np.dot(r.alpha, (eps[t - r.q : t] ** 2)[::-1]) if r.q else 0.0
            garch = np.dot(r.beta, sigma2[t - r.p : t][::-1]) if r.p else 0.0
            sigma2[t] = r.omega + arch + garch
            eps[t] = np.sqrt(sigma2[t]) * rng.standard_normal()
        return {
            "returns": r.mu + eps[burn:],
            "volatility": np.sqrt(sigma2[burn:]),
        }
