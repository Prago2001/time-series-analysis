"""Vector Autoregression VAR(p): OLS estimation, forecasting and impulse responses."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class VARResults:
    p: int
    k: int  # number of variables
    names: list[str]
    const: np.ndarray             # (k,)
    coefs: np.ndarray             # (p, k, k) lag matrices A_1..A_p
    sigma_u: np.ndarray           # (k, k) residual covariance
    nobs: int
    resid: np.ndarray = field(repr=False)

    @property
    def nparams(self) -> int:
        return self.k * (1 + self.k * self.p)

    @property
    def aic(self) -> float:
        ld = np.linalg.slogdet(self.sigma_u)[1]
        return ld + 2.0 * self.nparams / self.nobs

    @property
    def bic(self) -> float:
        ld = np.linalg.slogdet(self.sigma_u)[1]
        return ld + np.log(self.nobs) * self.nparams / self.nobs

    def is_stable(self) -> bool:
        """True if all companion-matrix eigenvalues lie inside the unit circle."""
        comp = _companion(self.coefs)
        return bool(np.all(np.abs(np.linalg.eigvals(comp)) < 1.0))

    def summary(self) -> str:
        lines = [
            f"VAR({self.p}) — OLS  [{', '.join(self.names)}]",
            "=" * 44,
            f"nobs={self.nobs}  AIC={self.aic:.4f}  BIC={self.bic:.4f}  "
            f"stable={self.is_stable()}",
        ]
        return "\n".join(lines)


def _companion(coefs: np.ndarray) -> np.ndarray:
    """Build the (kp x kp) companion matrix from lag coefficient matrices."""
    p, k, _ = coefs.shape
    comp = np.zeros((k * p, k * p))
    comp[:k] = np.hstack([coefs[i] for i in range(p)])
    if p > 1:
        comp[k:, : k * (p - 1)] = np.eye(k * (p - 1))
    return comp


class VAR:
    """Reduced-form Vector Autoregression of order ``p``.

    .. math::
        y_t = c + A_1 y_{t-1} + \\dots + A_p y_{t-p} + u_t,
        \\quad u_t \\sim (0, \\Sigma_u).

    Each equation is estimated by ordinary least squares (which is also the
    Gaussian MLE here because every equation shares the same regressors).
    """

    def __init__(self, p: int = 1):
        if p < 1:
            raise ValueError("VAR order p must be >= 1.")
        self.p = int(p)
        self.results_: VARResults | None = None
        self._data: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    def _build_design(self, y: np.ndarray):
        p, (n, k) = self.p, y.shape
        rows = n - p
        Z = np.ones((rows, 1 + k * p))
        for t in range(p, n):
            lags = np.concatenate([y[t - i] for i in range(1, p + 1)])
            Z[t - p, 1:] = lags
        Y = y[p:]
        return Y, Z

    def fit(self, data) -> VARResults:
        """Fit the VAR to a 2-D array or :class:`pandas.DataFrame` (T x k)."""
        if isinstance(data, pd.DataFrame):
            names = list(data.columns.astype(str))
            y = data.to_numpy(dtype=float)
        else:
            y = np.asarray(data, dtype=float)
            names = [f"y{i + 1}" for i in range(y.shape[1])]
        if y.ndim != 2 or y.shape[1] < 1:
            raise ValueError("data must be a 2-D array of shape (T, k).")
        n, k = y.shape
        if n <= self.p * k + 1:
            raise ValueError("Not enough observations for the requested VAR order.")
        self._data = y

        Y, Z = self._build_design(y)
        # OLS:  B = (Z'Z)^{-1} Z'Y , with rows = [const; A_1; ...; A_p]
        B, *_ = np.linalg.lstsq(Z, Y, rcond=None)
        resid = Y - Z @ B
        dof = Z.shape[0] - Z.shape[1]
        sigma_u = (resid.T @ resid) / dof

        const = B[0]
        coefs = np.stack([B[1 + i * k : 1 + (i + 1) * k].T for i in range(self.p)])
        self.results_ = VARResults(
            p=self.p,
            k=k,
            names=names,
            const=const,
            coefs=coefs,
            sigma_u=sigma_u,
            nobs=Y.shape[0],
            resid=resid,
        )
        return self.results_

    # ------------------------------------------------------------------ #
    def forecast(self, h: int = 1, alpha: float = 0.05) -> dict:
        """Forecast ``h`` steps ahead with Gaussian prediction intervals."""
        if self.results_ is None:
            raise RuntimeError("Call fit() before forecast().")
        r = self.results_
        history = [self._data[-i] for i in range(1, self.p + 1)]  # y_{t}, y_{t-1}, ...
        preds = []
        for _ in range(h):
            yhat = r.const.copy()
            for i in range(self.p):
                yhat = yhat + r.coefs[i] @ history[i]
            preds.append(yhat)
            history = [yhat] + history[:-1]
        preds = np.array(preds)

        # Forecast-error covariance via the MA(inf) (psi) representation.
        psi = self.ma_representation(h)
        fe_cov = np.zeros((h, r.k, r.k))
        acc = np.zeros((r.k, r.k))
        for step in range(h):
            acc = acc + psi[step] @ r.sigma_u @ psi[step].T
            fe_cov[step] = acc
        se = np.sqrt(np.stack([np.diag(c) for c in fe_cov]))
        z = stats.norm.ppf(1 - alpha / 2)
        return {
            "mean": preds,
            "se": se,
            "lower": preds - z * se,
            "upper": preds + z * se,
            "names": r.names,
        }

    def ma_representation(self, h: int) -> np.ndarray:
        """psi_0..psi_{h-1} matrices of the Wold/MA(inf) representation."""
        r = self.results_
        k, p = r.k, self.p
        psi = np.zeros((h, k, k))
        psi[0] = np.eye(k)
        for s in range(1, h):
            acc = np.zeros((k, k))
            for i in range(1, min(s, p) + 1):
                acc += r.coefs[i - 1] @ psi[s - i]
            psi[s] = acc
        return psi

    def impulse_response(self, h: int = 10, orthogonalized: bool = True) -> np.ndarray:
        """Impulse response functions over horizons ``0..h``.

        Returns an array of shape ``(h+1, k, k)`` where entry ``[s, i, j]`` is
        the response of variable ``i`` at horizon ``s`` to a shock in ``j``.
        With ``orthogonalized=True`` the shocks are made contemporaneously
        uncorrelated via a Cholesky factor of ``Sigma_u`` (recursive ordering
        following the column order supplied to :meth:`fit`).
        """
        if self.results_ is None:
            raise RuntimeError("Call fit() before impulse_response().")
        psi = self.ma_representation(h + 1)
        if not orthogonalized:
            return psi
        P = np.linalg.cholesky(self.results_.sigma_u)
        return np.stack([psi[s] @ P for s in range(h + 1)])

    def simulate(self, n: int, burn: int = 100, seed: int | None = None) -> np.ndarray:
        """Simulate a VAR path of length ``n`` from the fitted parameters."""
        if self.results_ is None:
            raise RuntimeError("Call fit() before simulate().")
        r = self.results_
        rng = np.random.default_rng(seed)
        total = n + burn
        k, p = r.k, self.p
        u = rng.multivariate_normal(np.zeros(k), r.sigma_u, size=total)
        y = np.zeros((total, k))
        for t in range(total):
            val = r.const.copy()
            for i in range(1, p + 1):
                if t - i >= 0:
                    val = val + r.coefs[i - 1] @ y[t - i]
            y[t] = val + u[t]
        return y[burn:]
