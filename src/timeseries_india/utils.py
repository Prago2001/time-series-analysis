"""Common time-series utilities: transforms, diagnostics and accuracy metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ArrayLike = np.ndarray | pd.Series | list


def to_1d_array(x: ArrayLike) -> np.ndarray:
    """Coerce a series/array/list into a contiguous 1-D float array."""
    if isinstance(x, pd.Series):
        x = x.to_numpy()
    arr = np.asarray(x, dtype=float).ravel()
    if arr.ndim != 1:
        raise ValueError("Expected a 1-dimensional series.")
    return arr


def log_returns(prices: ArrayLike) -> np.ndarray:
    """Continuously-compounded (log) returns of a price series."""
    p = to_1d_array(prices)
    if np.any(p <= 0):
        raise ValueError("Prices must be strictly positive to take log returns.")
    return np.diff(np.log(p))


def simple_returns(prices: ArrayLike) -> np.ndarray:
    """Simple (arithmetic) returns of a price series."""
    p = to_1d_array(prices)
    return p[1:] / p[:-1] - 1.0


def difference(x: ArrayLike, d: int = 1) -> np.ndarray:
    """Apply ``d`` successive first-differences."""
    arr = to_1d_array(x)
    for _ in range(d):
        arr = np.diff(arr)
    return arr


def acf(x: ArrayLike, nlags: int = 20) -> np.ndarray:
    """Sample autocorrelation function for lags ``0..nlags`` (lag 0 == 1)."""
    arr = to_1d_array(x)
    arr = arr - arr.mean()
    n = arr.size
    denom = np.dot(arr, arr)
    out = np.empty(nlags + 1)
    for k in range(nlags + 1):
        out[k] = np.dot(arr[: n - k], arr[k:]) / denom
    return out


def pacf(x: ArrayLike, nlags: int = 20) -> np.ndarray:
    """Partial autocorrelation via the Durbin-Levinson recursion."""
    r = acf(x, nlags)
    phi = np.zeros((nlags + 1, nlags + 1))
    out = np.empty(nlags + 1)
    out[0] = 1.0
    if nlags >= 1:
        phi[1, 1] = r[1]
        out[1] = r[1]
    for k in range(2, nlags + 1):
        num = r[k] - np.dot(phi[k - 1, 1:k], r[1:k][::-1])
        den = 1.0 - np.dot(phi[k - 1, 1:k], r[1:k])
        phi[k, k] = num / den
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
        out[k] = phi[k, k]
    return out


@dataclass
class LjungBoxResult:
    lags: np.ndarray
    statistic: np.ndarray
    pvalue: np.ndarray

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        rows = "\n".join(
            f"  lag {int(l):>3}: Q={q:10.3f}  p={p:.4f}"
            for l, q, p in zip(self.lags, self.statistic, self.pvalue)
        )
        return f"LjungBoxResult(\n{rows}\n)"


def ljung_box(x: ArrayLike, lags: int = 10, dof: int = 0) -> LjungBoxResult:
    """Ljung-Box portmanteau test for autocorrelation up to ``lags``.

    ``dof`` is the number of estimated model parameters to subtract from the
    chi-squared degrees of freedom (e.g. ``p+q`` for an ARMA residual check).
    """
    from scipy import stats

    arr = to_1d_array(x)
    n = arr.size
    r = acf(arr, lags)[1:]
    ks = np.arange(1, lags + 1)
    q = n * (n + 2) * np.cumsum(r**2 / (n - ks))
    df = np.maximum(ks - dof, 1)
    p = stats.chi2.sf(q, df)
    return LjungBoxResult(ks, q, p)


# --------------------------------------------------------------------------- #
# Forecast-accuracy metrics
# --------------------------------------------------------------------------- #
def rmse(actual: ArrayLike, predicted: ArrayLike) -> float:
    a, f = to_1d_array(actual), to_1d_array(predicted)
    return float(np.sqrt(np.mean((a - f) ** 2)))


def mae(actual: ArrayLike, predicted: ArrayLike) -> float:
    a, f = to_1d_array(actual), to_1d_array(predicted)
    return float(np.mean(np.abs(a - f)))


def mape(actual: ArrayLike, predicted: ArrayLike) -> float:
    a, f = to_1d_array(actual), to_1d_array(predicted)
    mask = a != 0
    return float(np.mean(np.abs((a[mask] - f[mask]) / a[mask])) * 100.0)


def train_test_split(x: ArrayLike, test_size: float | int = 0.2):
    """Chronological split (no shuffling) into train/test segments."""
    arr = to_1d_array(x) if not isinstance(x, pd.Series) else x
    n = len(arr)
    n_test = test_size if isinstance(test_size, int) else int(round(n * test_size))
    n_test = max(1, min(n_test, n - 1))
    return arr[: n - n_test], arr[n - n_test :]
