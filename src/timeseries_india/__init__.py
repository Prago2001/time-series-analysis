"""timeseries-india: foundational time-series models for Indian-market data.

A small, dependency-light library (NumPy / pandas / SciPy) implementing the
core building blocks of applied time-series econometrics:

* :class:`~timeseries_india.arma.ARMA`  — ARMA(p, q) for the conditional mean
* :class:`~timeseries_india.garch.GARCH` — GARCH(p, q) for conditional variance
* :class:`~timeseries_india.var.VAR`    — Vector autoregression with IRFs

plus diagnostics (ACF/PACF, Ljung-Box), accuracy metrics and a bundled
synthetic NSE dataset (with an optional live ``yfinance`` loader).
"""

from __future__ import annotations

from . import data, utils
from .arma import ARMA, ARMAResults
from .garch import GARCH, GARCHResults
from .utils import acf, ljung_box, log_returns, mae, mape, pacf, rmse
from .var import VAR, VARResults

__version__ = "0.1.0"

__all__ = [
    "ARMA",
    "ARMAResults",
    "GARCH",
    "GARCHResults",
    "VAR",
    "VARResults",
    "data",
    "utils",
    "acf",
    "pacf",
    "ljung_box",
    "log_returns",
    "rmse",
    "mae",
    "mape",
]
