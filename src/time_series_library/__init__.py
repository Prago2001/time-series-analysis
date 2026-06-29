"""time-series-library: foundational time-series models.

A small, dependency-light library (NumPy / pandas / SciPy) implementing the
core building blocks of applied time-series econometrics:

* :class:`~time_series_library.arma.ARMA`  — ARMA(p, q) for the conditional mean
* :class:`~time_series_library.garch.GARCH` — GARCH(p, q) for conditional variance
* :class:`~time_series_library.var.VAR`    — Vector autoregression with IRFs

plus diagnostics (ACF/PACF, Ljung-Box) and accuracy metrics.
"""

from __future__ import annotations

from . import utils
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
    "utils",
    "acf",
    "pacf",
    "ljung_box",
    "log_returns",
    "rmse",
    "mae",
    "mape",
]
