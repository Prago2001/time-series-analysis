"""Tests for the timeseries_india library."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import timeseries_india as tsi
from timeseries_india import ARMA, GARCH, VAR, data, utils


# --------------------------------------------------------------------------- #
# Utilities & data
# --------------------------------------------------------------------------- #
def test_log_returns_roundtrip():
    p = np.array([100.0, 110.0, 99.0, 105.0])
    r = utils.log_returns(p)
    recon = p[0] * np.exp(np.cumsum(r))
    assert np.allclose(recon, p[1:])


def test_acf_lag0_is_one():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    a = utils.acf(x, 10)
    assert a[0] == pytest.approx(1.0)
    assert abs(a[1]) < 0.2  # white noise -> near zero


def test_pacf_ar1_cutoff():
    # AR(1) with phi=0.6: pacf[1]~0.6, pacf[2]~0
    rng = np.random.default_rng(1)
    n = 4000
    e = rng.standard_normal(n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.6 * x[t - 1] + e[t]
    pa = utils.pacf(x, 5)
    assert pa[1] == pytest.approx(0.6, abs=0.06)
    assert abs(pa[2]) < 0.06


def test_bundled_data_loads():
    df = data.load_nse_prices()
    assert df.shape[0] > 1000
    assert "NIFTY50" in df.columns
    assert (df.values > 0).all()
    nav = data.load_mutual_fund_nav()
    assert isinstance(nav, pd.Series)
    assert len(nav) > 12
    assert "RELIANCE" in data.available_symbols()


def test_ljung_box_detects_autocorrelation():
    rng = np.random.default_rng(2)
    n = 1000
    e = rng.standard_normal(n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.7 * x[t - 1] + e[t]
    lb = utils.ljung_box(x, lags=10)
    assert (lb.pvalue < 0.05).all()


# --------------------------------------------------------------------------- #
# ARMA
# --------------------------------------------------------------------------- #
def test_arma_recovers_ar1_parameter():
    rng = np.random.default_rng(3)
    n = 3000
    phi = 0.5
    e = rng.standard_normal(n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    res = ARMA(p=1, q=0).fit(x)
    assert res.ar_params[0] == pytest.approx(phi, abs=0.05)
    assert res.sigma2 == pytest.approx(1.0, abs=0.1)


def test_arma_recovers_ma1_parameter():
    rng = np.random.default_rng(4)
    n = 4000
    theta = 0.4
    e = rng.standard_normal(n)
    x = e[1:] + theta * e[:-1]
    res = ARMA(p=0, q=1).fit(x)
    assert res.ma_params[0] == pytest.approx(theta, abs=0.07)


def test_arma_forecast_shapes_and_intervals():
    x = np.random.default_rng(5).standard_normal(200)
    model = ARMA(p=1, q=1)
    model.fit(x)
    fc = model.forecast(h=8)
    for key in ("mean", "se", "lower", "upper"):
        assert fc[key].shape == (8,)
    assert np.all(fc["lower"] <= fc["mean"]) and np.all(fc["mean"] <= fc["upper"])
    # Forecast uncertainty must be non-decreasing with horizon.
    assert np.all(np.diff(fc["se"]) >= -1e-9)


def test_arma_aic_bic_finite():
    x = np.random.default_rng(6).standard_normal(300)
    res = ARMA(1, 1).fit(x)
    assert np.isfinite(res.aic) and np.isfinite(res.bic)
    assert "ARMA(1,1)" in res.summary()


# --------------------------------------------------------------------------- #
# GARCH
# --------------------------------------------------------------------------- #
def _simulate_garch(n, omega, alpha, beta, seed):
    rng = np.random.default_rng(seed)
    eps = np.zeros(n)
    s2 = np.zeros(n)
    s2[0] = omega / (1 - alpha - beta)
    for t in range(1, n):
        s2[t] = omega + alpha * eps[t - 1] ** 2 + beta * s2[t - 1]
        eps[t] = np.sqrt(s2[t]) * rng.standard_normal()
    return eps


def test_garch_fits_and_is_stationary():
    eps = _simulate_garch(3000, 0.02, 0.08, 0.9, seed=7)
    res = GARCH(p=1, q=1).fit(eps)
    assert 0 < res.persistence < 1
    assert res.omega > 0
    assert res.alpha[0] >= 0 and res.beta[0] >= 0
    assert np.all(res.conditional_vol > 0)


def test_garch_forecast_converges_to_unconditional():
    eps = _simulate_garch(3000, 0.05, 0.1, 0.85, seed=8)
    model = GARCH(1, 1)
    res = model.fit(eps)
    fc = model.forecast(h=200)
    assert fc["volatility"].shape == (200,)
    assert fc["volatility"][-1] == pytest.approx(res.unconditional_vol, rel=0.15)


def test_garch_simulate_shapes():
    eps = np.random.default_rng(9).standard_normal(500) * 0.1
    model = GARCH(1, 1)
    model.fit(eps)
    sim = model.simulate(100, seed=1)
    assert sim["returns"].shape == (100,)
    assert np.all(sim["volatility"] > 0)


# --------------------------------------------------------------------------- #
# VAR
# --------------------------------------------------------------------------- #
def _simulate_var(n=2000, seed=10):
    rng = np.random.default_rng(seed)
    A = np.array([[0.5, 0.1], [0.2, 0.3]])
    c = np.array([0.0, 0.0])
    y = np.zeros((n, 2))
    for t in range(1, n):
        y[t] = c + A @ y[t - 1] + rng.standard_normal(2) * 0.5
    return y, A


def test_var_recovers_coefficients():
    y, A = _simulate_var()
    res = VAR(p=1).fit(pd.DataFrame(y, columns=["a", "b"]))
    assert np.allclose(res.coefs[0], A, atol=0.06)
    assert res.is_stable()
    assert res.names == ["a", "b"]


def test_var_forecast_intervals_widen():
    y, _ = _simulate_var()
    model = VAR(p=1)
    model.fit(y)
    fc = model.forecast(h=10)
    assert fc["mean"].shape == (10, 2)
    assert np.all(np.diff(fc["se"][:, 0]) >= -1e-9)
    assert np.all(fc["lower"] <= fc["mean"])


def test_var_impulse_response_shape_and_impact():
    y, _ = _simulate_var()
    model = VAR(p=1)
    model.fit(y)
    irf = model.impulse_response(h=12, orthogonalized=True)
    assert irf.shape == (13, 2, 2)
    # Orthogonalized impact (horizon 0) is lower-triangular by construction.
    assert irf[0][0, 1] == pytest.approx(0.0, abs=1e-9)


def test_var_simulate_shape():
    y, _ = _simulate_var()
    model = VAR(p=2)
    model.fit(y)
    sim = model.simulate(50, seed=3)
    assert sim.shape == (50, 2)


def test_package_exports():
    assert hasattr(tsi, "ARMA")
    assert tsi.__version__
