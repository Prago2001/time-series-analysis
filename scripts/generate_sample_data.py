"""Regenerate the bundled synthetic NSE dataset (reproducible).

Run with::

    uv run python scripts/generate_sample_data.py

The series are *simulated* (no live data required) but calibrated to the
empirical behaviour of Indian equities: positive drift, fat-tailed returns and
strong GARCH-style volatility clustering. Ticker names mirror NSE symbols so the
data is a drop-in stand-in for ``yfinance`` downloads (e.g. ``RELIANCE.NS``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20240101
OUT_DIR = Path(__file__).resolve().parents[1] / "tests" / "datasets"


def garch_returns(rng, n, mu_annual, vol_annual, alpha, beta, jump_p=0.0):
    """Simulate daily log-returns with a GARCH(1,1) variance and t-distributed shocks."""
    mu = mu_annual / 252.0
    lr_var = (vol_annual ** 2) / 252.0          # target long-run daily variance
    omega = lr_var * (1.0 - alpha - beta)
    eps = np.empty(n)
    sigma2 = np.empty(n)
    sigma2[0] = lr_var
    for t in range(n):
        if t > 0:
            sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        z = rng.standard_t(df=6) / np.sqrt(6 / (6 - 2))   # unit-variance t(6)
        eps[t] = np.sqrt(sigma2[t]) * z
        if jump_p and rng.random() < jump_p:               # occasional jumps
            eps[t] += rng.normal(0.0, 4.0 * np.sqrt(lr_var))
    return mu + eps


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    dates = pd.bdate_range("2018-01-01", "2023-12-29")  # ~NSE trading days
    n = len(dates)

    specs = {
        "NIFTY50":  dict(mu_annual=0.11, vol_annual=0.16, alpha=0.09, beta=0.88, jump_p=0.004, p0=10500),
        "RELIANCE": dict(mu_annual=0.15, vol_annual=0.26, alpha=0.10, beta=0.86, jump_p=0.006, p0=920),
        "TCS":      dict(mu_annual=0.13, vol_annual=0.22, alpha=0.08, beta=0.88, jump_p=0.005, p0=1380),
        "HDFCBANK": dict(mu_annual=0.10, vol_annual=0.24, alpha=0.11, beta=0.85, jump_p=0.006, p0=1900),
        "INFY":     dict(mu_annual=0.12, vol_annual=0.25, alpha=0.09, beta=0.87, jump_p=0.006, p0=1050),
    }

    prices = {}
    for name, spec in specs.items():
        p0 = spec.pop("p0")
        r = garch_returns(rng, n, **spec)
        prices[name] = p0 * np.exp(np.cumsum(r))

    price_df = pd.DataFrame(prices, index=dates).round(2)
    price_df.index.name = "Date"
    price_df.to_csv(OUT_DIR / "nse_prices.csv")

    # Monthly equity mutual-fund NAV: broad-market tracker with a small expense drag.
    monthly = price_df["NIFTY50"].resample("ME").last()
    fund_ret = np.log(monthly).diff().fillna(0.0).to_numpy()
    fund_ret = 0.95 * fund_ret - 0.01 / 12 + rng.normal(0, 0.01, len(fund_ret))
    nav = 100 * np.exp(np.cumsum(fund_ret))
    fund_df = pd.DataFrame({"NAV": nav.round(4)}, index=monthly.index)
    fund_df.index.name = "Date"
    fund_df.to_csv(OUT_DIR / "mutual_fund_nav.csv")

    print(f"Wrote prices {price_df.shape} and fund NAV {fund_df.shape} to {OUT_DIR}")


if __name__ == "__main__":
    main()
