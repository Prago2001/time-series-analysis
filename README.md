# time-series-library

A compact, well-documented library of **foundational time-series models** —
`ARMA`, `GARCH` and `VAR` — implemented from first principles on top of
**NumPy / pandas / SciPy**, together with graduate-level tutorials and an
interactive Jupyter notebook. The Indian-market (NSE) data is used only
as a sample dataset for testing.

The goal is technical rigour with readable code: every model is estimated by
(quasi-)maximum likelihood or OLS, exposes forecasts with uncertainty bands, and
comes with the diagnostics needed to use it responsibly.

---

## Features

| Model | Module | Captures | Key methods |
|-------|--------|----------|-------------|
| **ARMA(p, q)** | `arma.py` | conditional mean / autocorrelation | `fit`, `forecast` (+ prediction intervals), `simulate`, AIC/BIC, `summary` |
| **GARCH(p, q)** | `garch.py` | conditional variance / volatility clustering | `fit`, `forecast` (volatility term structure), `simulate`, persistence & unconditional vol |
| **VAR(p)** | `var.py` | multivariate dynamics & spillovers | `fit`, `forecast`, `impulse_response` (orthogonalised), `ma_representation`, stability check, `simulate` |

**Diagnostics & utilities** (`utils.py`): log/simple returns, differencing,
`acf`, `pacf` (Durbin–Levinson), `ljung_box` test, RMSE/MAE/MAPE,
chronological train/test split.

**Test data** (`tests/data.py`): a **synthetic NSE dataset** (daily prices for
`NIFTY50`, `RELIANCE`, `TCS`, `HDFCBANK`, `INFY` over 2018–2023, plus a monthly
mutual-fund NAV series), used only for testing. An optional `download_nse(...)`
loader fetches live data via `yfinance`.

> **About the bundled data.** The sample series are *simulated* (no live network
> needed) but calibrated to the empirical behaviour of Indian equities: positive
> drift, fat-tailed returns and GARCH-style volatility clustering. Ticker names
> mirror NSE symbols (`RELIANCE.NS`, …) so the data is a drop-in stand-in for a
> `yfinance` download. They are **not** real historical quotes.

---

## Installation

This project uses [**uv**](https://docs.astral.sh/uv/) for environment and
dependency management.

```bash
# from the repository root
uv sync                 # create the venv and install core dependencies
```

Run anything inside the managed environment with `uv run`, e.g. `uv run python`,
`uv run pytest`, `uv run jupyter lab`.

---

## Quick start

```python
import numpy as np
import time_series_library as tsl
from time_series_library import ARMA, GARCH, VAR, utils
from tests import data  # sample dataset for testing

# 1. Load bundled NSE prices and compute daily log-returns (in %).
prices  = data.load_nse_prices(["NIFTY50", "RELIANCE", "TCS"])
returns = np.log(prices).diff().dropna() * 100

# 2. ARMA on the conditional mean of NIFTY returns.
arma = ARMA(p=1, q=1).fit(returns["NIFTY50"])
print(arma.summary())
fc = arma.forecast(h=10)            # fc["mean"], fc["lower"], fc["upper"]

# 3. GARCH on the conditional volatility of RELIANCE returns.
g = GARCH(p=1, q=1).fit(returns["RELIANCE"])
print("persistence:", g.persistence, "unconditional vol:", g.unconditional_vol)
vol = g.forecast(h=20)["volatility"]

# 4. VAR for joint dynamics + impulse responses.
var = VAR(p=2).fit(returns)
irf = var.impulse_response(h=20, orthogonalized=True)   # (h+1, k, k)
```

Using **live data** instead (requires internet):

```python
prices = data.download_nse(["RELIANCE", "TCS"], start="2020-01-01")
```

---

## Repository layout

```
src/time_series_library/
├── __init__.py        # public API
├── arma.py            # ARMA(p, q)  — conditional mean
├── garch.py           # GARCH(p, q) — conditional variance
├── var.py             # VAR(p)      — multivariate + IRFs
└── utils.py           # ACF/PACF, Ljung-Box, metrics, transforms
tutorials/
├── 01_arma.md         # ARMA: theory, identification, estimation, forecasting
├── 02_garch.md        # GARCH: stylised facts, ML estimation, volatility forecasts
└── 03_var.md          # VAR: OLS, stability, IRFs
notebooks/
└── demo.ipynb         # end-to-end walkthrough on Indian-market data
tests/
├── data.py            # sample NSE dataset loaders (+ optional yfinance)
├── datasets/          # synthetic NSE CSVs (test data)
└── test_models.py     # parameter-recovery & behavioural tests
```

---

## Tutorials

Graduate-level write-ups explaining the intuition, mathematical structure and
typical use cases of each model:

- [`tutorials/01_arma.md`](tutorials/01_arma.md) — ARMA: lag polynomials,
  stationarity/invertibility, ACF/PACF identification, conditional MLE,
  information criteria, Ljung–Box diagnostics, forecasting with intervals.
- [`tutorials/02_garch.md`](tutorials/02_garch.md) — GARCH: volatility
  clustering and fat tails, ARCH→GARCH, persistence and the unconditional
  variance, quasi-MLE, volatility forecasting and Value-at-Risk.
- [`tutorials/03_var.md`](tutorials/03_var.md) — VAR: companion form and
  stability, equation-by-equation OLS, Wold representation, orthogonalised
  impulse responses and the role of variable ordering.

## Notebook

[`notebooks/demo.ipynb`](notebooks/demo.ipynb) is a runnable, end-to-end tour:
load data → explore returns → fit ARMA (ACF/PACF, model selection, residual
checks, forecast) → fit GARCH (conditional volatility, VaR, vol term structure)
→ fit VAR (joint forecasts, impulse responses). Launch it with:

```bash
uv run jupyter lab notebooks/demo.ipynb
```

---

## Testing

```bash
uv run pytest -q
```

The suite verifies, among other things, that each estimator **recovers known
parameters** from simulated data (AR(1)/MA(1) coefficients, GARCH
persistence/positivity, VAR coefficient matrices), that **forecast uncertainty
grows with the horizon**, and that **orthogonalised IRFs respect the recursive
ordering**.

---

## Modelling notes & scope

- Always model **returns / differences** (stationary), not raw price levels.
- ARMA captures the mean; GARCH captures the variance — they are complementary
  (fit ARMA, then GARCH on the residuals).
- VAR impulse responses use a **Cholesky** identification and are therefore
  **ordering-dependent**; place the most exogenous variable first.
- This is a **foundational/basic** library by design (Gaussian innovations,
  symmetric GARCH, reduced-form VAR). Natural extensions — ARIMA/seasonal terms,
  EGARCH/GJR-GARCH and Student-t innovations, VECM/structural VARs — are noted
  in the tutorials.

## Roadmap

- [x] ARMA, GARCH, VAR with forecasting, simulation and diagnostics
- [x] Indian-market sample dataset for testing + optional live loader
- [x] Tutorials and an end-to-end notebook
- [ ] ARIMA / seasonal differencing
- [ ] Asymmetric GARCH (EGARCH, GJR) and Student-t innovations
- [ ] VECM for cointegrated levels; FEVD reporting

## License

MIT — see [LICENSE](LICENSE).
