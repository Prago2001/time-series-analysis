# Time Series Analysis Tutorials

This directory contains tutorials for common time series analysis methods. Each
tutorial explains **how the method works** and includes a **runnable example**
that uses only the Python standard library, so you can try them without
installing any dependencies.

## Methods

| Tutorial | Method | Example script |
| --- | --- | --- |
| [01 Moving Average](01_moving_average.md) | Smoothing trends with a rolling mean | [examples/moving_average.py](examples/moving_average.py) |
| [02 Exponential Smoothing](02_exponential_smoothing.md) | Weighted smoothing / one-step forecasting | [examples/exponential_smoothing.py](examples/exponential_smoothing.py) |
| [03 Differencing & Stationarity](03_differencing.md) | Removing trend to make a series stationary | [examples/differencing.py](examples/differencing.py) |
| [04 Autocorrelation](04_autocorrelation.md) | Measuring how a series relates to its own past | [examples/autocorrelation.py](examples/autocorrelation.py) |
| [05 Seasonal Decomposition](05_decomposition.md) | Splitting into trend, seasonal, residual | [examples/decomposition.py](examples/decomposition.py) |

## Modelling Notebooks

Interactive Jupyter notebooks that demonstrate fitting popular time-series
models on open-source datasets while using the library's helper functions for
data exploration.

| Notebook | Model | Dataset |
| --- | --- | --- |
| [06 ARMA Model](06_ARMA_model.ipynb) | ARMA (via `statsmodels`) | Mauna Loa CO2 (`statsmodels`) |
| [07 GARCH Model](07_GARCH_model.ipynb) | GARCH (via `arch`) | S&P 500 daily returns (`arch`) |
| [08 VAR Model](08_VAR_model.ipynb) | VAR (via `statsmodels`) | US macroeconomic data (`statsmodels`) |

All methods are implemented in [src/timeseries.py](../src/timeseries.py) and the
examples import them from there.

## Running the examples

```bash
python tutorials/examples/moving_average.py
python tutorials/examples/exponential_smoothing.py
python tutorials/examples/differencing.py
python tutorials/examples/autocorrelation.py
python tutorials/examples/decomposition.py
```

## Running the notebooks

```bash
pip install -r requirements.txt
jupyter notebook tutorials/
```
