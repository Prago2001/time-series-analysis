# Time-Series Analysis Library: ARMA, GARCH, VAR

This repository provides a small, teaching-oriented Python library for foundational time-series models (ARMA, GARCH, VAR), together with tutorial notebooks and testing scripts.[cite:11][cite:18] The project is designed for graduate-level coursework in simulation and time-series analysis.

## Features

- Core time-series utilities: moving average, exponential smoothing, differencing, autocorrelation, decomposition.
- ARMA model for conditional mean dynamics.
- GARCH model for conditional volatility and volatility clustering.
- VAR model for multivariate time-series relationships.
- Jupyter notebooks that walk through each technique step by step.
- Example workflow for NIFTY 100 index data using the full daily dataset with an added `log_return` column.

## Repository Layout

- `src/timeseries.py`: Core library implementations of utilities and models.
- `tutorials/`: Markdown and notebook tutorials on moving averages, smoothing, differencing, autocorrelation, ARMA, GARCH, and VAR.
- `testing/`: Test suite, visualization scripts, and NIFTY-based data file `NIFTY_100_enriched.csv` derived from the full `NIFTY-100_day.csv`.

## Installation

```bash
pip install -r requirements.txt
```

You can then run tests and visualizations:

```bash
python -m pytest testing/test_functions.py
python testing/demo_visualization.py
```

## NIFTY 100 Workflow (High Level)

1. Load `testing/NIFTY_100_enriched.csv` with Pandas and parse the `date` column.
2. Use the `log_return` series for modelling; price levels are non-stationary.
3. Fit ARMA(1,1) on log returns using `ARMAModel`.
4. Fit GARCH(1,1) on ARMA residuals using `GARCHModel` to capture volatility clustering.
5. Optionally build a small multivariate dataset and fit a VAR model with `VARModel`.

See `PROJECT_REPORT.md` and the companion notebook in `tutorials/` for a fully documented workflow aligned with the course rubric.
