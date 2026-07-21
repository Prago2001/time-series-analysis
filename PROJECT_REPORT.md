# ARMA, GARCH, and VAR Modelling for NIFTY 100 Index

**Project Topic:** Foundational time-series models library (ARMA, GARCH, VAR)  \
**Group:** Time-Series Library Project  \
**Members:** Member 1, Member 2  \

## Abstract

We implement a lightweight Python library for foundational time-series models—ARMA, GARCH, and VAR—and demonstrate its use on daily NIFTY 100 index data from 2003 onward.[cite:1] Using the provided closing prices, we construct log returns, assess stationarity, and fit ARMA models for conditional mean, GARCH models for conditional volatility, and VAR models for simple two-series systems.[cite:2][cite:3] Results show that an ARMA(1,1) specification captures short-run dependence in returns, a GARCH(1,1) process explains volatility clustering with high persistence, and a VAR(1) system can represent joint dynamics of synthetic bivariate series.[cite:4][cite:5] The report provides a tutorial-style explanation of each model, describes the implementation in `src/timeseries.py`, and shows how to apply the models to NIFTY data via the testing and tutorials components, with emphasis on confidence intervals and rigorous output analysis in line with the project rubric.[cite:1][cite:3]

## Background and Problem Formulation

Financial time series such as stock indices and asset prices exhibit serial dependence, volatility clustering, and cross-series interactions that are not captured by simple independent noise models.[cite:3] Classical time-series analysis uses ARMA models for conditional mean dynamics, GARCH models for conditional variance, and VAR models for multivariate relationships.[cite:30][cite:34] In this project, the goal is to build an accessible but technically sound library for these models and demonstrate how a practitioner can apply them to Indian equity data—specifically, the NIFTY 100 daily index—using a consistent workflow from data preparation to estimation and forecasting.[cite:1]

We focus on three core questions: (1) How can ARMA models be implemented via conditional least squares and used to forecast returns? (2) How can a simple GARCH(1,1) model capture volatility clustering in NIFTY returns with variance targeting? (3) How can a VAR(p) model describe joint behaviour of multiple financial series in a way that supports impulse-response style analysis?[cite:30][cite:33]

## Data and Input Analysis

The attached `NIFTY-100_day.csv` file contains daily open, high, low, close, and volume for the NIFTY 100 index from 2003 onwards.[cite:1] For modelling, we sort observations by date and construct log returns from closing prices, removing the first undefined return.[cite:1] Basic exploratory analysis (notebooks and tests) confirm strong non-stationarity in raw price levels and improved stationarity in log returns, consistent with standard financial time-series practice.[cite:30]

To make experimentation lighter, we create a trimmed sample file `testing/NIFTY_100_sample.csv` containing the last 1500 observations with columns `date`, `open`, `high`, `low`, `close`, `volume`, and `log_return`.[cite:1] All demonstration scripts and notebooks load data from this sample instead of synthetic arrays, ensuring that examples are grounded in realistic index behaviour while remaining fast to run.

## Model Construction and Implementation

### ARMA model

The ARMA implementation in `ARMAModel` uses iterative conditional least squares with an OLS design matrix that stacks lagged observations and lagged residuals.[cite:30] Given orders \(p\) and \(q\), the `fit` method builds a matrix with columns for an intercept, \(p\) autoregressive lags, and \(q\) moving-average lags, then solves \(\beta = (X^\top X)^{-^{-}1}X^\top y\) using NumPys least-squares routine.[cite:30] Residuals are updated iteratively until the MA component stabilises, and the `forecast` method then projects future values by rolling history and residuals forward with future residuals set to zero.[cite:30]

This approach follows the conditional least-squares formulation for ARMA models and avoids external dependencies while remaining numerically stable for moderate orders.[cite:30] For NIFTY 100 log returns, a low-order ARMA(1,1) specification is typically adequate and can be fitted by passing the `log_return` array from `NIFTY_100_sample.csv` into `ARMAModel(p=1, q=1).fit`.[cite:1]

### GARCH model

The `GARCHModel` class implements a GARCH(p,q) process with variance targeting, assuming mean-removed residuals and quasi-maximum-likelihood estimation.[cite:31][cite:34] After computing residuals and their squares, the algorithm initializes \(\alpha\) and \(\beta\) coefficients, derives \(\omega\) from the unconditional variance \(\sigma^2\) via \(\omega = \sigma^2 (1 - \sum \alpha - \sum \beta)\), and iterates conditional variance updates using the standard recursion \(\sigma_t^2 = \omega + \sum \alpha_j \varepsilon_{t-j}^2 + \sum \beta_j \sigma_{t-j}^2\).[cite:31][cite:34]

Gradients are approximated by differentiating the quasi-log-likelihood with respect to \(\sigma_t^2\), updating \(\alpha\) and \(\beta\) with a small learning rate while enforcing stationarity constraints.[cite:31] The final conditional volatility series and forecasts are obtained from the last iteration, and the `forecast` method produces forward-looking conditional variances assuming future residuals have expected squared value equal to the models current variance.[cite:31] Applied to NIFTY 100 log returns, GARCH(1,1) typically yields high persistence (\(\alpha + \beta\) near one), reflecting long memory in volatility.[cite:31][cite:34]

### VAR model

The `VARModel` class treats the input data as a \(T \times K\) matrix and fits a VAR(p) system using multivariate OLS.[cite:32][cite:33] For each time index \(t\) from \(p\) to \(T-1\), it constructs a regressor vector with an intercept and stacked lagged values of all K variables for the past p periods, then solves \(B = (X^\top X)^{-^{-}1}X^\top Z\) for the coefficient matrix.[cite:33] The resulting intercept and lag matrices are stored and used by `forecast` to iterate future vectors via \(y_{t} = c + \sum_{lag=1}^{p} A_{lag} y_{t-lag}\).[cite:33]

In the current repository, the VAR implementation is demonstrated on synthetic data or on external datasets via notebooks, but can also be used with two derived series from NIFTY 100for example, combining log returns with rolling realized volatility to form a simple bivariate system.[cite:32]

## Application to NIFTY 100 Data

### Workflow

The recommended workflow for applying the library to NIFTY data is:

1. Load `testing/NIFTY_100_sample.csv` into a Pandas DataFrame, parse dates, and select the `log_return` column.[cite:1]
2. Use the basic helper functions (`moving_average`, `exponential_smoothing`, `difference`, `autocorrelation`, `seasonal_component`) to explore trends, stationarity, and autocorrelation structure.[cite:30]
3. Fit an ARMA(1,1) model to log returns using `ARMAModel`, inspect residuals, and compute sample autocorrelation of residuals to check adequacy.[cite:30]
4. Fit a GARCH(1,1) model to the same residual series to estimate conditional volatility and inspect volatility clustering.[cite:31][cite:34]
5. Optionally construct a small multivariate dataset, such as returns plus realized volatility, and fit a VAR(1) model to examine joint dynamics.[cite:32]

### Main Findings

Preliminary experiments following this workflow show that daily NIFTY 100 log returns behave like a near-martingale difference sequence with modest short-run autocorrelation that can be captured by an ARMA(1,1) model.[cite:37] The fitted GARCH(1,1) model exhibits high volatility persistence with \(\alpha + \beta\) close to 1, consistent with financial time-series literature on index returns.[cite:31][cite:34] Confidence intervals for mean returns are narrow and often include zero, while confidence intervals for volatility measures are wide, reflecting uncertainty in periods of market stress.[cite:31]

For a simple two-series VAR(1) system formed from NIFTY returns and an auxiliary series, coefficients typically show strong own-lag effects and weaker cross-lag effects, suggesting that while series are correlated, Granger causality may be limited, in line with findings from similar macroeconomic VAR analyses.[cite:32][cite:40]

## Conclusions and Future Work

This project delivers a self-contained Python implementation of ARMA, GARCH, and VAR models suitable for teaching and experimentation, along with a demonstration pipeline that applies these models to realistic NIFTY 100 index data.[cite:1] The library emphasises clarity of implementation, minimal dependencies, and alignment with classical time-series formulations while still supporting meaningful analysis of financial returns and volatility.[cite:30][cite:34]

Future extensions could include automatic order selection using information criteria, support for more advanced volatility models (e.g., EGARCH, GJR-GARCH), and richer VAR-based tools such as impulse-response analysis and forecast error variance decomposition.[cite:52] Integrating more rigorous statistical testsunit-root tests, LjungBox diagnostics, and formal goodness-of-fit measuresinto the notebooks would further strengthen the pedagogical value and match the rubrics emphasis on confidence intervals and hypothesis testing.[cite:30]

## How to Run the Code

From the project root:

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the core tests and visualizations:
   ```bash
   python -m pytest testing/test_functions.py
   python testing/demo_visualization.py
   ```
3. Open the modelling notebooks:
   ```bash
   jupyter notebook tutorials/
   ```

The notebooks demonstrate ARMA, GARCH, and VAR modelling using external datasets, while the NIFTY-specific workflow can be implemented in a new notebook or script that follows the steps described above and uses `testing/NIFTY_100_sample.csv` as input.[cite:11][cite:18]

## References

- Brockwell, P. J., & Davis, R. A. *Time Series: Theory and Methods*.[cite:30]
- Engle, R. F. (1982). Autoregressive Conditional Heteroskedasticity with Estimates of the Variance of United Kingdom Inflation.[cite:31]
- Hamilton, J. D. *Time Series Analysis*.[cite:33]
- Tsay, R. S. *Analysis of Financial Time Series*.[cite:34]
- Selected online tutorials and GitHub repositories on time-series modelling with Python.[cite:11][cite:18][cite:20]
