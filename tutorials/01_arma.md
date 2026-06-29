# ARMA Models — Modelling the Conditional Mean

> Tutorial for `time_series_library.ARMA`. Prerequisites: basic probability,
> linear algebra and the idea of a stochastic process.

## 1. Motivation

Let $\{x_t\}$ be a time series — daily log-returns of `RELIANCE`, a monthly
mutual-fund NAV growth rate, etc. The defining feature of a *time* series is
**serial dependence**: $x_t$ is correlated with its own past. Autoregressive
moving-average (ARMA) models are the workhorse linear models for capturing that
dependence in the **conditional mean** $\mathbb{E}[x_t \mid \mathcal{F}_{t-1}]$,
where $\mathcal{F}_{t-1}$ is the information set up to time $t-1$.

## 2. The model

A zero-mean process $\{ \tilde{x}_t = x_t - \mu \}$ is **ARMA(p, q)** if

$$
\tilde{x}_t = \underbrace{\sum_{i=1}^{p} \phi_i\, \tilde{x}_{t-i}}_{\text{AR part}}
            + \varepsilon_t
            + \underbrace{\sum_{j=1}^{q} \theta_j\, \varepsilon_{t-j}}_{\text{MA part}},
\qquad \varepsilon_t \sim \mathrm{WN}(0,\sigma^2).
$$

Using the lag operator $L^k x_t = x_{t-k}$ and the polynomials
$\phi(L) = 1 - \phi_1 L - \dots - \phi_p L^p$ and
$\theta(L) = 1 + \theta_1 L + \dots + \theta_q L^q$, the model is compactly

$$
\phi(L)\, \tilde{x}_t = \theta(L)\, \varepsilon_t .
$$

* **AR(p)** ($q=0$): today is a regression on its own recent past. Shocks have
  an *infinitely-lived but geometrically decaying* effect.
* **MA(q)** ($p=0$): today is a moving average of the last $q$ shocks. Shocks
  have a *finite memory* of exactly $q$ periods.

### Stationarity and invertibility

* **Causal/stationary** $\iff$ all roots of $\phi(z)=0$ lie **outside** the unit
  circle. Then the process has a convergent $\mathrm{MA}(\infty)$ form
  $\tilde{x}_t = \psi(L)\varepsilon_t$ with $\psi(L)=\theta(L)/\phi(L)$.
* **Invertible** $\iff$ all roots of $\theta(z)=0$ lie outside the unit circle.
  Then shocks can be recovered from observables,
  $\varepsilon_t = \pi(L)\tilde{x}_t$ — the requirement that makes the
  parameters identifiable from data.

## 3. Identification: ACF and PACF

Before fitting, the classical **Box–Jenkins** workflow inspects two functions:

| | AR(p) | MA(q) | ARMA(p,q) |
|---|---|---|---|
| **ACF** $\rho_k$ | tails off (decay) | **cuts off** after lag $q$ | tails off |
| **PACF** $\alpha_k$ | **cuts off** after lag $p$ | tails off | tails off |

```python
from time_series_library import utils
from tests import data  # sample dataset for testing
r = utils.log_returns(data.load_nse_prices("NIFTY50").squeeze())
utils.acf(r, 20)    # autocorrelations
utils.pacf(r, 20)   # partial autocorrelations
```

A 95% white-noise band is $\pm 1.96/\sqrt{n}$; spikes outside it are candidate
orders.

## 4. Estimation

`time_series_library` estimates parameters by **conditional maximum likelihood**,
equivalently **conditional sum of squares (CSS)**. Conditioning on the first $p$
observations and on $\varepsilon_t = 0$ for $t \le 0$, the residuals are computed
by the recursion

$$
\varepsilon_t = \tilde{x}_t - \sum_{i=1}^p \phi_i \tilde{x}_{t-i}
                            - \sum_{j=1}^q \theta_j \varepsilon_{t-j},
$$

and, assuming Gaussian innovations, the log-likelihood is

$$
\ell(\mu,\phi,\theta,\sigma^2) =
-\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_t \varepsilon_t^2 .
$$

Concentrating out $\sigma^2 = \tfrac1n\sum_t \varepsilon_t^2$ leaves a smooth
objective in $(\mu,\phi,\theta)$ that we minimise with L-BFGS-B (coefficients
box-constrained for numerical stability).

```python
from time_series_library import ARMA
res = ARMA(p=2, q=1).fit(r)
print(res.summary())          # coefficients, log-lik, AIC, BIC
```

### Model selection

Compare candidate orders with information criteria, which trade fit against
complexity ($k$ = number of parameters):

$$
\mathrm{AIC} = 2k - 2\ell, \qquad \mathrm{BIC} = k\log n - 2\ell .
$$

BIC penalises complexity more heavily and is consistent for the true order;
AIC is better for predictive performance. Pick the order minimising the
criterion, then **check residuals**.

## 5. Diagnostics

A well-specified model leaves residuals indistinguishable from white noise.
The **Ljung–Box** portmanteau test pools the first $m$ residual
autocorrelations:

$$
Q(m) = n(n+2)\sum_{k=1}^{m} \frac{\hat\rho_k^2}{n-k}
\ \sim\ \chi^2_{m - p - q}\quad\text{under }H_0.
$$

```python
utils.ljung_box(res.resid, lags=10, dof=res.p + res.q)
```

Large $p$-values ⇒ no remaining autocorrelation ⇒ the mean dynamics are
adequately captured.

## 6. Forecasting

Point forecasts use the recursion with future shocks set to their mean (zero):
$\hat{x}_{t+h} = \mathbb{E}[x_{t+h}\mid\mathcal{F}_t]$. The $h$-step forecast
error variance follows from the $\mathrm{MA}(\infty)$ weights $\psi_j$:

$$
\mathrm{Var}(x_{t+h}-\hat{x}_{t+h}) = \sigma^2\sum_{j=0}^{h-1}\psi_j^2 ,
$$

which is **non-decreasing in $h$** and converges to the unconditional variance.
A $(1-\alpha)$ Gaussian prediction interval is
$\hat{x}_{t+h} \pm z_{1-\alpha/2}\,\mathrm{se}_h$.

```python
fc = res.forecast(h=10, alpha=0.05)
fc["mean"], fc["lower"], fc["upper"]
```

## 7. Typical uses on Indian-market data

* **Return predictability**: ARMA on NIFTY/stock log-returns. Equity returns are
  famously close to white noise, so expect small coefficients — a useful
  reality check and a benchmark for richer models.
* **NAV growth / macro series**: monthly mutual-fund NAV growth or inflation
  often show genuine AR structure that ARMA captures well.
* **Pre-whitening**: fit ARMA to the mean, then model the *residuals'* variance
  with GARCH (next tutorial). Returns are typically mean-unpredictable but
  variance-predictable.

## 8. Limitations

ARMA assumes **linearity**, **constant variance** (homoskedasticity) and
**stationarity**. Financial returns violate the constant-variance assumption
(volatility clusters) — motivating GARCH — and price *levels* are non-stationary,
so always model **returns/differences**, not raw prices.
