# GARCH Models — Modelling Time-Varying Volatility

> Tutorial for `time_series_library.GARCH`. Prerequisites: the ARMA tutorial and
> the notion of conditional variance.

## 1. The stylised facts of returns

Plot daily returns of any Indian stock and three empirical regularities jump
out — the **stylised facts** that GARCH was designed to capture:

1. **Volatility clustering** — "large changes tend to be followed by large
   changes, of either sign" (Mandelbrot, 1963). Calm and turbulent regimes
   persist.
2. **Fat tails (leptokurtosis)** — extreme moves are far more frequent than a
   normal distribution predicts.
3. **Near-zero autocorrelation in returns, strong autocorrelation in squared
   returns** — the *level* is unpredictable but the *magnitude* is not.

ARMA assumes constant variance and cannot reproduce any of these. We need a
model for the **conditional variance** $\sigma_t^2 = \mathrm{Var}(r_t \mid
\mathcal{F}_{t-1})$.

## 2. From ARCH to GARCH

Write returns as a constant mean plus a shock whose scale varies in time:

$$
r_t = \mu + \varepsilon_t,\qquad
\varepsilon_t = \sigma_t z_t,\qquad
z_t \stackrel{iid}{\sim}\mathcal{N}(0,1).
$$

Engle's **ARCH(q)** (1982) lets variance depend on recent squared shocks:
$\sigma_t^2 = \omega + \sum_{i=1}^q \alpha_i \varepsilon_{t-i}^2$. In practice
many lags are needed. Bollerslev's **GARCH(p, q)** (1986) adds lagged variances,
giving a parsimonious, ARMA-like recursion for $\sigma_t^2$:

$$
\boxed{\;\sigma_t^2 = \omega
       + \sum_{i=1}^{q}\alpha_i\,\varepsilon_{t-i}^2
       + \sum_{j=1}^{p}\beta_j\,\sigma_{t-j}^2\;}
$$

The ubiquitous **GARCH(1,1)** is
$\sigma_t^2 = \omega + \alpha\varepsilon_{t-1}^2 + \beta\sigma_{t-1}^2$.

> **Notation.** Here `p` is the GARCH (lagged-variance) order and `q` the ARCH
> (lagged-shock) order, matching the `GARCH(p, q)` convention. `GARCH(p=0)`
> reduces to `ARCH(q)`.

### Parameter constraints and persistence

* **Positivity** of variance: $\omega>0$, $\alpha_i\ge 0$, $\beta_j\ge 0$.
* **Covariance stationarity**: the **persistence**
  $\bar\alpha+\bar\beta = \sum_i\alpha_i+\sum_j\beta_j < 1$.

Under stationarity the **unconditional (long-run) variance** is

$$
\bar\sigma^2 = \frac{\omega}{1-\bar\alpha-\bar\beta}.
$$

Persistence near 1 (common for equities, ~0.95–0.99) means shocks to volatility
decay slowly — turbulent periods are long-lived. The library exposes both:

```python
res.persistence          # alpha + beta
res.unconditional_vol    # sqrt(omega / (1 - persistence))
```

## 3. Why GARCH produces fat tails

Even with Gaussian $z_t$, the *unconditional* distribution of $\varepsilon_t$ is
a variance mixture of normals and therefore **leptokurtic** — GARCH generates
fat tails endogenously from volatility variation, matching stylised fact 2.

## 4. Estimation by (quasi-)maximum likelihood

Given parameters, run the variance recursion forward from a variance-targeted
starting value to obtain $\{\sigma_t^2\}$, then evaluate the Gaussian
conditional log-likelihood

$$
\ell = -\frac12\sum_{t}\left[\log(2\pi) + \log\sigma_t^2
        + \frac{\varepsilon_t^2}{\sigma_t^2}\right].
$$

`time_series_library` maximises this with L-BFGS-B subject to the positivity and
stationarity constraints. Because the recursion uses *past* information only,
the likelihood factorises cleanly. Even when $z_t$ is non-normal, maximising the
Gaussian likelihood is a **consistent quasi-MLE (QMLE)**.

```python
from time_series_library import GARCH, utils
from tests import data  # sample dataset for testing
r = utils.log_returns(data.load_nse_prices("RELIANCE").squeeze())
res = GARCH(p=1, q=1).fit(r)
print(res.summary())
res.conditional_vol      # the filtered volatility path sigma_t
```

## 5. Workflow

1. **Fit the mean** (a constant, or an ARMA) and take residuals $\varepsilon_t$.
2. **Test for ARCH effects**: Ljung–Box on **squared** residuals — significant
   autocorrelation signals time-varying volatility.
   ```python
   utils.ljung_box(res.resid**2, lags=10)   # before: significant
   ```
3. **Fit GARCH**, then re-test the **standardised** residuals
   $\hat z_t=\varepsilon_t/\hat\sigma_t$ and their squares — both should look
   like white noise if the variance dynamics are adequately captured.
   ```python
   z = res.resid / res.conditional_vol
   utils.ljung_box(z**2, lags=10)           # after: insignificant
   ```

## 6. Volatility forecasting

The multi-step forecast uses $\mathbb{E}[\varepsilon_{t+k}^2\mid\mathcal F_t]
= \sigma_{t+k}^2$, giving a recursion that **mean-reverts to the unconditional
variance** at rate $(\bar\alpha+\bar\beta)^h$:

$$
\mathbb{E}[\sigma_{t+h}^2\mid\mathcal F_t] = \bar\sigma^2
   + (\bar\alpha+\bar\beta)^{\,h-1}\big(\sigma_{t+1}^2 - \bar\sigma^2\big).
$$

```python
fc = res.forecast(h=20)
fc["volatility"]         # term structure of volatility forecasts
```

This **volatility term structure** is exactly what is needed to price options,
size positions and set risk limits.

## 7. Typical uses on Indian-market data

* **Risk management / Value-at-Risk**: a one-day 99% VaR is
  $\mathrm{VaR} = -(\mu + z_{0.01}\,\hat\sigma_{t+1})$ using the GARCH
  one-step volatility — far more responsive than a rolling standard deviation.
* **Volatility forecasting** for NIFTY/stock options around events (Budget,
  RBI policy, earnings) where volatility spikes and decays.
* **Portfolio sizing**: scale exposure inversely to forecast volatility
  (volatility targeting).

## 8. Extensions (beyond this library's basic scope)

Symmetric GARCH ignores the **leverage effect** — negative shocks raise
volatility more than positive ones. Asymmetric variants (**EGARCH**,
**GJR-GARCH**) and heavy-tailed innovations (Student-$t$) address this and are
natural next steps once the GARCH(1,1) baseline is understood.
