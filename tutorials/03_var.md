# VAR Models — Multivariate Dynamics and Impulse Responses

> Tutorial for `timeseries_india.VAR`. Prerequisites: the ARMA tutorial and
> comfort with matrix algebra.

## 1. Motivation

Markets do not move in isolation. `RELIANCE`, `TCS` and `HDFCBANK` co-move with
each other and with the `NIFTY50` index; a shock to one propagates to the
others. A **Vector Autoregression (VAR)** generalises the AR model to a *vector*
$y_t \in \mathbb{R}^k$ of variables, letting each variable depend on the lagged
values of **all** variables. VARs, popularised by Sims (1980), are the standard
tool for multivariate forecasting and for tracing how shocks propagate through a
system.

## 2. The model

A **VAR(p)** is

$$
y_t = c + A_1 y_{t-1} + A_2 y_{t-2} + \dots + A_p y_{t-p} + u_t,
\qquad u_t \sim (0, \Sigma_u),
$$

where each $A_i$ is a $k\times k$ coefficient matrix, $c\in\mathbb R^k$ is an
intercept and $\Sigma_u$ is the (generally non-diagonal) covariance of the
reduced-form innovations. Off-diagonal entries of $A_i$ encode **cross-variable
lagged effects**; off-diagonal entries of $\Sigma_u$ encode
**contemporaneous** correlation.

### Stability

Stack the system into its $kp\times kp$ **companion matrix** $\mathbf{A}$. The
VAR is **stable (stationary)** iff all eigenvalues of $\mathbf{A}$ lie inside the
unit circle, $|\lambda_i|<1$. The library checks this directly:

```python
res.is_stable()
```

## 3. Estimation by OLS

A key practical fact: because **every equation has identical regressors**
(a constant plus the same stacked lags), the system GLS estimator collapses to
**equation-by-equation OLS**, which is also the Gaussian MLE. Stacking
$Z_t = [1, y_{t-1}^\top, \dots, y_{t-p}^\top]^\top$,

$$
\hat B = (Z^\top Z)^{-1} Z^\top Y, \qquad
\hat\Sigma_u = \frac{1}{T-kp-1}\,\hat U^\top \hat U .
$$

```python
from timeseries_india import VAR, utils, data
prices = data.load_nse_prices(["NIFTY50", "RELIANCE", "TCS"])
rets = prices.apply(utils.log_returns, axis=0, result_type="expand").T  # or build a returns frame
res = VAR(p=2).fit(rets)
print(res.summary())
```

### Choosing the lag order

Select $p$ by minimising a multivariate information criterion based on
$\log\det\hat\Sigma_u$:

$$
\mathrm{AIC}(p) = \log\det\hat\Sigma_u + \frac{2}{T}\,(\text{\# params}),\quad
\mathrm{BIC}(p) = \log\det\hat\Sigma_u + \frac{\log T}{T}\,(\text{\# params}).
$$

```python
res.aic, res.bic
```

## 4. Forecasting

Forecasts iterate the VAR recursion forward, feeding predicted values back in:
$\hat y_{t+h} = c + \sum_{i=1}^p A_i \hat y_{t+h-i}$. The $h$-step
forecast-error covariance accumulates through the **Wold / MA($\infty$)**
representation $y_t = \mu + \sum_{s\ge0}\Psi_s u_{t-s}$:

$$
\Sigma(h) = \sum_{s=0}^{h-1} \Psi_s\, \Sigma_u\, \Psi_s^\top,
\qquad \Psi_0 = I_k,\ \ \Psi_s=\sum_{i=1}^{\min(s,p)}A_i\Psi_{s-i}.
$$

Per-variable prediction intervals use the diagonal of $\Sigma(h)$.

```python
fc = res.forecast(h=10, alpha=0.05)
fc["mean"]    # (h, k)
fc["lower"], fc["upper"]
```

## 5. Impulse Response Functions (IRFs)

The headline output of a VAR is the **IRF**: the dynamic response of every
variable to a one-time shock in one variable. From the MA representation,
$\partial y_{t+s}/\partial u_t = \Psi_s$. But reduced-form shocks $u_t$ are
**correlated** ($\Sigma_u$ non-diagonal), so "shock variable $j$ alone" is not
well defined.

### Orthogonalisation (Cholesky)

Factor $\Sigma_u = PP^\top$ with $P$ **lower-triangular** (Cholesky). Define
structural shocks $w_t = P^{-1}u_t$, which have identity covariance. The
**orthogonalised IRF** at horizon $s$ is

$$
\Theta_s = \Psi_s P .
$$

Because $P$ is lower-triangular, this imposes a **recursive identification**: the
variable ordered first can react contemporaneously to none but its own shock,
the second to the first and itself, and so on. **Ordering matters** — place the
"most exogenous" variable (e.g. the broad index) first.

```python
irf = res.impulse_response(h=20, orthogonalized=True)   # (h+1, k, k)
# irf[s, i, j] = response of variable i at horizon s to a shock in variable j
```

A unit (one-standard-deviation) shock to variable $j$ traces out
`irf[:, :, j]`; for a stable VAR these responses decay back to zero.

## 6. Typical uses on Indian-market data

* **Shock transmission**: how does a NIFTY shock propagate to individual stocks
  (and over how many days)? Read it straight off the IRF.
* **Lead–lag / spillovers**: significant off-diagonal $A_i$ reveal which series
  leads which — useful for pairs and index-vs-constituent dynamics.
* **Joint forecasting**: coherent multi-asset return/volatility forecasts that
  respect cross-correlations, for portfolio construction.
* **Macro–market links**: combine market returns with macro series (rates, FX)
  to study interactions.

## 7. Caveats

* IRFs from a Cholesky scheme are **order-dependent**; report the ordering and,
  ideally, check robustness to reorderings.
* VARs are estimated on **stationary** inputs — use returns/differences, not
  price levels. Cointegrated levels call for a VECM (a natural extension).
* Parameters proliferate as $k^2 p$; keep $k$ and $p$ modest or the model
  overfits.
