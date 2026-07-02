"""Simple time series analysis methods using numpy for efficient computation."""

import numpy as np


def moving_average(data, window):
    """Simple moving average; first window-1 entries are None.

    Parameters
    ----------
    data : list or array-like
        Input time series values.
    window : int
        Number of periods for the moving average.

    Returns
    -------
    list
        Moving average values (None for the first window-1 entries).
    """
    arr = np.asarray(data, dtype=float)
    n = len(arr)
    out = [None] * (window - 1)
    if n >= window:
        cumsum = np.cumsum(arr)
        cumsum = np.insert(cumsum, 0, 0.0)
        ma = (cumsum[window:] - cumsum[:-window]) / window
        out.extend(ma.tolist())
    return out


def exponential_smoothing(data, alpha):
    """Exponentially smoothed series.

    Parameters
    ----------
    data : list or array-like
        Input time series values.
    alpha : float
        Smoothing factor between 0 and 1.

    Returns
    -------
    list
        Smoothed values.
    """
    arr = np.asarray(data, dtype=float)
    if len(arr) == 0:
        return []
    smoothed = np.empty_like(arr)
    smoothed[0] = arr[0]
    for i in range(1, len(arr)):
        smoothed[i] = alpha * arr[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed.tolist()


def difference(data, lag=1):
    """Lag-differenced series.

    Parameters
    ----------
    data : list or array-like
        Input time series values.
    lag : int, optional
        Number of periods to lag (default 1).

    Returns
    -------
    list
        Differenced values.
    """
    arr = np.asarray(data, dtype=float)
    return (arr[lag:] - arr[:-lag]).tolist()


def autocorrelation(data, lag):
    """Autocorrelation at the given lag.

    Parameters
    ----------
    data : list or array-like
        Input time series values.
    lag : int
        Lag at which to compute autocorrelation.

    Returns
    -------
    float
        Autocorrelation coefficient.
    """
    arr = np.asarray(data, dtype=float)
    mean = np.mean(arr)
    centered = arr - mean
    denom = np.sum(centered ** 2)
    if denom == 0:
        return 0.0
    num = np.sum(centered[lag:] * centered[:-lag])
    return float(num / denom)


def seasonal_component(data, period):
    """Additive seasonal component for the given period.

    Parameters
    ----------
    data : list or array-like
        Input time series values.
    period : int
        Length of one seasonal cycle.

    Returns
    -------
    list
        Seasonal component repeated over the full series length.
    """
    arr = np.asarray(data, dtype=float)
    overall_mean = np.mean(arr)
    seasonal = np.array([np.mean(arr[p::period]) - overall_mean for p in range(period)])
    return np.tile(seasonal, len(arr) // period + 1)[:len(arr)].tolist()


# ---------------------------------------------------------------------------
# ARMA Model
# ---------------------------------------------------------------------------

class ARMAModel:
    """ARMA(p, q) model fitted with OLS (conditional least squares).

    Parameters
    ----------
    p : int
        Number of autoregressive lags.
    q : int
        Number of moving-average lags.
    """

    def __init__(self, p, q):
        self.p = p
        self.q = q
        self.intercept = 0.0
        self.ar_coeffs = np.zeros(p)
        self.ma_coeffs = np.zeros(q)
        self.residuals = None
        self._fitted = False

    def fit(self, data, max_iter=50):
        """Fit the model using iterative conditional least squares.

        Parameters
        ----------
        data : list or array-like
            Input time series (should be stationary).
        max_iter : int
            Number of iterations to refine MA residuals.

        Returns
        -------
        self
        """
        arr = np.asarray(data, dtype=float)
        n = len(arr)
        start = max(self.p, self.q)
        residuals = np.zeros(n)

        for _ in range(max_iter):
            # Build design matrix: [1, y_{t-1}, ..., y_{t-p}, e_{t-1}, ..., e_{t-q}]
            rows = n - start
            X = np.ones((rows, 1 + self.p + self.q))
            y = arr[start:]
            for i in range(rows):
                t = start + i
                for j in range(self.p):
                    X[i, 1 + j] = arr[t - 1 - j]
                for j in range(self.q):
                    X[i, 1 + self.p + j] = residuals[t - 1 - j]

            # OLS solve: beta = (X'X)^-1 X'y
            beta = np.linalg.lstsq(X, y, rcond=None)[0]

            self.intercept = beta[0]
            self.ar_coeffs = beta[1: 1 + self.p]
            self.ma_coeffs = beta[1 + self.p:]

            # Update residuals
            new_residuals = np.zeros(n)
            for t in range(start, n):
                pred = self.intercept
                for j in range(self.p):
                    pred += self.ar_coeffs[j] * arr[t - 1 - j]
                for j in range(self.q):
                    pred += self.ma_coeffs[j] * new_residuals[t - 1 - j]
                new_residuals[t] = arr[t] - pred

            if np.max(np.abs(new_residuals - residuals)) < 1e-8:
                residuals = new_residuals
                break
            residuals = new_residuals

        self.residuals = residuals
        self._data = arr
        self._fitted = True
        return self

    def forecast(self, steps=1):
        """Forecast future values.

        Parameters
        ----------
        steps : int
            Number of steps ahead.

        Returns
        -------
        list
            Forecasted values.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before forecasting.")
        history = list(self._data)
        res = list(self.residuals)
        preds = []
        for _ in range(steps):
            val = self.intercept
            for j in range(self.p):
                val += self.ar_coeffs[j] * history[-(1 + j)]
            for j in range(self.q):
                val += self.ma_coeffs[j] * res[-(1 + j)]
            preds.append(float(val))
            history.append(val)
            res.append(0.0)  # future residuals are zero
        return preds


# ---------------------------------------------------------------------------
# GARCH Model
# ---------------------------------------------------------------------------

class GARCHModel:
    """GARCH(p, q) model fitted with simple variance targeting.

    Parameters
    ----------
    p : int
        Number of lagged conditional variance terms.
    q : int
        Number of lagged squared residual terms.
    """

    def __init__(self, p=1, q=1):
        self.p = p
        self.q = q
        self.omega = 0.0
        self.alpha = np.zeros(q)
        self.beta = np.zeros(p)
        self.conditional_volatility = None
        self._fitted = False

    def fit(self, data, max_iter=100, lr=1e-5):
        """Fit the GARCH model using simple iterative estimation.

        The mean is assumed to be the sample mean. Estimation uses
        variance targeting: omega is derived from alpha, beta and
        the unconditional variance so that stationarity is enforced.

        Parameters
        ----------
        data : list or array-like
            Return series (e.g. percentage returns).
        max_iter : int
            Maximum optimisation iterations.
        lr : float
            Learning rate for gradient updates.

        Returns
        -------
        self
        """
        arr = np.asarray(data, dtype=float)
        n = len(arr)
        mu = np.mean(arr)
        eps = arr - mu  # residuals (mean-removed)
        eps2 = eps ** 2
        var_unc = np.var(arr)  # unconditional variance

        start = max(self.p, self.q)

        # Initialise with reasonable values
        alpha = np.full(self.q, 0.05 / self.q)
        beta = np.full(self.p, 0.90 / self.p)

        for _ in range(max_iter):
            # Variance targeting: omega = var_unc * (1 - sum(alpha) - sum(beta))
            ab_sum = np.sum(alpha) + np.sum(beta)
            if ab_sum >= 1.0:
                alpha *= 0.9 / (np.sum(alpha) + 1e-12) * 0.05
                beta *= 0.9 / (np.sum(beta) + 1e-12) * 0.90
                ab_sum = np.sum(alpha) + np.sum(beta)
            omega = var_unc * (1.0 - ab_sum)
            if omega < 1e-10:
                omega = 1e-10

            # Compute conditional variances
            sigma2 = np.full(n, var_unc)
            for t in range(start, n):
                s2 = omega
                for j in range(self.q):
                    s2 += alpha[j] * eps2[t - 1 - j]
                for j in range(self.p):
                    s2 += beta[j] * sigma2[t - 1 - j]
                sigma2[t] = max(s2, 1e-10)

            # Quasi-ML gradient step for alpha and beta
            grad_alpha = np.zeros(self.q)
            grad_beta = np.zeros(self.p)
            for t in range(start, n):
                s2 = sigma2[t]
                dl_ds2 = -0.5 * (1.0 / s2 - eps2[t] / (s2 * s2))
                for j in range(self.q):
                    grad_alpha[j] += dl_ds2 * eps2[t - 1 - j]
                for j in range(self.p):
                    grad_beta[j] += dl_ds2 * sigma2[t - 1 - j]

            alpha += lr * grad_alpha
            beta += lr * grad_beta
            alpha = np.clip(alpha, 1e-6, 0.5)
            beta = np.clip(beta, 1e-6, 0.999)
            if np.sum(alpha) + np.sum(beta) >= 0.999:
                scale = 0.998 / (np.sum(alpha) + np.sum(beta))
                alpha *= scale
                beta *= scale

        # Store final estimates
        ab_sum = np.sum(alpha) + np.sum(beta)
        self.omega = var_unc * (1.0 - ab_sum)
        self.alpha = alpha
        self.beta = beta

        # Final conditional volatility series
        sigma2 = np.full(n, var_unc)
        for t in range(start, n):
            s2 = self.omega
            for j in range(self.q):
                s2 += self.alpha[j] * eps2[t - 1 - j]
            for j in range(self.p):
                s2 += self.beta[j] * sigma2[t - 1 - j]
            sigma2[t] = max(s2, 1e-10)

        self.conditional_volatility = np.sqrt(sigma2)
        self._eps2 = eps2
        self._sigma2 = sigma2
        self._fitted = True
        return self

    def forecast(self, steps=1):
        """Forecast conditional variance (and volatility) ahead.

        Parameters
        ----------
        steps : int
            Number of steps ahead.

        Returns
        -------
        dict
            ``{"variance": [...], "volatility": [...]}``
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before forecasting.")
        eps2_hist = list(self._eps2)
        s2_hist = list(self._sigma2)
        var_fc = []
        for _ in range(steps):
            s2 = self.omega
            for j in range(self.q):
                s2 += self.alpha[j] * eps2_hist[-(1 + j)]
            for j in range(self.p):
                s2 += self.beta[j] * s2_hist[-(1 + j)]
            s2 = max(s2, 1e-10)
            var_fc.append(float(s2))
            # For future steps, E[eps^2] = sigma^2
            eps2_hist.append(s2)
            s2_hist.append(s2)
        return {"variance": var_fc, "volatility": [v ** 0.5 for v in var_fc]}


# ---------------------------------------------------------------------------
# VAR Model
# ---------------------------------------------------------------------------

class VARModel:
    """VAR(p) model fitted with OLS.

    Parameters
    ----------
    p : int
        Number of lags.
    """

    def __init__(self, p=1):
        self.p = p
        self.intercept = None
        self.coefficients = None  # list of K×K matrices, one per lag
        self._fitted = False

    def fit(self, data):
        """Fit the VAR(p) model via OLS on each equation.

        Parameters
        ----------
        data : array-like of shape (T, K)
            Multivariate time series with T observations and K variables.

        Returns
        -------
        self
        """
        Y = np.asarray(data, dtype=float)
        T, K = Y.shape

        # Build matrices: for each t = p..T-1, stack [1, Y_{t-1}, ..., Y_{t-p}]
        rows = T - self.p
        X = np.ones((rows, 1 + K * self.p))
        Z = Y[self.p:]  # dependent variables
        for i in range(rows):
            for lag in range(self.p):
                X[i, 1 + lag * K: 1 + (lag + 1) * K] = Y[self.p + i - 1 - lag]

        # OLS: B = (X'X)^-1 X'Z   — B has shape (1 + K*p, K)
        B = np.linalg.lstsq(X, Z, rcond=None)[0]

        self.intercept = B[0]
        self.coefficients = [B[1 + lag * K: 1 + (lag + 1) * K] for lag in range(self.p)]
        self._data = Y
        self._K = K
        self._fitted = True
        return self

    def forecast(self, steps=1):
        """Forecast future values.

        Parameters
        ----------
        steps : int
            Number of steps ahead.

        Returns
        -------
        numpy.ndarray of shape (steps, K)
            Forecasted values.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before forecasting.")
        history = list(self._data)
        preds = []
        for _ in range(steps):
            val = self.intercept.copy()
            for lag in range(self.p):
                val = val + self.coefficients[lag].T @ history[-(1 + lag)]
            preds.append(val)
            history.append(val)
        return np.array(preds)
