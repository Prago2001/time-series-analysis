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
