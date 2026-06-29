"""Simple time series analysis methods (standard library only)."""


def moving_average(data, window):
    """Simple moving average; first window-1 entries are None."""
    out = []
    for i in range(len(data)):
        if i + 1 < window:
            out.append(None)
        else:
            out.append(sum(data[i - window + 1:i + 1]) / window)
    return out


def exponential_smoothing(data, alpha):
    """Exponentially smoothed series."""
    if not data:
        return []
    smoothed = [data[0]]
    for x in data[1:]:
        smoothed.append(alpha * x + (1 - alpha) * smoothed[-1])
    return smoothed


def difference(data, lag=1):
    """Lag-differenced series."""
    return [data[i] - data[i - lag] for i in range(lag, len(data))]


def autocorrelation(data, lag):
    """Autocorrelation at the given lag."""
    n = len(data)
    mean = sum(data) / n
    denom = sum((x - mean) ** 2 for x in data)
    if denom == 0:
        return 0.0
    num = sum((data[t] - mean) * (data[t - lag] - mean) for t in range(lag, n))
    return num / denom


def seasonal_component(data, period):
    """Additive seasonal component for the given period."""
    overall = sum(data) / len(data)
    seasonal = [sum(data[p::period]) / len(data[p::period]) - overall
                for p in range(period)]
    return [seasonal[i % period] for i in range(len(data))]
