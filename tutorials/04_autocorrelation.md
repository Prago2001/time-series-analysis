# Autocorrelation

## How it works

Autocorrelation measures how well a series correlates with a lagged copy of
itself. For lag `k`:

```
acf(k) = sum((x[t] - mean) * (x[t-k] - mean)) / sum((x[t] - mean)^2)
```

Values range from -1 to 1. A high value at lag `k` means observations `k` steps
apart move together — useful for detecting seasonality and choosing model
orders.

## Example

`examples/autocorrelation.py` computes the lag-1 autocorrelation of an
alternating series, which should be strongly negative:

```
data: 1, -1, 1, -1, 1, -1
acf(1): -0.833
```

Run it:

```bash
python tutorials/examples/autocorrelation.py
```
