# Exponential Smoothing

## How it works

Exponential smoothing forecasts the next value as a weighted average of the
current observation and the previous smoothed value. The smoothing factor
`alpha` (0 < alpha <= 1) controls how quickly old observations are forgotten:

```
s[0] = x[0]
s[t] = alpha * x[t] + (1 - alpha) * s[t-1]
```

A high `alpha` reacts fast to recent changes; a low `alpha` produces a smoother,
slower-moving line. `s[t]` is also the one-step-ahead forecast for `x[t+1]`.

## Example

`examples/exponential_smoothing.py` smooths a series with `alpha = 0.5`:

```
data:   3, 5, 8, 6, 9
smooth: 3.0, 4.0, 6.0, 6.0, 7.5
```

Each smoothed value blends the new point with the running estimate.

Run it:

```bash
python tutorials/examples/exponential_smoothing.py
```
