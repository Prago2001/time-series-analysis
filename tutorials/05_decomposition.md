# Seasonal Decomposition

## How it works

Decomposition splits a series into three parts:

- **Trend**: the long-term direction, estimated with a moving average.
- **Seasonal**: the repeating pattern of a fixed period, estimated by averaging
  each position across cycles after removing the trend.
- **Residual**: whatever remains (`data - trend - seasonal`).

This additive model assumes `data = trend + seasonal + residual` and helps you
understand and forecast each component separately.

## Example

`examples/decomposition.py` decomposes a series with period 2:

```
data:     [10, 20, 12, 22, 14, 24]
seasonal: [-5, 5, -5, 5, -5, 5]
```

The seasonal part captures the regular low/high swing.

Run it:

```bash
python tutorials/examples/decomposition.py
```
