# Moving Average

## How it works

A moving average smooths short-term noise by replacing each point with the
average of a fixed-size window of neighbouring points. For a window of size `w`,
the value at index `t` becomes:

```
MA[t] = (x[t-w+1] + ... + x[t]) / w
```

A larger window removes more noise but reacts more slowly to changes. It is the
simplest tool to reveal the underlying trend of a series.

## Example

`examples/moving_average.py` computes a 3-point moving average of a noisy
upward series:

```
data:    10, 12, 11, 13, 15, 14, 16
ma(3):   -, -, 11.0, 12.0, 13.0, 14.0, 15.0
```

The output trends smoothly upward even though the raw data zig-zags.

Run it:

```bash
python tutorials/examples/moving_average.py
```
