# Differencing & Stationarity

## How it works

Many methods assume a series is **stationary** (constant mean over time).
Differencing removes a trend by replacing each value with the change from the
previous step:

```
diff[t] = x[t] - x[t-1]
```

A series with a linear trend becomes roughly constant after one difference; you
can difference again to remove a quadratic trend. The differenced series is
usually closer to stationary, making it suitable for models like ARIMA.

## Example

`examples/differencing.py` differences a steadily rising series:

```
data: 2, 4, 6, 8, 10
diff: 2, 2, 2, 2
```

The trend disappears: the differenced values are constant.

Run it:

```bash
python tutorials/examples/differencing.py
```
