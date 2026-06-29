"""Autocorrelation example (standard library only)."""


def autocorrelation(data, lag):
    """Return the autocorrelation at the given lag."""
    n = len(data)
    mean = sum(data) / n
    denom = sum((x - mean) ** 2 for x in data)
    if denom == 0:
        return 0.0
    num = sum((data[t] - mean) * (data[t - lag] - mean) for t in range(lag, n))
    return num / denom


def main():
    data = [1, -1, 1, -1, 1, -1]
    print("data:", data)
    print("acf(1):", autocorrelation(data, 1))


if __name__ == "__main__":
    main()
