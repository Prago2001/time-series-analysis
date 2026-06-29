"""Differencing example to remove trend (standard library only)."""


def difference(data, lag=1):
    """Return the lag-differenced series."""
    return [data[i] - data[i - lag] for i in range(lag, len(data))]


def main():
    data = [2, 4, 6, 8, 10]
    diff = difference(data)
    print("data:", data)
    print("diff:", diff)


if __name__ == "__main__":
    main()
