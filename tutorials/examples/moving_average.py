"""Moving average smoothing example (standard library only)."""


def moving_average(data, window):
    """Return the simple moving average; first window-1 entries are None."""
    result = []
    for i in range(len(data)):
        if i + 1 < window:
            result.append(None)
        else:
            avg = sum(data[i - window + 1:i + 1]) / window
            result.append(avg)
    return result


def main():
    data = [10, 12, 11, 13, 15, 14, 16]
    ma = moving_average(data, 3)
    print("data:", data)
    print("ma(3):", ma)


if __name__ == "__main__":
    main()
