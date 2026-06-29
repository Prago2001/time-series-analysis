"""Exponential smoothing example (standard library only)."""


def exponential_smoothing(data, alpha):
    """Return the exponentially smoothed series."""
    if not data:
        return []
    smoothed = [data[0]]
    for x in data[1:]:
        smoothed.append(alpha * x + (1 - alpha) * smoothed[-1])
    return smoothed


def main():
    data = [3, 5, 8, 6, 9]
    smooth = exponential_smoothing(data, 0.5)
    print("data:", data)
    print("smooth:", smooth)


if __name__ == "__main__":
    main()
