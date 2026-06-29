"""Simple additive seasonal decomposition example (standard library only)."""


def seasonal_component(data, period):
    """Estimate an additive seasonal component for the given period."""
    overall = sum(data) / len(data)
    seasonal = []
    for pos in range(period):
        vals = data[pos::period]
        seasonal.append(sum(vals) / len(vals) - overall)
    return [seasonal[i % period] for i in range(len(data))]


def main():
    data = [10, 20, 12, 22, 14, 24]
    seasonal = seasonal_component(data, 2)
    print("data:", data)
    print("seasonal:", seasonal)


if __name__ == "__main__":
    main()
