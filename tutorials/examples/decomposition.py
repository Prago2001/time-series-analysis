"""Simple additive seasonal decomposition example."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from timeseries import seasonal_component


def main():
    data = [10, 20, 12, 22, 14, 24]
    print("data:", data)
    print("seasonal:", seasonal_component(data, 2))


if __name__ == "__main__":
    main()
