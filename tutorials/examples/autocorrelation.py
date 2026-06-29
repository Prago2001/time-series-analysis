"""Autocorrelation example."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from timeseries import autocorrelation


def main():
    data = [1, -1, 1, -1, 1, -1]
    print("data:", data)
    print("acf(1):", autocorrelation(data, 1))


if __name__ == "__main__":
    main()
