"""Exponential smoothing example."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from timeseries import exponential_smoothing


def main():
    data = [3, 5, 8, 6, 9]
    print("data:", data)
    print("smooth:", exponential_smoothing(data, 0.5))


if __name__ == "__main__":
    main()
