"""Moving average smoothing example."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from timeseries import moving_average


def main():
    data = [10, 12, 11, 13, 15, 14, 16]
    print("data:", data)
    print("ma(3):", moving_average(data, 3))


if __name__ == "__main__":
    main()
