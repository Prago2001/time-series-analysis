"""Differencing example to remove trend."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from timeseries import difference


def main():
    data = [2, 4, 6, 8, 10]
    print("data:", data)
    print("diff:", difference(data))


if __name__ == "__main__":
    main()
