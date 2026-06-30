"""Unit tests for timeseries module using Indian stock/MF data."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from timeseries import (
    moving_average,
    exponential_smoothing,
    difference,
    autocorrelation,
    seasonal_component,
)
from sample_data import (
    RELIANCE_MONTHLY,
    TCS_MONTHLY,
    NIFTY50_MONTHLY,
    SBI_BLUECHIP_NAV,
    RELIANCE_DAILY,
)


def test_moving_average_reliance():
    """Test moving average on Reliance monthly prices."""
    result = moving_average(RELIANCE_MONTHLY, 3)
    # First 2 values should be None
    assert result[0] is None
    assert result[1] is None
    # Third value should be average of first 3
    expected = (RELIANCE_MONTHLY[0] + RELIANCE_MONTHLY[1] + RELIANCE_MONTHLY[2]) / 3
    assert abs(result[2] - expected) < 1e-6
    assert len(result) == len(RELIANCE_MONTHLY)
    print("✓ Moving average (Reliance) passed")


def test_moving_average_nifty():
    """Test moving average on Nifty50 with window=4."""
    result = moving_average(NIFTY50_MONTHLY, 4)
    assert result[:3] == [None, None, None]
    expected = sum(NIFTY50_MONTHLY[:4]) / 4
    assert abs(result[3] - expected) < 1e-6
    print("✓ Moving average (Nifty50) passed")


def test_exponential_smoothing_tcs():
    """Test exponential smoothing on TCS prices."""
    result = exponential_smoothing(TCS_MONTHLY, 0.3)
    assert len(result) == len(TCS_MONTHLY)
    assert result[0] == TCS_MONTHLY[0]
    # Second value: alpha * data[1] + (1-alpha) * data[0]
    expected = 0.3 * TCS_MONTHLY[1] + 0.7 * TCS_MONTHLY[0]
    assert abs(result[1] - expected) < 1e-6
    print("✓ Exponential smoothing (TCS) passed")


def test_exponential_smoothing_nav():
    """Test exponential smoothing on mutual fund NAV."""
    result = exponential_smoothing(SBI_BLUECHIP_NAV, 0.5)
    assert len(result) == len(SBI_BLUECHIP_NAV)
    assert result[0] == SBI_BLUECHIP_NAV[0]
    print("✓ Exponential smoothing (SBI Bluechip NAV) passed")


def test_difference_reliance():
    """Test differencing on Reliance monthly data."""
    result = difference(RELIANCE_MONTHLY)
    assert len(result) == len(RELIANCE_MONTHLY) - 1
    expected_first = RELIANCE_MONTHLY[1] - RELIANCE_MONTHLY[0]
    assert abs(result[0] - expected_first) < 1e-6
    print("✓ Differencing (Reliance) passed")


def test_difference_lag2_nifty():
    """Test differencing with lag=2 on Nifty50."""
    result = difference(NIFTY50_MONTHLY, lag=2)
    assert len(result) == len(NIFTY50_MONTHLY) - 2
    expected = NIFTY50_MONTHLY[2] - NIFTY50_MONTHLY[0]
    assert abs(result[0] - expected) < 1e-6
    print("✓ Differencing lag=2 (Nifty50) passed")


def test_autocorrelation_reliance_daily():
    """Test autocorrelation on Reliance daily prices."""
    acf1 = autocorrelation(RELIANCE_DAILY, 1)
    # Autocorrelation should be between -1 and 1
    assert -1.0 <= acf1 <= 1.0
    # Daily stock prices typically have positive autocorrelation
    assert acf1 > 0
    print(f"✓ Autocorrelation lag=1 (Reliance daily): {acf1:.4f}")


def test_autocorrelation_lag5():
    """Test autocorrelation at higher lags."""
    acf5 = autocorrelation(RELIANCE_DAILY, 5)
    assert -1.0 <= acf5 <= 1.0
    print(f"✓ Autocorrelation lag=5 (Reliance daily): {acf5:.4f}")


def test_seasonal_nifty():
    """Test seasonal decomposition on Nifty50 data."""
    # Treat as quarterly seasonality (period=3)
    result = seasonal_component(NIFTY50_MONTHLY, 3)
    assert len(result) == len(NIFTY50_MONTHLY)
    # Seasonal components should sum to approximately zero over one period
    period_sum = sum(result[:3])
    assert abs(period_sum) < 1e-6
    print("✓ Seasonal component (Nifty50, period=3) passed")


def test_seasonal_mutual_fund():
    """Test seasonal decomposition on mutual fund NAV."""
    result = seasonal_component(SBI_BLUECHIP_NAV, 4)
    assert len(result) == len(SBI_BLUECHIP_NAV)
    period_sum = sum(result[:4])
    assert abs(period_sum) < 1e-6
    print("✓ Seasonal component (SBI Bluechip NAV, period=4) passed")


def test_empty_data():
    """Test edge case with empty data."""
    assert exponential_smoothing([], 0.5) == []
    print("✓ Empty data edge case passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running tests with Indian stocks and mutual funds data")
    print("=" * 60)
    print()

    test_moving_average_reliance()
    test_moving_average_nifty()
    test_exponential_smoothing_tcs()
    test_exponential_smoothing_nav()
    test_difference_reliance()
    test_difference_lag2_nifty()
    test_autocorrelation_reliance_daily()
    test_autocorrelation_lag5()
    test_seasonal_nifty()
    test_seasonal_mutual_fund()
    test_empty_data()

    print()
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
