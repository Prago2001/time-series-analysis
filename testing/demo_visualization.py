"""Visualization demonstrations using Indian stocks and mutual funds data.

Run this script to generate plots demonstrating all time series analysis methods.
Plots are saved to the testing/plots/ directory.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib.pyplot as plt

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
    INFOSYS_MONTHLY,
    HDFC_BANK_MONTHLY,
    NIFTY50_MONTHLY,
    SBI_BLUECHIP_NAV,
    AXIS_ELSS_NAV,
    MIRAE_LARGECAP_NAV,
    PPFAS_FLEXICAP_NAV,
    RELIANCE_DAILY,
    MONTHS_2023,
)


def ensure_plots_dir():
    """Create plots directory if it doesn't exist."""
    plots_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


def plot_moving_average_stocks(plots_dir):
    """Plot moving averages for Indian stocks."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Moving Average Analysis - Indian Stocks (2023)", fontsize=14)

    stocks = [
        ("Reliance Industries", RELIANCE_MONTHLY),
        ("TCS", TCS_MONTHLY),
        ("Infosys", INFOSYS_MONTHLY),
        ("HDFC Bank", HDFC_BANK_MONTHLY),
    ]

    for ax, (name, data) in zip(axes.flat, stocks):
        ma3 = moving_average(data, 3)
        ma5 = moving_average(data, 5)

        ax.plot(MONTHS_2023, data, "o-", label="Price", color="steelblue")
        ax.plot(MONTHS_2023, ma3, "s--", label="MA(3)", color="orange")
        ax.plot(MONTHS_2023, ma5, "^--", label="MA(5)", color="green")
        ax.set_title(f"{name} (₹)")
        ax.set_xlabel("Month")
        ax.set_ylabel("Price (INR)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "moving_average_stocks.png"), dpi=150)
    plt.close()
    print("✓ Saved: moving_average_stocks.png")


def plot_moving_average_mutual_funds(plots_dir):
    """Plot moving averages for Indian mutual funds."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Moving Average Analysis - Indian Mutual Funds NAV (2023)", fontsize=14)

    funds = [
        ("SBI Bluechip Fund", SBI_BLUECHIP_NAV),
        ("Axis ELSS Fund", AXIS_ELSS_NAV),
        ("Mirae Asset Large Cap", MIRAE_LARGECAP_NAV),
        ("PPFAS Flexi Cap", PPFAS_FLEXICAP_NAV),
    ]

    for ax, (name, data) in zip(axes.flat, funds):
        ma3 = moving_average(data, 3)

        ax.plot(MONTHS_2023, data, "o-", label="NAV", color="darkblue")
        ax.plot(MONTHS_2023, ma3, "s--", label="MA(3)", color="red")
        ax.set_title(f"{name}")
        ax.set_xlabel("Month")
        ax.set_ylabel("NAV (₹)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "moving_average_mutual_funds.png"), dpi=150)
    plt.close()
    print("✓ Saved: moving_average_mutual_funds.png")


def plot_exponential_smoothing(plots_dir):
    """Plot exponential smoothing comparison for Nifty50."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Exponential Smoothing - Nifty 50 Index (2023)", fontsize=14)

    ax.plot(MONTHS_2023, NIFTY50_MONTHLY, "o-", label="Nifty 50", color="black", linewidth=2)

    alphas = [0.2, 0.5, 0.8]
    colors = ["blue", "green", "red"]
    for alpha, color in zip(alphas, colors):
        smoothed = exponential_smoothing(NIFTY50_MONTHLY, alpha)
        ax.plot(MONTHS_2023, smoothed, "--", label=f"α={alpha}", color=color)

    ax.set_xlabel("Month")
    ax.set_ylabel("Index Value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "exponential_smoothing_nifty.png"), dpi=150)
    plt.close()
    print("✓ Saved: exponential_smoothing_nifty.png")


def plot_differencing(plots_dir):
    """Plot differencing for Reliance and TCS."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Differencing Analysis - Indian Stocks (2023)", fontsize=14)

    # Reliance - original and differenced
    axes[0, 0].plot(MONTHS_2023, RELIANCE_MONTHLY, "o-", color="steelblue")
    axes[0, 0].set_title("Reliance - Original Prices (₹)")
    axes[0, 0].set_ylabel("Price (INR)")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis="x", rotation=45)

    diff_rel = difference(RELIANCE_MONTHLY)
    axes[0, 1].bar(MONTHS_2023[1:], diff_rel, color=["green" if d > 0 else "red" for d in diff_rel])
    axes[0, 1].axhline(y=0, color="black", linewidth=0.5)
    axes[0, 1].set_title("Reliance - Month-over-Month Change (₹)")
    axes[0, 1].set_ylabel("Change (INR)")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis="x", rotation=45)

    # TCS - original and differenced
    axes[1, 0].plot(MONTHS_2023, TCS_MONTHLY, "o-", color="purple")
    axes[1, 0].set_title("TCS - Original Prices (₹)")
    axes[1, 0].set_ylabel("Price (INR)")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis="x", rotation=45)

    diff_tcs = difference(TCS_MONTHLY)
    axes[1, 1].bar(MONTHS_2023[1:], diff_tcs, color=["green" if d > 0 else "red" for d in diff_tcs])
    axes[1, 1].axhline(y=0, color="black", linewidth=0.5)
    axes[1, 1].set_title("TCS - Month-over-Month Change (₹)")
    axes[1, 1].set_ylabel("Change (INR)")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "differencing_stocks.png"), dpi=150)
    plt.close()
    print("✓ Saved: differencing_stocks.png")


def plot_autocorrelation(plots_dir):
    """Plot autocorrelation function for Reliance daily prices."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Autocorrelation Analysis - Indian Stocks", fontsize=14)

    # ACF for Reliance daily
    max_lag = 8
    lags = range(1, max_lag + 1)
    acf_values = [autocorrelation(RELIANCE_DAILY, lag) for lag in lags]

    axes[0].bar(lags, acf_values, color="steelblue")
    axes[0].axhline(y=0, color="black", linewidth=0.5)
    axes[0].set_title("Reliance Daily - ACF")
    axes[0].set_xlabel("Lag")
    axes[0].set_ylabel("Autocorrelation")
    axes[0].grid(True, alpha=0.3)

    # ACF for Nifty monthly
    acf_nifty = [autocorrelation(NIFTY50_MONTHLY, lag) for lag in lags]
    axes[1].bar(lags, acf_nifty, color="darkgreen")
    axes[1].axhline(y=0, color="black", linewidth=0.5)
    axes[1].set_title("Nifty 50 Monthly - ACF")
    axes[1].set_xlabel("Lag")
    axes[1].set_ylabel("Autocorrelation")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "autocorrelation_analysis.png"), dpi=150)
    plt.close()
    print("✓ Saved: autocorrelation_analysis.png")


def plot_seasonal_decomposition(plots_dir):
    """Plot seasonal decomposition for mutual fund NAVs."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle("Seasonal Decomposition - SBI Bluechip Fund NAV (Quarterly, 2023)", fontsize=14)

    data = SBI_BLUECHIP_NAV
    period = 4  # Quarterly seasonality

    seasonal = seasonal_component(data, period)
    trend = moving_average(data, period)
    # Residual where trend is available
    residual = [
        data[i] - seasonal[i] - trend[i] if trend[i] is not None else None
        for i in range(len(data))
    ]

    axes[0].plot(MONTHS_2023, data, "o-", color="darkblue", label="NAV")
    if any(t is not None for t in trend):
        trend_vals = [(m, t) for m, t in zip(MONTHS_2023, trend) if t is not None]
        axes[0].plot([m for m, _ in trend_vals], [t for _, t in trend_vals],
                     "s--", color="red", label="Trend (MA-4)")
    axes[0].set_title("Original NAV + Trend")
    axes[0].set_ylabel("NAV (₹)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(MONTHS_2023, seasonal, color="orange")
    axes[1].axhline(y=0, color="black", linewidth=0.5)
    axes[1].set_title("Seasonal Component")
    axes[1].set_ylabel("Seasonal (₹)")
    axes[1].grid(True, alpha=0.3)

    residual_plot = [r if r is not None else 0 for r in residual]
    colors = ["green" if r is not None else "lightgray" for r in residual]
    axes[2].bar(MONTHS_2023, residual_plot, color=colors)
    axes[2].axhline(y=0, color="black", linewidth=0.5)
    axes[2].set_title("Residual (where trend available)")
    axes[2].set_ylabel("Residual (₹)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "seasonal_decomposition_mf.png"), dpi=150)
    plt.close()
    print("✓ Saved: seasonal_decomposition_mf.png")


def main():
    """Generate all demonstration plots."""
    print("=" * 60)
    print("Generating visualizations with Indian stocks & mutual funds")
    print("=" * 60)
    print()

    plots_dir = ensure_plots_dir()

    plot_moving_average_stocks(plots_dir)
    plot_moving_average_mutual_funds(plots_dir)
    plot_exponential_smoothing(plots_dir)
    plot_differencing(plots_dir)
    plot_autocorrelation(plots_dir)
    plot_seasonal_decomposition(plots_dir)

    print()
    print(f"All plots saved to: {plots_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
