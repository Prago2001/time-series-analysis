"""Test data: Indian-market sample dataset and an optional live loader.

This is a sample dataset used only for testing the library; it is not part of
the published package.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

__all__ = ["load_nse_prices", "load_mutual_fund_nav", "available_symbols", "download_nse"]

_DATASETS = Path(__file__).resolve().parent / "datasets"


def _read_bundled(filename: str) -> pd.DataFrame:
    return pd.read_csv(_DATASETS / filename, index_col="Date", parse_dates=["Date"])


def load_nse_prices(symbols: str | list[str] | None = None) -> pd.DataFrame:
    """Load the daily NSE close-price panel (2018-2023, synthetic).

    Parameters
    ----------
    symbols : str | list[str] | None
        One or more column names (e.g. ``"RELIANCE"`` or ``["NIFTY50", "TCS"]``).
        ``None`` returns the full panel.
    """
    df = _read_bundled("nse_prices.csv")
    if symbols is None:
        return df
    if isinstance(symbols, str):
        return df[[symbols]]
    return df[list(symbols)]


def load_mutual_fund_nav() -> pd.Series:
    """Load the monthly equity mutual-fund NAV series (synthetic)."""
    df = _read_bundled("mutual_fund_nav.csv")
    return df["NAV"]


def available_symbols() -> list[str]:
    """Return the equity symbols available in the price panel."""
    return list(_read_bundled("nse_prices.csv").columns)


def download_nse(
    symbols: str | list[str],
    start: str | None = None,
    end: str | None = None,
    column: str = "Close",
) -> pd.DataFrame:
    """Download live NSE prices via :mod:`yfinance` (optional dependency).

    NSE tickers use the ``.NS`` suffix on Yahoo Finance; it is appended
    automatically when missing.  Requires internet access and ``yfinance``.

    Falls back with a clear error if the package or network is unavailable.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - optional path
        raise ImportError(
            "download_nse requires yfinance. Install it with `uv add yfinance`, "
            "or use load_nse_prices() for the offline sample data."
        ) from exc

    if isinstance(symbols, str):
        symbols = [symbols]
    tickers = [s if s.endswith(".NS") else f"{s}.NS" for s in symbols]
    data = yf.download(tickers, start=start, end=end, progress=False)
    prices = data[column]
    prices.columns = [c.replace(".NS", "") for c in prices.columns]
    return prices
