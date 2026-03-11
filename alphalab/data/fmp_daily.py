"""FMP-based daily OHLC data provider.

Uses Financial Modeling Prep stable API for daily price data.
Advantages over yfinance:
  - VWAP included (better execution price modeling)
  - No dividend-adjusted prices (raw market prices)
  - Consistent with FMP financials (same source)
  - More reliable for backtesting (no retroactive adjustments)

Usage:
    from alphalab.data.fmp_daily import FMPDailyProvider
    provider = FMPDailyProvider(api_key="...", cache_dir="data/fmp_daily/cache")
    df = provider.get_ohlc("NVDA", start="2020-01-01", end="2025-12-31")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from alphalab.data.cache import FileCache

logger = logging.getLogger(__name__)

FMP_STABLE_URL = "https://financialmodelingprep.com/stable"


class FMPDailyProvider:
    """Daily OHLC provider using FMP stable API."""

    def __init__(
        self,
        api_key: str,
        cache_dir: str = "data/fmp_daily/cache",
        cache_ttl_hours: int = 12,
    ):
        self.api_key = api_key
        self.cache = FileCache(cache_dir=cache_dir, ttl_hours=cache_ttl_hours)

    def _fmp_get(self, endpoint: str, params: dict | None = None) -> list[dict]:
        params = params or {}
        params["apikey"] = self.api_key
        url = f"{FMP_STABLE_URL}/{endpoint}"
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []

    def get_ohlc(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get daily OHLC + VWAP data from FMP.

        Returns DataFrame with columns: Open, High, Low, Close, Volume, VWAP
        and a DatetimeIndex sorted ascending.
        """
        # Download full history and cache
        cache_key = f"fmp_daily_{ticker}"
        cached = self.cache.get_df(cache_key)

        if cached is not None:
            logger.info("FMP daily %s: using cached data (%d rows)", ticker, len(cached))
            df = cached
        else:
            logger.info("FMP daily %s: downloading from FMP", ticker)
            params = {"symbol": ticker}
            if start:
                params["from"] = start
            data = self._fmp_get("historical-price-eod/full", params)

            if not data:
                raise ValueError(f"No FMP daily data for {ticker}")

            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()

            # Rename to standard format
            df = df.rename(columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
                "vwap": "VWAP",
            })

            # Keep only needed columns
            keep = ["Open", "High", "Low", "Close", "Volume", "VWAP"]
            df = df[[c for c in keep if c in df.columns]]

            self.cache.set_df(cache_key, df)

        # Slice to requested range
        if start:
            df = df.loc[start:]
        if end:
            df = df.loc[:end]

        if df.empty:
            raise ValueError(f"No FMP daily data for {ticker} in range {start} to {end}")

        return df

    def get_ohlc_batch(
        self,
        tickers: list[str],
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> dict[str, pd.DataFrame]:
        """Get daily OHLC for multiple tickers."""
        result = {}
        for ticker in tickers:
            try:
                result[ticker] = self.get_ohlc(ticker, start, end)
            except Exception as e:
                logger.warning("Failed to get %s: %s", ticker, e)
        return result
