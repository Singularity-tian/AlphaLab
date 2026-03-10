"""Data provider using yfinance for OHLC and FMP stable API for financials."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from alphalab.config import AlphaLabConfig
from alphalab.data.cache import FileCache

logger = logging.getLogger(__name__)

FMP_STABLE_URL = "https://financialmodelingprep.com/stable"


class DataProvider:
    """Unified data provider for OHLC prices, financial ratios, and transcripts."""

    def __init__(self, config: AlphaLabConfig):
        self.config = config
        self.cache = FileCache(
            cache_dir=config.data.cache_dir,
            ttl_hours=config.data.cache_ttl_hours,
        )
        self._yf_cache: dict[str, yf.Ticker] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_yf_ticker(self, ticker: str) -> yf.Ticker:
        if ticker not in self._yf_cache:
            self._yf_cache[ticker] = yf.Ticker(ticker)
        return self._yf_cache[ticker]

    def _fmp_get(self, endpoint: str, params: dict | None = None) -> list[dict]:
        """Authenticated GET to FMP stable API. Returns parsed JSON list."""
        params = params or {}
        params["apikey"] = self.config.data.fmp_api_key
        url = f"{FMP_STABLE_URL}/{endpoint}"
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
        return []

    # ------------------------------------------------------------------
    # OHLC (yfinance — free and reliable)
    # ------------------------------------------------------------------

    def get_ohlc(self, ticker: str) -> pd.DataFrame:
        """Get daily OHLC data formatted for backtesting.py.

        Returns DataFrame with columns: Open, High, Low, Close, Volume
        and a DatetimeIndex.

        Downloads full history (from 2015-01-01) and caches it date-independently,
        then slices to the requested backtest date range.
        """
        # Date-independent cache: download full history once per ticker
        full_cache_key = f"ohlc_full_{ticker}"
        full_df = self.cache.get_df(full_cache_key)

        if full_df is not None:
            logger.info("OHLC %s: using cached data (%d rows)", ticker, len(full_df))

        if full_df is None:
            logger.info("OHLC %s: downloading from yfinance", ticker)
            t = self._get_yf_ticker(ticker)
            hist = t.history(start="2015-01-01")

            if hist.empty:
                raise ValueError(f"No OHLC data returned for {ticker}")

            # yfinance returns timezone-aware index; strip timezone for compatibility
            if hist.index.tz is not None:
                hist.index = hist.index.tz_localize(None)
            hist.index.name = None

            # Keep only OHLC + Volume columns
            keep_cols = ["Open", "High", "Low", "Close", "Volume"]
            hist = hist[[c for c in keep_cols if c in hist.columns]]

            for col in ["Open", "High", "Low", "Close"]:
                if col not in hist.columns:
                    raise ValueError(f"Missing required OHLC column: {col}")

            self.cache.set_df(full_cache_key, hist)
            full_df = hist

        # Slice to requested date range
        start = self.config.backtest.start_date
        end = self.config.backtest.end_date
        sliced = full_df.loc[start:end]

        if sliced.empty:
            raise ValueError(f"No OHLC data for {ticker} in range {start} to {end}")

        return sliced

    # ------------------------------------------------------------------
    # Financial ratios (FMP stable API — 80 quarters)
    # ------------------------------------------------------------------

    def _get_fmp_statements(self, ticker: str) -> tuple[list[dict], list[dict], list[dict]]:
        """Fetch and cache raw FMP financial statements (date-independent).

        Returns (income, balance, cashflow) lists. Each is cached separately
        so that different backtest date ranges share the same raw data.
        """
        income_key = f"fmp_income_{ticker}"
        balance_key = f"fmp_balance_{ticker}"
        cashflow_key = f"fmp_cashflow_{ticker}"

        income = self.cache.get_json(income_key)
        balance = self.cache.get_json(balance_key)
        cashflow = self.cache.get_json(cashflow_key)

        if income is not None and balance is not None and cashflow is not None:
            logger.info("FMP %s: using cached statements", ticker)
            return income, balance, cashflow

        logger.info("FMP %s: downloading from API", ticker)
        try:
            if income is None:
                income = self._fmp_get(
                    "income-statement", {"symbol": ticker, "period": "quarter", "limit": 80}
                )
                self.cache.set_json(income_key, income)
            if balance is None:
                balance = self._fmp_get(
                    "balance-sheet-statement", {"symbol": ticker, "period": "quarter", "limit": 80}
                )
                self.cache.set_json(balance_key, balance)
            if cashflow is None:
                cashflow = self._fmp_get(
                    "cash-flow-statement", {"symbol": ticker, "period": "quarter", "limit": 80}
                )
                self.cache.set_json(cashflow_key, cashflow)
        except Exception as e:
            logger.warning("FMP API error for %s: %s", ticker, e)
            return income or [], balance or [], cashflow or []

        return income, balance, cashflow

    def get_ratios(self, ticker: str) -> dict[str, pd.Series]:
        """Get financial ratios as daily-frequency Series (forward-filled).

        Uses FMP stable API for quarterly financial statements (up to 80
        quarters / 20 years), then computes ratios and forward-fills to
        daily OHLC frequency.

        Raw FMP statements are cached date-independently per ticker.
        Computed ratios are cached per ticker+date range for speed.

        Returns dict mapping ratio name to pd.Series with DatetimeIndex.
        """
        cache_key = (
            f"ratios_fmp_{ticker}_{self.config.backtest.start_date}_{self.config.backtest.end_date}"
        )
        cached = self.cache.get_df(cache_key)
        if cached is not None:
            return {col: cached[col] for col in cached.columns}

        ohlc = self.get_ohlc(ticker)

        income, balance, cashflow = self._get_fmp_statements(ticker)

        if not income and not balance and not cashflow:
            logger.warning("No FMP statement data for %s", ticker)
            return {}

        quarterly_data = self._compute_ratios_from_fmp(income, balance, cashflow, ohlc)

        if not quarterly_data:
            return {}

        ratios_df = self._statements_to_daily(quarterly_data, ohlc.index)

        if not ratios_df.empty:
            self.cache.set_df(cache_key, ratios_df)

        return {col: ratios_df[col] for col in ratios_df.columns}

    def _compute_ratios_from_fmp(
        self,
        income: list[dict],
        balance: list[dict],
        cashflow: list[dict],
        ohlc: pd.DataFrame,
    ) -> dict[str, pd.Series]:
        """Compute ratio time series from FMP statement data."""
        result: dict[str, pd.Series] = {}

        # Build lookup dicts keyed by date
        def _to_series(records: list[dict], field: str) -> pd.Series:
            vals: dict[pd.Timestamp, float] = {}
            for rec in records:
                dt = pd.Timestamp(rec.get("date", rec.get("fillingDate", "")))
                v = rec.get(field)
                if dt is not pd.NaT and v is not None:
                    try:
                        vals[dt] = float(v)
                    except (ValueError, TypeError):
                        pass
            if not vals:
                return pd.Series(dtype=float)
            s = pd.Series(vals).sort_index()
            return s[~s.index.duplicated(keep="first")]

        # ------ Raw series from statements ------
        eps = _to_series(income, "eps")
        net_income = _to_series(income, "netIncome")
        shares = _to_series(income, "weightedAverageShsOut")

        equity = _to_series(balance, "totalStockholdersEquity")
        current_assets = _to_series(balance, "totalCurrentAssets")
        current_liabilities = _to_series(balance, "totalCurrentLiabilities")

        fcf = _to_series(cashflow, "freeCashFlow")
        dividends_paid = _to_series(cashflow, "commonDividendsPaid")

        # ------ EPS ------
        if not eps.empty:
            result["earnings_per_share"] = eps
            logger.debug("EPS: %d data points", len(eps))

        # ------ Book Value Per Share ------
        if not equity.empty and not shares.empty:
            common_idx = equity.index.intersection(shares.index)
            if len(common_idx) > 0:
                bvps = equity[common_idx] / shares[common_idx]
                bvps = bvps.replace([np.inf, -np.inf], np.nan).dropna()
                if not bvps.empty:
                    result["book_value_per_share"] = bvps

        # ------ Current Ratio ------
        if not current_assets.empty and not current_liabilities.empty:
            common_idx = current_assets.index.intersection(current_liabilities.index)
            if len(common_idx) > 0:
                cr = current_assets[common_idx] / current_liabilities[common_idx]
                cr = cr.replace([np.inf, -np.inf], np.nan).dropna()
                if not cr.empty:
                    result["current_ratio"] = cr

        # ------ PE Ratio = Price / EPS ------
        if not eps.empty:
            pe_values: dict[pd.Timestamp, float] = {}
            for dt in eps.index:
                e = eps[dt]
                if e > 0:
                    price_slice = ohlc["Close"].loc[:dt]
                    if not price_slice.empty:
                        pe_values[dt] = price_slice.iloc[-1] / e
            if pe_values:
                result["pe_ratio"] = pd.Series(pe_values).sort_index()

        # ------ Price to Book = Price / BVPS ------
        if "book_value_per_share" in result:
            pb_values: dict[pd.Timestamp, float] = {}
            for dt in result["book_value_per_share"].index:
                bv = result["book_value_per_share"][dt]
                if bv > 0:
                    price_slice = ohlc["Close"].loc[:dt]
                    if not price_slice.empty:
                        pb_values[dt] = price_slice.iloc[-1] / bv
            if pb_values:
                result["price_to_book"] = pd.Series(pb_values).sort_index()

        # ------ Dividend Yield (TTM) ------
        if not dividends_paid.empty:
            # dividendsPaid is typically negative (cash outflow); take abs
            div_abs = dividends_paid.abs()
            if not shares.empty:
                common_idx = div_abs.index.intersection(shares.index)
                if len(common_idx) > 0:
                    dps = div_abs[common_idx] / shares[common_idx]
                    # Compute trailing 4-quarter sum for TTM dividend
                    dps_sorted = dps.sort_index()
                    dy_values: dict[pd.Timestamp, float] = {}
                    for i in range(len(dps_sorted)):
                        dt = dps_sorted.index[i]
                        # Sum last 4 quarters of DPS
                        lookback = dps_sorted.iloc[max(0, i - 3): i + 1]
                        ttm_dps = lookback.sum()
                        price_slice = ohlc["Close"].loc[:dt]
                        if not price_slice.empty and price_slice.iloc[-1] > 0:
                            dy_values[dt] = ttm_dps / price_slice.iloc[-1]
                    if dy_values:
                        result["dividend_yield"] = pd.Series(dy_values).sort_index()

        # ------ ROE = Net Income / Equity ------
        if not net_income.empty and not equity.empty:
            common_idx = net_income.index.intersection(equity.index)
            if len(common_idx) > 0:
                roe = net_income[common_idx] / equity[common_idx]
                roe = roe.replace([np.inf, -np.inf], np.nan).dropna()
                if not roe.empty:
                    result["roe"] = roe

        # ------ FCF Yield = FCF / Market Cap ------
        if not fcf.empty and not shares.empty:
            common_idx = fcf.index.intersection(shares.index)
            if len(common_idx) > 0:
                fcf_yield_values: dict[pd.Timestamp, float] = {}
                for dt in common_idx:
                    price_slice = ohlc["Close"].loc[:dt]
                    if not price_slice.empty:
                        mktcap = price_slice.iloc[-1] * shares[dt]
                        if mktcap > 0:
                            fcf_yield_values[dt] = fcf[dt] / mktcap
                if fcf_yield_values:
                    result["fcf_yield"] = pd.Series(fcf_yield_values).sort_index()

        for name, series in result.items():
            logger.debug(
                "Raw ratio %s: %d data points, range %s to %s",
                name, len(series),
                series.index.min() if not series.empty else "N/A",
                series.index.max() if not series.empty else "N/A",
            )

        return result

    def _statements_to_daily(
        self, ratio_data: dict[str, pd.Series], daily_index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Convert statement-frequency ratio data to daily via forward-fill."""
        result_dict: dict[str, pd.Series] = {}

        for name, series in ratio_data.items():
            ts = series.sort_index()
            # Reindex to daily OHLC dates and forward-fill
            daily = ts.reindex(daily_index).ffill()
            result_dict[name] = daily

            non_null = daily.notna().sum()
            logger.debug(
                "Ratio %s: %d statement values -> %d/%d daily values",
                name, len(series), non_null, len(daily),
            )

        if not result_dict:
            return pd.DataFrame()

        return pd.DataFrame(result_dict, index=daily_index)

    # ------------------------------------------------------------------
    # Transcripts (FMP stable API)
    # ------------------------------------------------------------------

    def get_earnings_transcripts(self, ticker: str, year: int, quarter: int) -> str | None:
        """Fetch earnings call transcript from FMP stable API."""
        cache_key = f"transcript_{ticker}_{year}_Q{quarter}"
        cached = self.cache.get_json(cache_key)
        if cached is not None:
            return cached.get("content")

        try:
            data = self._fmp_get(
                "earning-call-transcript",
                {"symbol": ticker, "quarter": quarter, "year": year},
            )
            if data and len(data) > 0:
                content = data[0].get("content", "")
                self.cache.set_json(cache_key, {"content": content})
                return content
        except Exception as e:
            logger.warning("Failed to fetch transcript %s %dQ%d: %s", ticker, year, quarter, e)

        return None

    def get_available_transcript_dates(self, ticker: str) -> list[tuple[int, int]]:
        """Get list of (year, quarter) tuples for available transcripts."""
        cache_key = f"transcript_dates_{ticker}"
        cached = self.cache.get_json(cache_key)
        if cached is not None:
            return [(d["year"], d["quarter"]) for d in cached]

        try:
            data = self._fmp_get(
                "earning-call-transcript", {"symbol": ticker}
            )
            self.cache.set_json(cache_key, data)
            return [(d["year"], d["quarter"]) for d in data]
        except Exception as e:
            logger.warning("Failed to fetch transcript dates for %s: %s", ticker, e)
            return []

    # ------------------------------------------------------------------
    # Company profile & segmentation (FMP stable API)
    # ------------------------------------------------------------------

    def get_company_profile(self, ticker: str) -> dict | None:
        """Fetch company profile (description, sector, industry, etc.)."""
        cache_key = f"profile_{ticker}"
        cached = self.cache.get_json(cache_key)
        if cached is not None:
            return cached

        try:
            rows = self._fmp_get("profile", {"symbol": ticker})
            if not rows:
                return None

            row = rows[0]
            profile = {
                "company_name": row.get("companyName", ticker),
                "description": row.get("description", ""),
                "sector": row.get("sector", ""),
                "industry": row.get("industry", ""),
                "mkt_cap": row.get("mktCap"),
                "full_time_employees": row.get("fullTimeEmployees"),
                "country": row.get("country", ""),
            }
            self.cache.set_json(cache_key, profile)
            return profile
        except Exception as e:
            logger.warning("Failed to fetch profile for %s: %s", ticker, e)
            return None

    def get_revenue_segmentation(self, ticker: str) -> dict | None:
        """Fetch revenue breakdown by product and geography.

        Returns dict with 'by_product' and 'by_geography' percentage dicts,
        or None if unavailable.
        """
        cache_key = f"revenue_seg_{ticker}"
        cached = self.cache.get_json(cache_key)
        if cached is not None:
            return cached

        result: dict = {}

        try:
            # Product segmentation
            prod_data = self._fmp_get(
                "revenue-product-segmentation",
                {"symbol": ticker, "period": "annual"},
            )
            if prod_data:
                # Most recent year first; each item is {date: ..., segment_name: value}
                latest = prod_data[0]
                # Extract segment dict (all keys except non-segment metadata)
                segments = {}
                for item in prod_data:
                    if isinstance(item, dict):
                        for k, v in item.items():
                            if isinstance(v, dict):
                                segments = v
                                break
                        if segments:
                            break

                if segments:
                    total = sum(abs(v) for v in segments.values() if isinstance(v, (int, float)))
                    if total > 0:
                        result["by_product"] = {
                            k: round(abs(v) / total * 100, 1)
                            for k, v in segments.items()
                            if isinstance(v, (int, float))
                        }
        except Exception as e:
            logger.warning("Failed to fetch product segmentation for %s: %s", ticker, e)

        try:
            # Geographic segmentation
            geo_data = self._fmp_get(
                "revenue-geographic-segments",
                {"symbol": ticker, "period": "annual"},
            )
            if geo_data:
                segments = {}
                for item in geo_data:
                    if isinstance(item, dict):
                        for k, v in item.items():
                            if isinstance(v, dict):
                                segments = v
                                break
                        if segments:
                            break

                if segments:
                    total = sum(abs(v) for v in segments.values() if isinstance(v, (int, float)))
                    if total > 0:
                        result["by_geography"] = {
                            k: round(abs(v) / total * 100, 1)
                            for k, v in segments.items()
                            if isinstance(v, (int, float))
                        }
        except Exception as e:
            logger.warning("Failed to fetch geo segmentation for %s: %s", ticker, e)

        if not result:
            return None

        self.cache.set_json(cache_key, result)
        return result

    def get_employee_count(self, ticker: str) -> int | None:
        """Fetch latest employee count."""
        cache_key = f"employees_{ticker}"
        cached = self.cache.get_json(cache_key)
        if cached is not None:
            return cached.get("count")

        try:
            data = self._fmp_get("employee-count", {"symbol": ticker})
            if data:
                count = data[0].get("employeeCount")
                if count is not None:
                    self.cache.set_json(cache_key, {"count": count})
                    return count
        except Exception as e:
            logger.warning("Failed to fetch employee count for %s: %s", ticker, e)

        # Fallback: try from profile
        profile = self.get_company_profile(ticker)
        if profile and profile.get("full_time_employees"):
            return profile["full_time_employees"]

        return None
