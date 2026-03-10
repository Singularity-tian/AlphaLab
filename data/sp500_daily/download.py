"""Download all S&P 500 daily OHLC and financial data for offline evaluation.

Run this once to populate the cache, then run strategy backtests
without hitting APIs.

Usage:
    python data/sp500_daily/download.py
    python data/sp500_daily/download.py --ohlc-only   # skip FMP financials
"""

from __future__ import annotations

import argparse
import logging
import time

from dotenv import load_dotenv

load_dotenv()

from alphalab.config import AlphaLabConfig
from alphalab.data.providers import DataProvider
from alphalab.data.tickers import SP500

logging.basicConfig(level=logging.WARNING, format="%(message)s")


def main():
    parser = argparse.ArgumentParser(description="Download S&P 500 data")
    parser.add_argument("--ohlc-only", action="store_true", help="Skip FMP financials")
    args = parser.parse_args()

    config = AlphaLabConfig()
    config.data.cache_ttl_hours = 87600
    data = DataProvider(config)

    total = len(SP500)
    ok, skip, fail = 0, 0, 0
    start = time.time()

    print(f"Downloading data for {total} tickers...")
    print(f"Cache dir: {config.data.cache_dir}")
    print()

    for i, ticker in enumerate(SP500, 1):
        tag = f"[{i:>3}/{total}] {ticker:<6}"
        try:
            # OHLC (yfinance) — cached date-independently
            data.get_ohlc(ticker)

            # FMP financials — cached date-independently
            if not args.ohlc_only:
                data._get_fmp_statements(ticker)

            print(f"  {tag} OK")
            ok += 1
        except ValueError as e:
            print(f"  {tag} SKIP: {e}")
            skip += 1
        except Exception as e:
            print(f"  {tag} FAIL: {e}")
            fail += 1

    elapsed = time.time() - start
    print()
    print(f"Done in {elapsed:.0f}s — OK: {ok}, Skipped: {skip}, Failed: {fail}")


if __name__ == "__main__":
    main()
