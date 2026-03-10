"""Download hourly OHLC data for all S&P 500 tickers from FMP API.

Stores one Parquet file per ticker in the EXPMA12&KDJ/sp500_hourly_data/ folder.
Supports resume — skips tickers that already have a Parquet file.

FMP API constraints:
  - Max 90-day date range per request
  - Max 4000 records per request
  - Rate limit: 300 requests/minute

Usage:
    python "EXPMA12&KDJ/download_hourly.py"
    python "EXPMA12&KDJ/download_hourly.py" --force   # re-download all
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# Add project root to path for importing SP500 list
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from examples.portfolio_backtest import SP500

FMP_STABLE_URL = "https://financialmodelingprep.com/stable"
OUTPUT_DIR = Path(__file__).resolve().parent / "sp500_hourly_data"
START_DATE = date(2021, 1, 1)
END_DATE = date(2026, 3, 9)
CHUNK_DAYS = 90
RATE_LIMIT = 300  # requests per minute


class RateLimiter:
    """Token-bucket rate limiter for API requests."""

    def __init__(self, max_requests: int = 300, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window = window_seconds
        self.timestamps: deque[float] = deque(maxlen=max_requests)

    def wait_if_needed(self):
        now = time.time()
        if len(self.timestamps) >= self.max_requests:
            elapsed = now - self.timestamps[0]
            if elapsed < self.window:
                sleep_time = self.window - elapsed + 0.1
                print(f"    Rate limit reached, sleeping {sleep_time:.1f}s...")
                time.sleep(sleep_time)
        self.timestamps.append(time.time())


def generate_90day_chunks(start: date, end: date) -> list[tuple[str, str]]:
    """Generate (from_date, to_date) string pairs in 90-day increments."""
    chunks = []
    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=CHUNK_DAYS - 1), end)
        chunks.append((current.isoformat(), chunk_end.isoformat()))
        current = chunk_end + timedelta(days=1)
    return chunks


def fetch_hourly(ticker: str, api_key: str, rate_limiter: RateLimiter) -> pd.DataFrame:
    """Fetch all hourly OHLC data for a ticker across the full date range."""
    chunks = generate_90day_chunks(START_DATE, END_DATE)
    all_data: list[dict] = []

    for from_date, to_date in chunks:
        rate_limiter.wait_if_needed()

        resp = requests.get(
            f"{FMP_STABLE_URL}/historical-chart/1hour",
            params={
                "symbol": ticker,
                "from": from_date,
                "to": to_date,
                "apikey": api_key,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, list) and data:
            all_data.extend(data)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    df = df.set_index("date")

    keep = ["open", "high", "low", "close", "volume"]
    df = df[[c for c in keep if c in df.columns]]

    return df


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Download hourly OHLC for S&P 500")
    parser.add_argument("--force", action="store_true", help="Re-download even if parquet exists")
    args = parser.parse_args()

    api_key = os.environ.get("FMP_API_KEY", "")
    if not api_key:
        print("ERROR: FMP_API_KEY not set in environment")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rate_limiter = RateLimiter(max_requests=RATE_LIMIT)
    total = len(SP500)
    ok, skip, fail = 0, 0, 0
    start_time = time.time()

    chunks_per_ticker = len(generate_90day_chunks(START_DATE, END_DATE))
    total_requests = total * chunks_per_ticker
    print(f"Downloading hourly OHLC for {total} tickers")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Chunks per ticker: {chunks_per_ticker} (90-day each)")
    print(f"Estimated total requests: {total_requests} (~{total_requests // RATE_LIMIT} min)")
    print(f"Output dir: {OUTPUT_DIR}")
    print()

    for i, ticker in enumerate(SP500, 1):
        tag = f"[{i:>3}/{total}] {ticker:<6}"
        out_path = OUTPUT_DIR / f"{ticker}.parquet"

        if out_path.exists() and not args.force:
            print(f"  {tag} SKIP (already exists)")
            skip += 1
            continue

        try:
            df = fetch_hourly(ticker, api_key, rate_limiter)
            if df.empty:
                print(f"  {tag} SKIP (no data)")
                skip += 1
                continue

            df.to_parquet(out_path)
            print(f"  {tag} OK ({len(df):,} rows)")
            ok += 1
        except Exception as e:
            print(f"  {tag} FAIL: {e}")
            fail += 1

    elapsed = time.time() - start_time
    print()
    print(f"Done in {elapsed:.0f}s — OK: {ok}, Skipped: {skip}, Failed: {fail}")


if __name__ == "__main__":
    main()
