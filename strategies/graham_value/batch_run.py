"""Batch backtest: run Graham strategy on top 100 stocks by market cap.

Usage:
    python strategies/graham_value/batch_run.py
"""

from __future__ import annotations

import sys
import time
import logging

from dotenv import load_dotenv

load_dotenv()

import pandas as pd

from alphalab.config import AlphaLabConfig
from alphalab.backtest.runner import BacktestRunner

import strategies.graham_value.factors  # noqa: F401  — register Graham factors

# Top 100 US stocks by market cap (as of early 2025)
TOP_100 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "LLY", "AVGO", "JPM",
    "TSLA", "UNH", "XOM", "V", "MA", "PG", "COST", "JNJ", "HD", "ABBV",
    "WMT", "NFLX", "BAC", "CRM", "CVX", "MRK", "KO", "ORCL", "AMD", "PEP",
    "ACN", "TMO", "LIN", "MCD", "CSCO", "ADBE", "ABT", "WFC", "PM", "IBM",
    "GE", "CAT", "ISRG", "INTU", "TXN", "QCOM", "VZ", "CMCSA", "DHR", "NOW",
    "AMGN", "PFE", "SPGI", "T", "UNP", "NEE", "RTX", "HON", "LOW", "UPS",
    "GS", "BLK", "ELV", "BKNG", "SYK", "AXP", "SBUX", "LMT", "MDLZ", "DE",
    "GILD", "MMC", "PLD", "ADI", "CB", "VRTX", "SCHW", "TMUS", "AMT", "CI",
    "MO", "DUK", "SO", "BMY", "CME", "CL", "ICE", "ZTS", "SLB", "BDX",
    "EOG", "APD", "FDX", "NOC", "ITW", "PNC", "USB", "TGT", "MMM", "F",
]

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    config = AlphaLabConfig.from_yaml("strategies/graham_value/configs/default.yaml")
    runner = BacktestRunner(config)

    results = []
    errors = []

    total = len(TOP_100)
    for i, ticker in enumerate(TOP_100, 1):
        print(f"[{i:3d}/{total}] {ticker:6s} ... ", end="", flush=True)
        try:
            result = runner.run(ticker)
            stats = result["stats"]
            signal = result["combined_signal"]
            row = {
                "ticker": ticker,
                "return_pct": stats["Return [%]"],
                "annual_return_pct": stats["Return (Ann.) [%]"],
                "sharpe": stats["Sharpe Ratio"],
                "max_dd_pct": stats["Max. Drawdown [%]"],
                "trades": stats["# Trades"],
                "win_rate": stats["Win Rate [%]"],
                "equity_final": stats["Equity Final [$]"],
                "avg_signal": signal.mean(),
                "long_days": (signal > 0).sum(),
            }
            results.append(row)
            ret = row["return_pct"]
            ann = row["annual_return_pct"]
            trades = row["trades"]
            print(f"Return: {ret:+7.2f}% ({ann:+.2f}%/yr)  Trades: {trades:3d}  Sharpe: {row['sharpe']:.3f}")
        except Exception as e:
            errors.append({"ticker": ticker, "error": str(e)})
            print(f"ERROR: {e}")

        # Small delay to avoid rate limiting
        time.sleep(0.1)

    # Summary
    df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("  BATCH BACKTEST SUMMARY — Graham Long-Only Strategy (2019-2024)")
    print("=" * 80)
    print(f"  Stocks tested: {len(results)}/{total}")
    print(f"  Errors: {len(errors)}")

    if df.empty:
        print("  No results.")
        return

    traded = df[df["trades"] > 0]
    no_trade = df[df["trades"] == 0]

    print(f"  Stocks with trades: {len(traded)}")
    print(f"  Stocks with 0 trades (never cheap enough): {len(no_trade)}")

    if not traded.empty:
        print(f"\n  --- Stocks that traded ---")
        print(f"  Avg Return:        {traded['return_pct'].mean():+.2f}%")
        print(f"  Avg Annual Return: {traded['annual_return_pct'].mean():+.2f}%/yr")
        print(f"  Median Return:     {traded['return_pct'].median():+.2f}%")
        print(f"  Avg Sharpe:        {traded['sharpe'].mean():.3f}")
        print(f"  Avg Max DD:        {traded['max_dd_pct'].mean():.2f}%")
        print(f"  Avg Trades:        {traded['trades'].mean():.1f}")
        print(f"  Avg Win Rate:      {traded['win_rate'].mean():.1f}%")

        profitable = traded[traded["return_pct"] > 0]
        print(f"\n  Profitable: {len(profitable)}/{len(traded)} ({100*len(profitable)/len(traded):.0f}%)")

        print(f"\n  --- Top 10 by Return ---")
        top10 = traded.nlargest(10, "return_pct")
        for _, r in top10.iterrows():
            print(f"  {r['ticker']:6s}  Return: {r['return_pct']:+7.2f}% ({r['annual_return_pct']:+.2f}%/yr)  "
                  f"Sharpe: {r['sharpe']:.3f}  Trades: {int(r['trades']):3d}  "
                  f"WinRate: {r['win_rate']:.0f}%")

        print(f"\n  --- Bottom 10 by Return ---")
        bot10 = traded.nsmallest(10, "return_pct")
        for _, r in bot10.iterrows():
            print(f"  {r['ticker']:6s}  Return: {r['return_pct']:+7.2f}% ({r['annual_return_pct']:+.2f}%/yr)  "
                  f"Sharpe: {r['sharpe']:.3f}  Trades: {int(r['trades']):3d}  "
                  f"WinRate: {r['win_rate']:.0f}%")

    if errors:
        print(f"\n  --- Errors ---")
        for e in errors:
            print(f"  {e['ticker']:6s}: {e['error']}")

    print("=" * 80)

    # Save full results into timestamped subfolder
    from datetime import datetime
    from pathlib import Path
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    method = "graham"
    run_dir = Path(__file__).parent / "reports" / f"{method}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(run_dir / "batch_backtest_results.csv", index=False)
    print(f"\nFull results saved to {run_dir}/batch_backtest_results.csv")


if __name__ == "__main__":
    main()
