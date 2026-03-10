"""Run EXPMA(12) & KDJ hourly strategy backtest on S&P 500.

Usage:
    source .venv/bin/activate
    python "expma&kdj/run_backtest.py"
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from examples.portfolio_backtest import SP500
from strategy import run_portfolio
from report import generate_html_report

DATA_DIR = Path(__file__).resolve().parent / "sp500_hourly_data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EXPMA(12) & KDJ Hourly Strategy — S&P 500 Backtest")
    print("=" * 60)
    print()

    start = time.time()
    results = run_portfolio(
        data_dir=DATA_DIR,
        tickers=SP500,
        initial_cash=100_000.0,
        commission=0.001,
        max_positions=20,
        position_pct=0.05,
        stop_loss_pct=0.05,
    )
    elapsed = time.time() - start

    # --- Print stats ---
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    stats = results["stats"]
    for k, v in stats.items():
        if k == "Sell Reasons":
            print(f"  {k}:")
            for reason, count in v.items():
                print(f"    {reason}: {count}")
        else:
            print(f"  {k}: {v}")

    print(f"\n  Simulation time: {elapsed:.1f}s")

    # --- Save results ---
    equity = results["equity_curve"]
    equity.to_csv(RESULTS_DIR / "equity_curve.csv")
    print(f"\n  Saved equity curve → {RESULTS_DIR / 'equity_curve.csv'}")

    trades = results["trades"]
    if not trades.empty:
        trades.to_csv(RESULTS_DIR / "trades.csv", index=False)
        print(f"  Saved trades ({len(trades)} records) → {RESULTS_DIR / 'trades.csv'}")

    pnl = results["per_stock_pnl"]
    if not pnl.empty:
        pnl.to_csv(RESULTS_DIR / "per_stock_pnl.csv", index=False)
        print(f"  Saved per-stock P&L → {RESULTS_DIR / 'per_stock_pnl.csv'}")

        # Top/bottom performers
        print("\n  Top 5 stocks by P&L:")
        for _, row in pnl.head(5).iterrows():
            print(f"    {row['ticker']:6s} ${row['total_pnl']:>10,.2f}")
        print("\n  Bottom 5 stocks by P&L:")
        for _, row in pnl.tail(5).iterrows():
            print(f"    {row['ticker']:6s} ${row['total_pnl']:>10,.2f}")

    # --- Generate HTML report ---
    report_path = generate_html_report(
        result=results,
        output_path=RESULTS_DIR / "report.html",
        initial_cash=100_000.0,
        max_positions=20,
        position_pct=0.05,
        stop_loss_pct=0.05,
    )
    print(f"  Saved HTML report → {report_path}")

    print()


if __name__ == "__main__":
    main()
