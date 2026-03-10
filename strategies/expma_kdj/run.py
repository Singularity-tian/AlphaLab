"""Run EXPMA(12) & KDJ hourly strategy backtest on S&P 500.

Usage:
    source .venv/bin/activate
    python strategies/expma_kdj/run.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import yaml

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from alphalab.data.tickers import SP500
from strategies.expma_kdj.strategy import run_portfolio
from strategies.expma_kdj.report import generate_html_report

STRATEGY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STRATEGY_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "sp500_hourly" / "cache"
REPORTS_DIR = STRATEGY_DIR / "reports"
CONFIG_PATH = STRATEGY_DIR / "configs" / "default.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config().get("backtest", {})

    initial_cash = cfg.get("initial_cash", 100_000.0)
    commission = cfg.get("commission", 0.001)
    max_positions = cfg.get("max_positions", 20)
    position_pct = cfg.get("position_pct", 0.05)
    stop_loss_pct = cfg.get("stop_loss_pct", 0.05)

    print("=" * 60)
    print("EXPMA(12) & KDJ Hourly Strategy — S&P 500 Backtest")
    print("=" * 60)
    print()

    start = time.time()
    results = run_portfolio(
        data_dir=DATA_DIR,
        tickers=SP500,
        initial_cash=initial_cash,
        commission=commission,
        max_positions=max_positions,
        position_pct=position_pct,
        stop_loss_pct=stop_loss_pct,
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
    equity.to_csv(REPORTS_DIR / "equity_curve.csv")
    print(f"\n  Saved equity curve → {REPORTS_DIR / 'equity_curve.csv'}")

    trades = results["trades"]
    if not trades.empty:
        trades.to_csv(REPORTS_DIR / "trades.csv", index=False)
        print(f"  Saved trades ({len(trades)} records) → {REPORTS_DIR / 'trades.csv'}")

    pnl = results["per_stock_pnl"]
    if not pnl.empty:
        pnl.to_csv(REPORTS_DIR / "per_stock_pnl.csv", index=False)
        print(f"  Saved per-stock P&L → {REPORTS_DIR / 'per_stock_pnl.csv'}")

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
        output_path=REPORTS_DIR / "report.html",
        initial_cash=initial_cash,
        max_positions=max_positions,
        position_pct=position_pct,
        stop_loss_pct=stop_loss_pct,
    )
    print(f"  Saved HTML report → {report_path}")

    print()


if __name__ == "__main__":
    main()
