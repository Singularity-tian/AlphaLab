"""Portfolio backtest: single $100k account across top 100 stocks."""

from __future__ import annotations

import time
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import pandas as pd

from alphalab.config import AlphaLabConfig
from alphalab.backtest.portfolio_runner import PortfolioBacktestRunner
from alphalab.backtest.portfolio_report import generate_html_report

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
    config = AlphaLabConfig.from_yaml("configs/default.yaml")
    runner = PortfolioBacktestRunner(config)

    print("=" * 80)
    print("  PORTFOLIO BACKTEST — Graham Value Strategy (Dynamic Signals)")
    print(f"  Initial Capital: ${config.backtest.initial_cash:,.0f}")
    print(f"  Universe: {len(TOP_100)} stocks")
    print(f"  Period: {config.backtest.start_date} to {config.backtest.end_date}")
    print(f"  Trailing Stop: {config.backtest.trailing_stop_pct*100:.0f}%")
    print(f"  Holding Period: {config.backtest.holding_period_days} days")
    print(f"  Max Positions: {config.backtest.max_positions}")
    print(f"  Signal Weighted: {config.backtest.signal_weighted}")
    print(f"  Scale-in: {config.backtest.scale_in} ({config.backtest.scale_in_pct*100:.0f}% initial)")
    print("=" * 80)
    print()

    print("Computing signals for all stocks (this may take a few minutes)...")
    start_time = time.time()
    result = runner.run(TOP_100)
    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.1f}s\n")

    stats = result["stats"]
    equity_curve = result["equity_curve"]
    trades_df = result["trades"]
    daily = result["daily"]
    per_stock_pnl = result["per_stock_pnl"]

    # Portfolio summary
    print("=" * 80)
    print("  PORTFOLIO PERFORMANCE")
    print("=" * 80)
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key:<30} {value}")
        else:
            print(f"  {key:<30} {value:>15}")

    # Position statistics
    print()
    print(f"  {'Avg Positions Held':<30} {daily['num_positions'].mean():>15.1f}")
    print(f"  {'Max Positions Held':<30} {daily['num_positions'].max():>15d}")
    print(f"  {'Min Positions Held':<30} {daily['num_positions'].min():>15d}")

    # Top/bottom contributors
    if not per_stock_pnl.empty:
        print()
        print("  --- Top 10 Contributors (Total P&L) ---")
        top10 = per_stock_pnl.head(10)
        for _, r in top10.iterrows():
            open_tag = f"  [open: {int(r['open_shares'])} sh]" if r['open_shares'] > 0 else ""
            print(f"  {r['ticker']:<6}  P&L: ${r['total_pnl']:>+10,.2f}  "
                  f"Buys: {int(r['num_buys']):>3}  Sells: {int(r['num_sells']):>3}{open_tag}")

        print()
        print("  --- Bottom 10 Contributors (Total P&L) ---")
        bot10 = per_stock_pnl.tail(10)
        for _, r in bot10.iterrows():
            open_tag = f"  [open: {int(r['open_shares'])} sh]" if r['open_shares'] > 0 else ""
            print(f"  {r['ticker']:<6}  P&L: ${r['total_pnl']:>+10,.2f}  "
                  f"Buys: {int(r['num_buys']):>3}  Sells: {int(r['num_sells']):>3}{open_tag}")

    # Recent trades
    if not trades_df.empty:
        print()
        print(f"  --- Last 20 Trades ---")
        print(f"  {'Date':<12} {'Ticker':<6} {'Side':<5} {'Reason':<14} {'Shares':>6} "
              f"{'Price':>10} {'Value':>12} {'Commission':>10}")
        print("  " + "-" * 84)
        recent = trades_df.tail(20)
        for _, t in recent.iterrows():
            reason = t.get('reason', '')
            print(
                f"  {str(t['date'])[:10]:<12} {t['ticker']:<6} {t['side']:<5} "
                f"{reason:<14} {int(t['shares']):>6} ${t['price']:>9.2f} "
                f"${t['value']:>11,.2f} ${t['commission']:>9.2f}"
            )

    print("=" * 80)

    # Save outputs
    Path("reports").mkdir(exist_ok=True)
    equity_curve.to_csv("reports/portfolio_equity_curve.csv", header=True)
    if not trades_df.empty:
        trades_df.to_csv("reports/portfolio_trades.csv", index=False)
    if not per_stock_pnl.empty:
        per_stock_pnl.to_csv("reports/portfolio_per_stock_pnl.csv", index=False)

    # Generate HTML report
    config_summary = {
        "initial_cash": f"{config.backtest.initial_cash:,.0f}",
        "period": f"{config.backtest.start_date} to {config.backtest.end_date}",
        "universe": str(len(TOP_100)),
        "commission": f"{config.backtest.commission*100:.1f}%",
        "factors": ", ".join(config.factors.factors),
        "trailing_stop_pct": f"{config.backtest.trailing_stop_pct*100:.0f}%",
        "holding_period_days": str(config.backtest.holding_period_days),
        "max_positions": str(config.backtest.max_positions),
        "max_position_weight": f"{config.backtest.max_position_weight*100:.0f}%",
    }
    html_path = generate_html_report(result, config_summary)

    print(f"\n  Files saved:")
    print(f"    reports/portfolio_equity_curve.csv")
    print(f"    reports/portfolio_trades.csv")
    print(f"    reports/portfolio_per_stock_pnl.csv")
    print(f"    {html_path}  ← open in browser")
    print("=" * 80)


if __name__ == "__main__":
    main()
