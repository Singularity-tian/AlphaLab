"""Batch backtest with full trade-level detail for every stock."""

from __future__ import annotations

import time
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import pandas as pd

from alphalab.config import AlphaLabConfig
from alphalab.backtest.runner import BacktestRunner

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


def main():
    config = AlphaLabConfig.from_yaml("configs/default.yaml")
    initial_cash = config.backtest.initial_cash
    runner = BacktestRunner(config)

    all_trades = []
    stock_summaries = []
    total = len(TOP_100)

    for i, ticker in enumerate(TOP_100, 1):
        print(f"[{i:3d}/{total}] {ticker:6s} ... ", end="", flush=True)
        try:
            result = runner.run(ticker)
            stats = result["stats"]
            trades = result["trades"]
            signal = result["combined_signal"]
            equity_curve = result["equity_curve"]

            stock_summaries.append({
                "ticker": ticker,
                "return_pct": stats["Return [%]"],
                "annual_return_pct": stats["Return (Ann.) [%]"],
                "sharpe": stats["Sharpe Ratio"],
                "max_dd_pct": stats["Max. Drawdown [%]"],
                "num_trades": stats["# Trades"],
                "win_rate": stats["Win Rate [%]"],
                "equity_final": stats["Equity Final [$]"],
                "total_pnl": stats["Equity Final [$]"] - initial_cash,
                "days_in_market": int((signal > 0).sum()),
                "days_total": len(signal),
                "pct_time_in_market": 100 * (signal > 0).sum() / len(signal),
            })

            # Build detailed trade records
            running_equity = initial_cash
            for j, trade in trades.iterrows():
                shares = int(trade["Size"])
                entry_price = trade["EntryPrice"]
                exit_price = trade["ExitPrice"]
                pnl = trade["PnL"]
                position_value = abs(shares) * entry_price
                cash_after_buy = running_equity - position_value
                equity_after_sell = running_equity + pnl

                # Get signal at entry
                entry_signal = 0.0
                sig_before = signal.loc[:trade["EntryTime"]]
                if not sig_before.empty:
                    entry_signal = sig_before.iloc[-1]

                all_trades.append({
                    "ticker": ticker,
                    "trade_num": j + 1,
                    "action_buy": "BUY",
                    "entry_date": trade["EntryTime"].strftime("%Y-%m-%d"),
                    "entry_price": round(entry_price, 2),
                    "shares": abs(shares),
                    "position_value": round(position_value, 2),
                    "signal_strength": round(entry_signal, 3),
                    "equity_before_buy": round(running_equity, 2),
                    "cash_after_buy": round(cash_after_buy, 2),
                    "action_sell": "SELL",
                    "exit_date": trade["ExitTime"].strftime("%Y-%m-%d"),
                    "exit_price": round(exit_price, 2),
                    "sell_proceeds": round(abs(shares) * exit_price, 2),
                    "trade_pnl": round(pnl, 2),
                    "trade_return_pct": round(trade["ReturnPct"] * 100, 2),
                    "equity_after_sell": round(equity_after_sell, 2),
                    "hold_days": (trade["ExitTime"] - trade["EntryTime"]).days,
                })

                running_equity = equity_after_sell

            n_trades = stats["# Trades"]
            ret = stats["Return [%]"]
            print(f"Trades: {n_trades:2d}  Return: {ret:+7.2f}%")

        except Exception as e:
            stock_summaries.append({"ticker": ticker, "return_pct": None, "error": str(e)})
            print(f"ERROR: {e}")

        time.sleep(0.1)

    # Save trades CSV into timestamped subfolder
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    method = "graham"
    run_dir = Path(f"reports/{method}_{ts}")
    run_dir.mkdir(parents=True, exist_ok=True)

    trades_df = pd.DataFrame(all_trades)
    trades_df.to_csv(run_dir / "batch_all_trades.csv", index=False)

    # Save stock summaries
    summary_df = pd.DataFrame(stock_summaries)
    summary_df.to_csv(run_dir / "batch_stock_summaries.csv", index=False)

    # Print all trades
    print("\n" + "=" * 120)
    print("  ALL TRADES — Graham Long-Only Strategy (2019-2024)")
    print("=" * 120)
    print(f"  {'Ticker':<6} {'#':>2}  {'Entry Date':>10}  {'Buy @':>8}  {'Shares':>6}  "
          f"{'Position $':>11}  {'Signal':>6}  {'Cash Left':>10}  "
          f"{'Exit Date':>10}  {'Sell @':>8}  {'P&L':>10}  {'Ret%':>7}  {'Days':>4}  {'Equity After':>12}")
    print("-" * 120)

    for t in all_trades:
        print(f"  {t['ticker']:<6} {t['trade_num']:>2}  {t['entry_date']:>10}  "
              f"${t['entry_price']:>7.2f}  {t['shares']:>6}  "
              f"${t['position_value']:>10,.2f}  {t['signal_strength']:>6.3f}  "
              f"${t['cash_after_buy']:>9,.2f}  "
              f"{t['exit_date']:>10}  ${t['exit_price']:>7.2f}  "
              f"${t['trade_pnl']:>+9,.2f}  {t['trade_return_pct']:>+6.2f}%  "
              f"{t['hold_days']:>4}  ${t['equity_after_sell']:>11,.2f}")

    print("=" * 120)

    # Aggregate stats
    traded = [s for s in stock_summaries if s.get("num_trades", 0) and s["num_trades"] > 0]
    no_trade = [s for s in stock_summaries if s.get("num_trades") == 0]

    print(f"\n  SUMMARY")
    print(f"  Total stocks: {total}")
    print(f"  Stocks that traded: {len(traded)}")
    print(f"  Stocks with 0 trades: {len(no_trade)}")
    print(f"  Total trades: {len(all_trades)}")

    if all_trades:
        total_pnl = sum(t["trade_pnl"] for t in all_trades)
        winners = [t for t in all_trades if t["trade_pnl"] > 0]
        losers = [t for t in all_trades if t["trade_pnl"] <= 0]
        print(f"  Winning trades: {len(winners)}/{len(all_trades)} ({100*len(winners)/len(all_trades):.0f}%)")
        print(f"  Total P&L (sum across all stocks): ${total_pnl:+,.2f}")
        print(f"  Avg P&L per trade: ${total_pnl/len(all_trades):+,.2f}")
        if winners:
            print(f"  Avg winner: ${sum(t['trade_pnl'] for t in winners)/len(winners):+,.2f}")
        if losers:
            print(f"  Avg loser:  ${sum(t['trade_pnl'] for t in losers)/len(losers):+,.2f}")
        print(f"  Avg hold period: {sum(t['hold_days'] for t in all_trades)/len(all_trades):.0f} days")

    print(f"\n  Files saved to {run_dir}/:")
    print(f"    batch_all_trades.csv      — every trade with full accounting")
    print(f"    batch_stock_summaries.csv  — per-stock summary")
    print("=" * 120)


if __name__ == "__main__":
    main()
