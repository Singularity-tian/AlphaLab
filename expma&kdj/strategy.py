"""EXPMA(12) & KDJ hourly trading strategy with portfolio-level backtest.

Strategy rules:
  Buy (50%):  EXPMA(12) > EXPMA(50) AND J_{n-1} < 0 AND J_n > J_{n-1}
  Add (50%):  Next bar after buy, Close_n > Close_{n-1}
  Sell (all): EXPMA(12) < EXPMA(50) OR price <= avg_cost * 0.95
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------

def compute_indicators(df: pd.DataFrame, kdj_period: int = 9) -> pd.DataFrame:
    """Add EXPMA and KDJ columns to hourly OHLC DataFrame.

    Adds: expma12, expma50, K, D, J
    """
    df = df.copy()

    # EXPMA (Exponential Moving Average)
    df["expma12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["expma50"] = df["close"].ewm(span=50, adjust=False).mean()

    # KDJ
    low_n = df["low"].rolling(window=kdj_period).min()
    high_n = df["high"].rolling(window=kdj_period).max()
    rsv = (df["close"] - low_n) / (high_n - low_n) * 100
    rsv = rsv.fillna(50.0)  # neutral when high == low

    # Smoothed K and D (2/3 previous + 1/3 new)
    k_values = np.empty(len(df))
    d_values = np.empty(len(df))
    k_values[0] = 50.0
    d_values[0] = 50.0

    rsv_arr = rsv.values
    for i in range(1, len(df)):
        k_values[i] = 2.0 / 3.0 * k_values[i - 1] + 1.0 / 3.0 * rsv_arr[i]
        d_values[i] = 2.0 / 3.0 * d_values[i - 1] + 1.0 / 3.0 * k_values[i]

    df["K"] = k_values
    df["D"] = d_values
    df["J"] = 3.0 * df["K"] - 2.0 * df["D"]

    return df


# ---------------------------------------------------------------
# Position tracking
# ---------------------------------------------------------------

@dataclass
class Position:
    ticker: str
    shares: int
    avg_cost: float
    fully_invested: bool  # True after add-on buy
    entry_bar: int  # bar index of initial buy
    allocated_capital: float  # total capital allocated (for add-on sizing)


# ---------------------------------------------------------------
# Portfolio simulation
# ---------------------------------------------------------------

def run_portfolio(
    data_dir: str | Path,
    tickers: list[str],
    initial_cash: float = 100_000.0,
    commission: float = 0.001,
    max_positions: int = 20,
    position_pct: float = 0.05,
    stop_loss_pct: float = 0.05,
) -> dict[str, Any]:
    """Run EXPMA+KDJ strategy across multiple stocks with shared capital.

    Args:
        data_dir: Path to folder with {ticker}.parquet files.
        tickers: List of ticker symbols.
        initial_cash: Starting capital.
        commission: Commission rate per trade (e.g. 0.001 = 0.1%).
        max_positions: Max concurrent positions (default 20 = 5% each).
        position_pct: Fraction of portfolio per position (default 0.05).
        stop_loss_pct: Hard stop-loss from avg cost (default 0.05 = 5%).

    Returns:
        Dict with equity_curve, trades, stats, per_stock_pnl.
    """
    data_dir = Path(data_dir)

    # --- Load and prepare all data ---
    print(f"Loading data for {len(tickers)} tickers...")
    all_data: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        path = data_dir / f"{ticker}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if len(df) < 50:  # need enough bars for EXPMA(50)
            continue
        df = compute_indicators(df)
        all_data[ticker] = df

    valid_tickers = list(all_data.keys())
    print(f"Loaded {len(valid_tickers)} tickers with sufficient data")

    if not valid_tickers:
        raise ValueError("No valid tickers with data")

    # --- Build unified hourly timeline ---
    all_dates = sorted(set().union(*(df.index for df in all_data.values())))
    print(f"Simulation: {len(all_dates)} hourly bars from {all_dates[0]} to {all_dates[-1]}")

    # --- Pre-build numpy arrays for fast access ---
    # For each ticker: close prices, indicator values indexed by position in all_dates
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    n_bars = len(all_dates)

    ticker_arrays: dict[str, dict[str, np.ndarray]] = {}
    for ticker, df in all_data.items():
        arrays: dict[str, np.ndarray] = {}
        for col in ["close", "open", "expma12", "expma50", "J"]:
            arr = np.full(n_bars, np.nan)
            for dt, val in df[col].items():
                idx = date_to_idx.get(dt)
                if idx is not None:
                    arr[idx] = val
            arrays[col] = arr
        ticker_arrays[ticker] = arrays

    # --- Simulation ---
    cash = initial_cash
    positions: dict[str, Position] = {}
    trades: list[dict[str, Any]] = []
    hourly_records: list[dict[str, Any]] = []
    pending_addon: dict[str, int] = {}  # ticker -> bar index where addon should happen

    print("Running simulation...")
    for i in range(1, n_bars):  # start at 1 so we can look back
        dt = all_dates[i]

        # --- Step 1: Check sell conditions ---
        for ticker in list(positions):
            pos = positions[ticker]
            arrs = ticker_arrays[ticker]
            close_now = arrs["close"][i]
            open_now = arrs["open"][i]

            if np.isnan(close_now) or np.isnan(open_now):
                continue

            # Use open price for sell execution (next bar open rule)
            sell_price = open_now
            sell_reason = None

            # Trend break: EXPMA(12) < EXPMA(50) on previous bar → sell at this bar's open
            expma12_prev = arrs["expma12"][i - 1]
            expma50_prev = arrs["expma50"][i - 1]
            if not np.isnan(expma12_prev) and not np.isnan(expma50_prev):
                if expma12_prev < expma50_prev:
                    sell_reason = "TREND_BREAK"

            # Hard stop: price <= avg_cost * (1 - stop_loss_pct)
            if sell_reason is None and open_now <= pos.avg_cost * (1 - stop_loss_pct):
                sell_reason = "STOP_LOSS"

            if sell_reason:
                proceeds = pos.shares * sell_price
                comm = proceeds * commission
                cash += proceeds - comm
                trades.append(_trade_record(
                    dt, ticker, "SELL", pos.shares, sell_price,
                    proceeds, comm, cash, sell_reason,
                ))
                del positions[ticker]
                pending_addon.pop(ticker, None)

        # --- Step 2: Process add-on buys ---
        for ticker in list(pending_addon):
            if ticker not in positions:
                pending_addon.pop(ticker, None)
                continue
            if pending_addon[ticker] != i:
                continue

            pos = positions[ticker]
            arrs = ticker_arrays[ticker]
            close_now = arrs["close"][i]
            close_prev = arrs["close"][i - 1]

            if np.isnan(close_now) or np.isnan(close_prev):
                pending_addon.pop(ticker, None)
                pos.fully_invested = True
                continue

            if close_now > close_prev:
                # Add remaining 50%
                remaining_capital = pos.allocated_capital - (pos.shares * pos.avg_cost)
                if remaining_capital > 0 and close_now > 0:
                    addon_shares = math.floor(remaining_capital / (close_now * (1 + commission)))
                    if addon_shares > 0:
                        cost = addon_shares * close_now
                        comm = cost * commission
                        if cost + comm <= cash:
                            # Update avg cost
                            total_cost = pos.shares * pos.avg_cost + cost
                            pos.shares += addon_shares
                            pos.avg_cost = total_cost / pos.shares
                            cash -= cost + comm
                            trades.append(_trade_record(
                                dt, ticker, "BUY", addon_shares, close_now,
                                cost, comm, cash, "ADDON",
                            ))

            pos.fully_invested = True
            pending_addon.pop(ticker, None)

        # --- Step 3: Check new buy signals ---
        if len(positions) < max_positions:
            buy_candidates: list[tuple[str, float]] = []

            for ticker in valid_tickers:
                if ticker in positions:
                    continue
                arrs = ticker_arrays[ticker]
                close_now = arrs["close"][i]
                expma12_now = arrs["expma12"][i]
                expma50_now = arrs["expma50"][i]
                j_now = arrs["J"][i]
                j_prev = arrs["J"][i - 1]

                if any(np.isnan(v) for v in [close_now, expma12_now, expma50_now, j_now, j_prev]):
                    continue

                # Buy conditions: EXPMA(12) > EXPMA(50) AND J_prev < 0 AND J_now > J_prev
                if expma12_now > expma50_now and j_prev < 0 and j_now > j_prev:
                    buy_candidates.append((ticker, close_now))

            # Sort by... just take first available (or could sort by J momentum)
            available_slots = max_positions - len(positions)
            for ticker, price in buy_candidates[:available_slots]:
                portfolio_value = cash + _mark_to_market(positions, ticker_arrays, i)
                alloc = portfolio_value * position_pct

                # Buy 50% initially
                initial_alloc = alloc * 0.5
                shares = math.floor(initial_alloc / (price * (1 + commission)))
                if shares <= 0:
                    continue

                cost = shares * price
                comm = cost * commission
                if cost + comm > cash:
                    shares = math.floor(cash / (price * (1 + commission)))
                    if shares <= 0:
                        continue
                    cost = shares * price
                    comm = cost * commission
                    alloc = cost * 2  # adjust allocation

                cash -= cost + comm
                positions[ticker] = Position(
                    ticker=ticker,
                    shares=shares,
                    avg_cost=price,
                    fully_invested=False,
                    entry_bar=i,
                    allocated_capital=alloc,
                )
                trades.append(_trade_record(
                    dt, ticker, "BUY", shares, price,
                    cost, comm, cash, "SIGNAL_ENTRY",
                ))
                # Schedule add-on for next bar
                pending_addon[ticker] = i + 1

                if len(positions) >= max_positions:
                    break

        # --- Mark-to-market ---
        position_value = _mark_to_market(positions, ticker_arrays, i)
        portfolio_value = cash + position_value
        hourly_records.append({
            "date": dt,
            "portfolio_value": portfolio_value,
            "cash": cash,
            "position_value": position_value,
            "num_positions": len(positions),
        })

    # --- Build results ---
    daily_df = pd.DataFrame(hourly_records).set_index("date")
    equity_curve = daily_df["portfolio_value"]
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    final_holdings = {t: p.shares for t, p in positions.items()}
    final_prices = {}
    for ticker in valid_tickers:
        arrs = ticker_arrays[ticker]
        last_valid = arrs["close"][~np.isnan(arrs["close"])]
        if len(last_valid) > 0:
            final_prices[ticker] = last_valid[-1]
    final_prices_series = pd.Series(final_prices)

    stats = _compute_stats(equity_curve, trades_df, initial_cash)
    per_stock_pnl = _compute_per_stock_pnl(trades_df, final_holdings, final_prices_series)

    return {
        "equity_curve": equity_curve,
        "daily": daily_df,
        "trades": trades_df,
        "stats": stats,
        "per_stock_pnl": per_stock_pnl,
    }


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _mark_to_market(
    positions: dict[str, Position],
    ticker_arrays: dict[str, dict[str, np.ndarray]],
    bar_idx: int,
) -> float:
    value = 0.0
    for ticker, pos in positions.items():
        arrs = ticker_arrays.get(ticker)
        if arrs is None:
            continue
        price = arrs["close"][bar_idx]
        if not np.isnan(price):
            value += pos.shares * price
    return value


def _trade_record(
    date, ticker, side, shares, price, value, commission, cash_after, reason,
) -> dict[str, Any]:
    cash_impact = (value - commission) if side == "SELL" else -(value + commission)
    return {
        "date": date, "ticker": ticker, "side": side,
        "shares": shares, "price": price, "value": value,
        "commission": commission,
        "cash_impact": cash_impact, "cash_after": cash_after,
        "reason": reason,
    }


def _compute_stats(
    equity_curve: pd.Series,
    trades_df: pd.DataFrame,
    initial_cash: float,
) -> dict[str, Any]:
    final_value = equity_curve.iloc[-1]
    total_return = (final_value / initial_cash - 1) * 100

    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    years = days / 365.25
    ann_return = ((final_value / initial_cash) ** (1 / years) - 1) * 100 if years > 0 else 0

    # Use daily resampled returns for Sharpe/Sortino (hourly is too noisy)
    daily_equity = equity_curve.resample("D").last().dropna()
    daily_returns = daily_equity.pct_change().dropna()

    sharpe = 0.0
    sortino = 0.0
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        downside = daily_returns[daily_returns < 0]
        if len(downside) > 0 and downside.std() > 0:
            sortino = daily_returns.mean() / downside.std() * np.sqrt(252)

    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax * 100
    max_dd = drawdown.min()

    num_trades = len(trades_df)
    total_commission = trades_df["commission"].sum() if num_trades > 0 else 0

    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0

    # Profit factor
    profit_factor = 0.0
    if not trades_df.empty:
        gross_wins = 0.0
        gross_losses = 0.0
        for ticker in trades_df["ticker"].unique():
            tt = trades_df[trades_df["ticker"] == ticker]
            buys = tt[tt["side"] == "BUY"]
            sells = tt[tt["side"] == "SELL"]
            cost = buys["value"].sum() + buys["commission"].sum()
            proceeds = sells["value"].sum() - sells["commission"].sum()
            pnl = proceeds - cost
            if pnl > 0:
                gross_wins += pnl
            elif pnl < 0:
                gross_losses += abs(pnl)
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else (
            10.0 if gross_wins > 0 else 0.0
        )

    # Sell reason breakdown
    sell_reasons = {}
    if not trades_df.empty:
        sell_trades = trades_df[trades_df["side"] == "SELL"]
        if not sell_trades.empty:
            sell_reasons = sell_trades["reason"].value_counts().to_dict()

    return {
        "Total Return [%]": round(total_return, 2),
        "Annualized Return [%]": round(ann_return, 2),
        "Sharpe Ratio": round(sharpe, 3),
        "Sortino Ratio": round(sortino, 3),
        "Calmar Ratio": round(calmar, 3),
        "Profit Factor": round(profit_factor, 3),
        "Max Drawdown [%]": round(max_dd, 2),
        "Equity Final [$]": round(final_value, 2),
        "# Trades": num_trades,
        "Total Commission [$]": round(total_commission, 2),
        "Duration [days]": days,
        "Sell Reasons": sell_reasons,
    }


def _compute_per_stock_pnl(
    trades_df: pd.DataFrame,
    final_holdings: dict[str, int],
    final_prices: pd.Series,
) -> pd.DataFrame:
    all_tickers = set()
    if not trades_df.empty:
        all_tickers.update(trades_df["ticker"].unique())
    all_tickers.update(t for t, s in final_holdings.items() if s > 0)

    if not all_tickers:
        return pd.DataFrame()

    pnl_records = []
    for ticker in all_tickers:
        if not trades_df.empty:
            tt = trades_df[trades_df["ticker"] == ticker]
            buys = tt[tt["side"] == "BUY"]
            sells = tt[tt["side"] == "SELL"]
            total_cost = buys["value"].sum() + buys["commission"].sum()
            total_proceeds = sells["value"].sum() - sells["commission"].sum()
        else:
            total_cost = 0.0
            total_proceeds = 0.0
            buys = sells = pd.DataFrame()

        open_shares = final_holdings.get(ticker, 0)
        unrealized_value = 0.0
        if open_shares > 0:
            price = final_prices.get(ticker)
            if price is not None and not np.isnan(price):
                unrealized_value = open_shares * price

        total_pnl = total_proceeds + unrealized_value - total_cost
        pnl_records.append({
            "ticker": ticker,
            "total_cost": round(total_cost, 2),
            "total_proceeds": round(total_proceeds, 2),
            "unrealized_value": round(unrealized_value, 2),
            "total_pnl": round(total_pnl, 2),
            "open_shares": open_shares,
            "num_buys": len(buys),
            "num_sells": len(sells),
        })

    return pd.DataFrame(pnl_records).sort_values("total_pnl", ascending=False)
