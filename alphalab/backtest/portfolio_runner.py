"""Portfolio-level backtest: single account across multiple stocks.

Strategy:
- Daily signal check: buy when combined score > 0.6
- Signal-weighted position sizing with max 25% per stock
- Fixed holding period (252 trading days) then auto-sell
- 20% trailing stop-loss (no cooldown)
- Scale-in: 50% on entry, remaining 50% after 5 days if signal holds
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from alphalab.combiner.equal_weight import EqualWeightCombiner
from alphalab.config import AlphaLabConfig
from alphalab.data.providers import DataProvider
from alphalab.factors.base import Factor
from alphalab.factors.registry import get_factor

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Track a single open position."""

    ticker: str
    shares: int
    entry_day: int  # index into dates array
    entry_price: float
    trailing_high: float
    target_shares: int  # full target (for scale-in)
    scaled_in: bool = False  # whether scale-in is complete


class PortfolioBacktestRunner:
    """Single-account backtest across multiple stocks."""

    def __init__(self, config: AlphaLabConfig):
        self.config = config
        self.data = DataProvider(config)
        self.combiner = EqualWeightCombiner(config, self.data)

    def run(
        self,
        tickers: list[str],
        factor_names: list[str] | None = None,
    ) -> dict[str, Any]:
        if factor_names is None:
            factor_names = self.config.factors.factors

        factors: list[Factor] = []
        for name in factor_names:
            try:
                factors.append(get_factor(name))
            except KeyError as e:
                logger.warning(str(e))
        if not factors:
            raise ValueError("No valid factors found.")

        prices, signals, valid_tickers = self._precompute(tickers, factors)
        if not valid_tickers:
            raise ValueError("No tickers with valid data.")

        logger.info(
            "Portfolio backtest: %d tickers, %d trading days",
            len(valid_tickers), len(prices),
        )
        return self._simulate(prices, signals, valid_tickers)

    def _precompute(
        self,
        tickers: list[str],
        factors: list[Factor],
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        price_dict: dict[str, pd.Series] = {}
        signal_dict: dict[str, pd.Series] = {}
        valid_tickers: list[str] = []

        total = len(tickers)
        for i, ticker in enumerate(tickers, 1):
            print(f"  [{i:3d}/{total}] {ticker:6s} ... ", end="", flush=True)
            try:
                ohlc = self.data.get_ohlc(ticker)
                signal = self.combiner.combine_to_signal(
                    ticker, factors, ohlc.index
                )
                price_dict[ticker] = ohlc["Close"]
                signal_dict[ticker] = signal
                valid_tickers.append(ticker)
                print("OK")
            except Exception as e:
                logger.warning("Skipping %s: %s", ticker, e)
                print(f"SKIP: {e}")

        prices = pd.DataFrame(price_dict)
        signals = pd.DataFrame(signal_dict)
        common_idx = prices.dropna(how="all").index
        prices = prices.loc[common_idx]
        signals = signals.reindex(common_idx)
        return prices, signals, valid_tickers

    # ------------------------------------------------------------------
    # Core simulation
    # ------------------------------------------------------------------

    def _simulate(
        self,
        prices: pd.DataFrame,
        signals: pd.DataFrame,
        tickers: list[str],
    ) -> dict[str, Any]:
        cfg = self.config.backtest
        initial_cash = cfg.initial_cash
        commission_rate = cfg.commission

        cash = initial_cash
        positions: dict[str, Position] = {}  # ticker -> Position
        trades: list[dict[str, Any]] = []
        daily_records: list[dict[str, Any]] = []
        stop_loss_count = 0
        holding_expired_count = 0

        dates = prices.index

        for i, date in enumerate(dates):
            day_prices = prices.loc[date]
            day_signals = signals.loc[date]

            # --- Step 1: Check trailing stop-loss (daily) ---
            for ticker in list(positions):
                pos = positions[ticker]
                price = day_prices.get(ticker)
                if price is None or np.isnan(price):
                    continue
                # Update trailing high
                pos.trailing_high = max(pos.trailing_high, price)
                # Check stop
                if price < pos.trailing_high * (1 - cfg.trailing_stop_pct):
                    proceeds = pos.shares * price
                    commission = proceeds * commission_rate
                    cash += proceeds - commission
                    trades.append(self._trade_record(
                        date, ticker, "SELL", pos.shares, price,
                        proceeds, commission, cash, "STOP_LOSS",
                    ))
                    del positions[ticker]
                    stop_loss_count += 1
                    logger.debug(
                        "STOP_LOSS %s day %d: price=%.2f trail_high=%.2f",
                        ticker, i, price, pos.trailing_high,
                    )

            # --- Step 2: Check holding period expiry (252 days) ---
            for ticker in list(positions):
                pos = positions[ticker]
                if i - pos.entry_day >= cfg.holding_period_days:
                    price = day_prices.get(ticker)
                    if price is None or np.isnan(price):
                        continue
                    proceeds = pos.shares * price
                    commission = proceeds * commission_rate
                    cash += proceeds - commission
                    trades.append(self._trade_record(
                        date, ticker, "SELL", pos.shares, price,
                        proceeds, commission, cash, "EXPIRED",
                    ))
                    del positions[ticker]
                    holding_expired_count += 1

            # --- Step 3: Scale-in check ---
            if cfg.scale_in:
                for ticker in list(positions):
                    pos = positions[ticker]
                    if pos.scaled_in:
                        continue
                    if i - pos.entry_day >= cfg.scale_in_days:
                        # Check signal still good
                        sig = day_signals.get(ticker, 0)
                        if np.isnan(sig):
                            sig = 0
                        if sig > 0:
                            remaining = pos.target_shares - pos.shares
                            if remaining > 0:
                                price = day_prices.get(ticker)
                                if price is None or np.isnan(price) or price <= 0:
                                    continue
                                cost = remaining * price
                                commission = cost * commission_rate
                                if cost + commission <= cash:
                                    cash -= cost + commission
                                    pos.shares += remaining
                                    pos.trailing_high = max(pos.trailing_high, price)
                                    trades.append(self._trade_record(
                                        date, ticker, "BUY", remaining, price,
                                        cost, commission, cash, "SCALE_IN",
                                    ))
                        pos.scaled_in = True

            # --- Step 4: New entries (daily signal check) ---
            # Find candidates not already held
            candidates: list[tuple[str, float]] = []
            for ticker in signals.columns:
                if ticker in positions:
                    continue
                sig = day_signals.get(ticker, 0)
                if np.isnan(sig):
                    continue
                if sig > 0:
                    price = day_prices.get(ticker)
                    if price is not None and not np.isnan(price) and price > 0:
                        candidates.append((ticker, sig))

            if candidates:
                # Sort by signal strength, take top available slots
                candidates.sort(key=lambda x: x[1], reverse=True)
                available_slots = cfg.max_positions - len(positions)
                new_buys = candidates[:max(0, available_slots)]

                if new_buys:
                    # Compute target weights for new buys
                    target_values = self._compute_new_buy_values(
                        new_buys, day_signals, day_prices,
                        cash, positions, cfg,
                    )
                    for ticker, target_val in target_values.items():
                        price = day_prices[ticker]
                        if cfg.scale_in:
                            # Buy scale_in_pct of target initially
                            initial_val = target_val * cfg.scale_in_pct
                            initial_shares = math.floor(initial_val / price)
                            full_shares = math.floor(target_val / price)
                        else:
                            initial_shares = math.floor(target_val / price)
                            full_shares = initial_shares

                        if initial_shares <= 0:
                            continue
                        cost = initial_shares * price
                        commission = cost * commission_rate
                        if cost + commission > cash:
                            initial_shares = math.floor(
                                cash / (price * (1 + commission_rate))
                            )
                            if initial_shares <= 0:
                                continue
                            cost = initial_shares * price
                            commission = cost * commission_rate
                            full_shares = initial_shares  # can't scale in more

                        cash -= cost + commission
                        positions[ticker] = Position(
                            ticker=ticker,
                            shares=initial_shares,
                            entry_day=i,
                            entry_price=price,
                            trailing_high=price,
                            target_shares=full_shares,
                            scaled_in=not cfg.scale_in,
                        )
                        trades.append(self._trade_record(
                            date, ticker, "BUY", initial_shares, price,
                            cost, commission, cash, "SIGNAL_ENTRY",
                        ))

            # --- Mark-to-market ---
            position_value = self._mark_to_market(positions, day_prices)
            portfolio_value = cash + position_value

            daily_records.append({
                "date": date,
                "portfolio_value": portfolio_value,
                "cash": cash,
                "position_value": position_value,
                "num_positions": len(positions),
            })

        # Build results
        daily_df = pd.DataFrame(daily_records).set_index("date")
        equity_curve = daily_df["portfolio_value"]
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

        # Convert positions dict to holdings for P&L
        final_holdings = {t: p.shares for t, p in positions.items()}

        stats = self._compute_stats(equity_curve, trades_df, initial_cash)
        stats["Stop-Loss Exits"] = stop_loss_count
        stats["Holding Expired Exits"] = holding_expired_count
        stats["Avg Positions"] = round(daily_df["num_positions"].mean(), 1)

        # Compute avg holding days from trades
        if not trades_df.empty:
            sell_trades = trades_df[trades_df["side"] == "SELL"]
            if not sell_trades.empty and "reason" in sell_trades.columns:
                stats["Sell Reasons"] = sell_trades["reason"].value_counts().to_dict()

        final_prices = prices.iloc[-1]
        per_stock_pnl = self._compute_per_stock_pnl(
            trades_df, final_holdings, final_prices
        )

        return {
            "equity_curve": equity_curve,
            "daily": daily_df,
            "trades": trades_df,
            "stats": stats,
            "per_stock_pnl": per_stock_pnl,
        }

    # ------------------------------------------------------------------
    # Target weight computation
    # ------------------------------------------------------------------

    def _compute_new_buy_values(
        self,
        new_buys: list[tuple[str, float]],
        signals: pd.Series,
        prices: pd.Series,
        cash: float,
        positions: dict[str, Position],
        cfg,
    ) -> dict[str, float]:
        """Compute dollar allocation for new buys using signal-weighted sizing."""
        # Total portfolio value
        position_value = self._mark_to_market(positions, prices)
        portfolio_value = cash + position_value
        investable = portfolio_value * (1 - cfg.cash_reserve_pct)

        # Current position values
        current_invested = position_value

        # Available for new buys
        available = min(cash, investable - current_invested)
        if available <= 0:
            return {}

        if cfg.signal_weighted:
            # Signal-weighted allocation
            raw_weights: dict[str, float] = {}
            for ticker, sig in new_buys:
                # Map signal to weight: (score - threshold) / (1 - threshold)
                # signals are already mapped this way in combiner
                raw_weights[ticker] = max(sig, 0.05)

            total_w = sum(raw_weights.values())
            weights = {t: w / total_w for t, w in raw_weights.items()}

            # Cap at max_position_weight (relative to total portfolio)
            weights = self._cap_weights(
                weights, cfg.max_position_weight, portfolio_value, prices,
            )
        else:
            n = len(new_buys)
            weights = {t: 1.0 / n for t, _ in new_buys}

        # Convert weights to dollar values, capped by available cash
        target_values: dict[str, float] = {}
        total_target = sum(
            weights[t] * portfolio_value for t in weights
        )
        # Scale down if we don't have enough cash
        scale = min(1.0, available / total_target) if total_target > 0 else 0

        for ticker in weights:
            val = weights[ticker] * portfolio_value * scale
            if val > 0:
                target_values[ticker] = val

        return target_values

    @staticmethod
    def _cap_weights(
        weights: dict[str, float],
        max_weight: float,
        portfolio_value: float,
        prices: pd.Series,
    ) -> dict[str, float]:
        """Cap weights at max_weight, redistribute excess proportionally."""
        for _ in range(10):  # iterate until stable
            excess = 0.0
            uncapped = []
            for t, w in weights.items():
                if w > max_weight:
                    excess += w - max_weight
                    weights[t] = max_weight
                else:
                    uncapped.append(t)
            if excess <= 0 or not uncapped:
                break
            # Redistribute excess proportionally to uncapped
            uncapped_total = sum(weights[t] for t in uncapped)
            if uncapped_total <= 0:
                break
            for t in uncapped:
                weights[t] += excess * (weights[t] / uncapped_total)
        return weights

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
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

    @staticmethod
    def _mark_to_market(
        positions: dict[str, Position], prices: pd.Series,
    ) -> float:
        value = 0.0
        for ticker, pos in positions.items():
            price = prices.get(ticker)
            if price is not None and not np.isnan(price) and pos.shares > 0:
                value += pos.shares * price
        return value

    @staticmethod
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

        daily_returns = equity_curve.pct_change().dropna()

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

        # Calmar ratio: annualized return / |max drawdown|
        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0

        # Profit factor: gross wins / gross losses (per-ticker P&L)
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
        }

    @staticmethod
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
                ticker_trades = trades_df[trades_df["ticker"] == ticker]
                buys = ticker_trades[ticker_trades["side"] == "BUY"]
                sells = ticker_trades[ticker_trades["side"] == "SELL"]
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

        return pd.DataFrame(pnl_records).sort_values(
            "total_pnl", ascending=False
        )
