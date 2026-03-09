"""Portfolio-level backtest: single account across multiple stocks."""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from alphalab.combiner.equal_weight import EqualWeightCombiner
from alphalab.config import AlphaLabConfig
from alphalab.data.providers import DataProvider
from alphalab.factors.base import Factor
from alphalab.factors.registry import get_factor

logger = logging.getLogger(__name__)


class PortfolioBacktestRunner:
    """Single-account backtest across multiple stocks.

    Instead of running independent per-stock backtests, this allocates
    a single pool of capital across all stocks based on factor signals.
    """

    def __init__(self, config: AlphaLabConfig):
        self.config = config
        self.data = DataProvider(config)
        self.combiner = EqualWeightCombiner(config, self.data)

    def run(
        self,
        tickers: list[str],
        factor_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run a portfolio backtest across multiple tickers.

        Returns dict with: equity_curve, trades, stats, per_stock_pnl,
        holdings_history.
        """
        if factor_names is None:
            factor_names = self.config.factors.factors

        # Instantiate factors
        factors: list[Factor] = []
        for name in factor_names:
            try:
                factors.append(get_factor(name))
            except KeyError as e:
                logger.warning(str(e))
        if not factors:
            raise ValueError("No valid factors found.")

        # Phase 1: Pre-compute prices and signals for all tickers
        prices, signals, valid_tickers = self._precompute(tickers, factors)

        if not valid_tickers:
            raise ValueError("No tickers with valid data.")

        logger.info(
            "Portfolio backtest: %d tickers, %d trading days",
            len(valid_tickers),
            len(prices),
        )

        # Phase 2: Simulate portfolio
        return self._simulate(prices, signals, valid_tickers)

    def _precompute(
        self,
        tickers: list[str],
        factors: list[Factor],
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        """Fetch OHLC and compute signals for all tickers.

        Returns (prices_df, signals_df, valid_tickers).
        """
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

        # Align all series to a common date index
        prices = pd.DataFrame(price_dict)
        signals = pd.DataFrame(signal_dict)

        # Use intersection of dates where we have at least some data
        common_idx = prices.dropna(how="all").index
        prices = prices.loc[common_idx]
        signals = signals.reindex(common_idx)

        return prices, signals, valid_tickers

    def _simulate(
        self,
        prices: pd.DataFrame,
        signals: pd.DataFrame,
        tickers: list[str],
    ) -> dict[str, Any]:
        """Run the day-by-day portfolio simulation."""
        initial_cash = self.config.backtest.initial_cash
        commission_rate = self.config.backtest.commission

        cash = initial_cash
        holdings: dict[str, int] = {}  # ticker -> shares
        trades: list[dict[str, Any]] = []
        daily_records: list[dict[str, Any]] = []

        dates = prices.index

        for i, date in enumerate(dates):
            day_prices = prices.loc[date]

            # Check if this is a rebalance date
            is_rebalance = date.is_month_start or (i == 0)

            if is_rebalance:
                cash, holdings, new_trades = self._rebalance(
                    date,
                    day_prices,
                    signals.loc[date],
                    cash,
                    holdings,
                    commission_rate,
                )
                trades.extend(new_trades)

            # Mark-to-market
            position_value = self._mark_to_market(holdings, day_prices)
            portfolio_value = cash + position_value

            daily_records.append(
                {
                    "date": date,
                    "portfolio_value": portfolio_value,
                    "cash": cash,
                    "position_value": position_value,
                    "num_positions": sum(
                        1 for s in holdings.values() if s > 0
                    ),
                }
            )

        # Build results
        daily_df = pd.DataFrame(daily_records).set_index("date")
        equity_curve = daily_df["portfolio_value"]
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

        stats = self._compute_stats(equity_curve, trades_df, initial_cash)
        final_prices = prices.iloc[-1]
        per_stock_pnl = self._compute_per_stock_pnl(
            trades_df, holdings, final_prices
        )

        return {
            "equity_curve": equity_curve,
            "daily": daily_df,
            "trades": trades_df,
            "stats": stats,
            "per_stock_pnl": per_stock_pnl,
        }

    def _rebalance(
        self,
        date: pd.Timestamp,
        prices: pd.Series,
        signals: pd.Series,
        cash: float,
        holdings: dict[str, int],
        commission_rate: float,
    ) -> tuple[float, dict[str, int], list[dict]]:
        """Rebalance portfolio: sell unwanted, buy target positions.

        Returns (new_cash, new_holdings, trades).
        """
        new_trades: list[dict] = []

        # Identify target tickers (positive signal and valid price)
        target_tickers = [
            t
            for t in signals.index
            if not np.isnan(signals[t])
            and signals[t] > 0
            and not np.isnan(prices.get(t, np.nan))
        ]

        target_set = set(target_tickers)

        # Step 1: Sell positions not in target set
        for ticker, shares in list(holdings.items()):
            if shares <= 0:
                continue
            if ticker not in target_set:
                price = prices.get(ticker)
                if price is None or np.isnan(price):
                    continue
                proceeds = shares * price
                commission = proceeds * commission_rate
                cash += proceeds - commission
                holdings[ticker] = 0
                new_trades.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "side": "SELL",
                        "shares": shares,
                        "price": price,
                        "value": proceeds,
                        "commission": commission,
                        "cash_impact": proceeds - commission,
                        "cash_after": cash,
                    }
                )

        # Step 2: Compute target allocation using post-sell portfolio value
        position_value = self._mark_to_market(holdings, prices)
        portfolio_value = cash + position_value

        n_targets = len(target_tickers)
        weight_per_stock = 1.0 / n_targets if n_targets > 0 else 0.0

        # Compute target shares and determine trades needed
        buy_orders: list[tuple[str, int, float]] = []  # (ticker, delta_shares, price)
        sell_orders: list[tuple[str, int, float]] = []  # (ticker, delta_shares, price)

        for ticker in target_tickers:
            price = prices[ticker]
            if price <= 0:
                continue
            target_value = portfolio_value * weight_per_stock
            target_sh = math.floor(target_value / price)
            current = holdings.get(ticker, 0)
            delta = target_sh - current
            if delta > 0:
                buy_orders.append((ticker, delta, price))
            elif delta < 0:
                sell_orders.append((ticker, -delta, price))

        # Step 3: Execute sells (reduce overweight positions)
        for ticker, sell_shares, price in sell_orders:
            proceeds = sell_shares * price
            commission = proceeds * commission_rate
            cash += proceeds - commission
            holdings[ticker] = holdings.get(ticker, 0) - sell_shares
            new_trades.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "side": "SELL",
                    "shares": sell_shares,
                    "price": price,
                    "value": proceeds,
                    "commission": commission,
                    "cash_impact": proceeds - commission,
                    "cash_after": cash,
                }
            )

        # Step 4: Execute buys proportionally (no ordering bias)
        if buy_orders:
            total_buy_cost = sum(
                sh * pr * (1 + commission_rate) for _, sh, pr in buy_orders
            )
            # Compute scale ONCE before the loop
            if total_buy_cost > cash:
                scale = cash / total_buy_cost
            else:
                scale = 1.0

            scaled_orders = [
                (tk, math.floor(sh * scale), pr)
                for tk, sh, pr in buy_orders
            ]

            for ticker, buy_shares, price in scaled_orders:
                if buy_shares <= 0:
                    continue
                cost = buy_shares * price
                commission = cost * commission_rate
                # Safety guard: never spend more cash than we have
                if cost + commission > cash:
                    buy_shares = math.floor(
                        cash / (price * (1 + commission_rate))
                    )
                    if buy_shares <= 0:
                        continue
                    cost = buy_shares * price
                    commission = cost * commission_rate
                cash -= cost + commission
                holdings[ticker] = holdings.get(ticker, 0) + buy_shares
                new_trades.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "side": "BUY",
                        "shares": buy_shares,
                        "price": price,
                        "value": cost,
                        "commission": commission,
                        "cash_impact": -(cost + commission),
                        "cash_after": cash,
                    }
                )

        # Clean up zero holdings
        holdings = {t: s for t, s in holdings.items() if s > 0}

        return cash, holdings, new_trades

    @staticmethod
    def _mark_to_market(
        holdings: dict[str, int], prices: pd.Series
    ) -> float:
        """Compute total market value of current holdings."""
        value = 0.0
        for ticker, shares in holdings.items():
            price = prices.get(ticker)
            if price is not None and not np.isnan(price) and shares > 0:
                value += shares * price
        return value

    @staticmethod
    def _compute_stats(
        equity_curve: pd.Series,
        trades_df: pd.DataFrame,
        initial_cash: float,
    ) -> dict[str, Any]:
        """Compute portfolio-level performance statistics."""
        final_value = equity_curve.iloc[-1]
        total_return = (final_value / initial_cash - 1) * 100

        # Annualized return
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = days / 365.25
        ann_return = ((final_value / initial_cash) ** (1 / years) - 1) * 100 if years > 0 else 0

        # Daily returns for Sharpe/Sortino
        daily_returns = equity_curve.pct_change().dropna()

        sharpe = 0.0
        sortino = 0.0
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = (
                daily_returns.mean() / daily_returns.std() * np.sqrt(252)
            )
            downside = daily_returns[daily_returns < 0]
            if len(downside) > 0 and downside.std() > 0:
                sortino = (
                    daily_returns.mean() / downside.std() * np.sqrt(252)
                )

        # Max drawdown
        cummax = equity_curve.cummax()
        drawdown = (equity_curve - cummax) / cummax * 100
        max_dd = drawdown.min()

        # Trade stats
        num_trades = len(trades_df)
        total_commission = trades_df["commission"].sum() if num_trades > 0 else 0

        return {
            "Total Return [%]": round(total_return, 2),
            "Annualized Return [%]": round(ann_return, 2),
            "Sharpe Ratio": round(sharpe, 3),
            "Sortino Ratio": round(sortino, 3),
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
        """Compute P&L contribution per stock including unrealized gains."""
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

            # Add unrealized value for open positions
            open_shares = final_holdings.get(ticker, 0)
            unrealized_value = 0.0
            if open_shares > 0:
                price = final_prices.get(ticker)
                if price is not None and not np.isnan(price):
                    unrealized_value = open_shares * price

            total_pnl = total_proceeds + unrealized_value - total_cost
            pnl_records.append(
                {
                    "ticker": ticker,
                    "total_cost": round(total_cost, 2),
                    "total_proceeds": round(total_proceeds, 2),
                    "unrealized_value": round(unrealized_value, 2),
                    "total_pnl": round(total_pnl, 2),
                    "open_shares": open_shares,
                    "num_buys": len(buys),
                    "num_sells": len(sells),
                }
            )

        return pd.DataFrame(pnl_records).sort_values(
            "total_pnl", ascending=False
        )
