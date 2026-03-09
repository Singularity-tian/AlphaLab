"""Backtest runner orchestrating the full pipeline."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

from alphalab.combiner.equal_weight import EqualWeightCombiner
from alphalab.config import AlphaLabConfig
from alphalab.data.providers import DataProvider
from alphalab.factors.base import Factor
from alphalab.factors.registry import get_factor

logger = logging.getLogger(__name__)


class FactorStrategy(Strategy):
    """backtesting.py Strategy driven by pre-computed factor signals.

    The signal is injected as a column on the data DataFrame before
    the backtest runs.
    """

    def init(self):
        self.signal = self.I(
            lambda: self.data.df["signal"].values,
            name="Factor Signal",
            plot=True,
        )

    def next(self):
        current_signal = self.signal[-1]

        if np.isnan(current_signal):
            return

        if current_signal > 0 and not self.position.is_long:
            self.position.close()
            self.buy(size=min(abs(current_signal), 0.95))
        elif current_signal <= 0 and self.position:
            self.position.close()


class BacktestRunner:
    """Main entry point: data -> factors -> combine -> backtest -> results."""

    def __init__(self, config: AlphaLabConfig):
        self.config = config
        self.data = DataProvider(config)
        self.combiner = EqualWeightCombiner(config, self.data)

    def run(
        self,
        ticker: str,
        factor_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run a full backtest for a single ticker.

        Returns dict with: stats, equity_curve, trades, factor_scores,
        combined_signal, ohlc, bt.
        """
        if factor_names is None:
            factor_names = self.config.factors.factors

        logger.info("Starting backtest for %s with factors: %s", ticker, factor_names)

        # 1. Get OHLC data
        ohlc = self.data.get_ohlc(ticker)
        logger.info("OHLC: %d rows, %s to %s", len(ohlc), ohlc.index[0], ohlc.index[-1])

        # 2. Instantiate factors
        factors: list[Factor] = []
        for name in factor_names:
            try:
                factors.append(get_factor(name))
            except KeyError as e:
                logger.warning(str(e))

        if not factors:
            raise ValueError("No valid factors found.")

        # 3. Compute combined signal
        signal = self.combiner.combine_to_signal(ticker, factors, ohlc.index)

        # 4. Inject signal into OHLC
        ohlc_with_signal = ohlc.copy()
        ohlc_with_signal["signal"] = signal
        ohlc_with_signal = ohlc_with_signal.dropna(subset=["Open", "High", "Low", "Close"])

        # 5. Run backtest
        bt = Backtest(
            ohlc_with_signal,
            FactorStrategy,
            cash=self.config.backtest.initial_cash,
            commission=self.config.backtest.commission,
            trade_on_close=True,
            finalize_trades=True,
        )
        stats = bt.run()

        # 6. Collect individual factor scores for attribution
        factor_scores = pd.DataFrame(index=ohlc.index)
        for factor in factors:
            try:
                factor_scores[factor.name] = factor.compute_series(
                    ticker, ohlc.index, self.data
                )
            except Exception:
                pass

        result = {
            "stats": stats,
            "equity_curve": stats["_equity_curve"],
            "trades": stats["_trades"],
            "factor_scores": factor_scores,
            "combined_signal": signal,
            "ohlc": ohlc,
            "bt": bt,
        }

        logger.info("Return: %.2f%%", stats["Return [%]"])
        logger.info("Sharpe: %.3f", stats["Sharpe Ratio"])
        logger.info("Max Drawdown: %.2f%%", stats["Max. Drawdown [%]"])
        logger.info("Trades: %d", stats["# Trades"])

        return result
