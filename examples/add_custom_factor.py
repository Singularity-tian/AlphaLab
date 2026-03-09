"""Example: Adding a custom RSI factor to AlphaLab.

Shows how to create and register a custom factor, then run it
alongside the built-in factors.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

import logging
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

import alphalab.factors.momentum
import alphalab.factors.traditional
from alphalab.backtest.report import ReportGenerator
from alphalab.backtest.runner import BacktestRunner
from alphalab.config import AlphaLabConfig
from alphalab.factors.base import Factor, FactorResult
from alphalab.factors.registry import register_factor

logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")


@register_factor
class RSIFactor(Factor):
    """Relative Strength Index factor.

    RSI < 30 = oversold (bullish score near 1.0)
    RSI > 70 = overbought (bearish score near 0.0)
    """

    name = "rsi_14"
    description = "14-day Relative Strength Index"
    category = "traditional"

    period: int = 14

    def compute(
        self, ticker: str, as_of_date: date, data
    ) -> Optional[FactorResult]:
        ohlc = data.get_ohlc(ticker)
        close = ohlc.loc[: pd.Timestamp(as_of_date), "Close"]

        if len(close) < self.period + 1:
            return None

        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(self.period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(self.period).mean()

        rs = gain.iloc[-1] / max(loss.iloc[-1], 1e-10)
        rsi = 100 - (100 / (1 + rs))

        # Normalize: RSI 30 -> 1.0 (oversold/bullish), RSI 70 -> 0.0
        score = 1.0 - ((rsi - 30) / 40.0)
        return FactorResult(value=float(np.clip(score, 0.0, 1.0)))

    def compute_series(self, ticker, date_range, data) -> pd.Series:
        ohlc = data.get_ohlc(ticker)
        close = ohlc["Close"].reindex(date_range, method="ffill")

        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(self.period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(self.period).mean()

        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))

        score = 1.0 - ((rsi - 30) / 40.0)
        return score.clip(0.0, 1.0)


def main():
    config = AlphaLabConfig.default()
    config.factors.factors = ["pe_ratio", "roe", "momentum_12_1", "rsi_14"]

    runner = BacktestRunner(config)
    results = runner.run("AAPL")

    reporter = ReportGenerator()
    reporter.generate(results, "AAPL")


if __name__ == "__main__":
    main()
