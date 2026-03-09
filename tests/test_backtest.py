"""Tests for the backtest runner with mock data."""

import numpy as np
import pandas as pd
import pytest

from alphalab.backtest.runner import BacktestRunner, FactorStrategy
from alphalab.combiner.equal_weight import EqualWeightCombiner
from alphalab.config import AlphaLabConfig


class TestEqualWeightCombiner:
    def test_combine_produces_valid_range(self, config, mock_data_provider, sample_ohlc):
        from alphalab.factors.traditional import PERatioFactor, ROEFactor

        combiner = EqualWeightCombiner(config, mock_data_provider)
        factors = [PERatioFactor(), ROEFactor()]

        combined = combiner.combine("AAPL", factors, sample_ohlc.index)

        valid = combined.dropna()
        assert not valid.empty
        assert valid.between(0, 1).all()

    def test_signal_thresholds(self, config, mock_data_provider, sample_ohlc):
        from alphalab.factors.traditional import PERatioFactor, ROEFactor

        combiner = EqualWeightCombiner(config, mock_data_provider)
        factors = [PERatioFactor(), ROEFactor()]

        signal = combiner.combine_to_signal("AAPL", factors, sample_ohlc.index)

        # Signal should be between -1 and 1
        valid = signal.dropna()
        assert valid.between(-1, 1).all()


class TestFactorStrategy:
    def test_strategy_runs_with_signal(self, sample_ohlc):
        """End-to-end test with synthetic signal."""
        from backtesting import Backtest

        # Create a simple alternating signal
        signal = pd.Series(0.0, index=sample_ohlc.index)
        n = len(signal)
        signal.iloc[: n // 3] = 0.8  # Long first third
        signal.iloc[n // 3 : 2 * n // 3] = 0.0  # Flat middle third
        signal.iloc[2 * n // 3 :] = 0.5  # Long last third

        ohlc = sample_ohlc.copy()
        ohlc["signal"] = signal

        bt = Backtest(ohlc, FactorStrategy, cash=100_000, commission=0.001)
        stats = bt.run()

        assert stats["# Trades"] > 0
        assert "Return [%]" in stats.index
        assert "Sharpe Ratio" in stats.index
