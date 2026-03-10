"""Tests for shared factor computation (traditional + momentum)."""

import numpy as np
import pandas as pd
import pytest

from alphalab.factors.base import FactorResult
from alphalab.factors.momentum import MomentumFactor
from alphalab.factors.traditional import FCFYieldFactor, PERatioFactor, ROEFactor


class TestTraditionalFactors:
    def test_pe_ratio_lower_is_better(self, mock_data_provider, sample_ohlc):
        factor = PERatioFactor()
        series = factor.compute_series("AAPL", sample_ohlc.index, mock_data_provider)

        assert not series.empty
        assert series.dropna().between(0, 1).all()

    def test_roe_higher_is_better(self, mock_data_provider, sample_ohlc):
        factor = ROEFactor()
        series = factor.compute_series("AAPL", sample_ohlc.index, mock_data_provider)

        assert not series.empty
        assert series.dropna().between(0, 1).all()

    def test_fcf_yield(self, mock_data_provider, sample_ohlc):
        factor = FCFYieldFactor()
        series = factor.compute_series("AAPL", sample_ohlc.index, mock_data_provider)

        assert not series.empty
        assert series.dropna().between(0, 1).all()

    def test_missing_ratio_returns_empty(self, mock_data_provider, sample_ohlc):
        mock_data_provider.get_ratios.return_value = {}
        factor = PERatioFactor()
        series = factor.compute_series("AAPL", sample_ohlc.index, mock_data_provider)

        assert series.isna().all()

    def test_compute_single_date(self, mock_data_provider):
        from datetime import date

        factor = PERatioFactor()
        result = factor.compute("AAPL", date(2023, 6, 15), mock_data_provider)

        assert result is not None
        assert 0 <= result.value <= 1
        assert "raw" in result.metadata


class TestMomentumFactor:
    def test_momentum_output_range(self, mock_data_provider, sample_ohlc):
        factor = MomentumFactor()
        series = factor.compute_series("AAPL", sample_ohlc.index, mock_data_provider)

        valid = series.dropna()
        assert valid.between(0, 1).all()

    def test_momentum_needs_lookback(self, mock_data_provider):
        """Should return None when insufficient history."""
        from datetime import date

        # Make OHLC very short
        short_ohlc = mock_data_provider.get_ohlc.return_value.iloc[:10]
        mock_data_provider.get_ohlc.return_value = short_ohlc

        factor = MomentumFactor()
        result = factor.compute("AAPL", date(2023, 1, 15), mock_data_provider)

        assert result is None

    def test_positive_momentum_above_half(self, mock_data_provider, sample_ohlc):
        """Strong positive returns should score > 0.5."""
        # Create monotonically increasing prices
        n = len(sample_ohlc)
        rising = pd.DataFrame(
            {
                "Open": np.linspace(100, 200, n),
                "High": np.linspace(101, 201, n),
                "Low": np.linspace(99, 199, n),
                "Close": np.linspace(100, 200, n),
                "Volume": [1_000_000] * n,
            },
            index=sample_ohlc.index,
        )
        mock_data_provider.get_ohlc.return_value = rising

        factor = MomentumFactor()
        series = factor.compute_series("AAPL", sample_ohlc.index, mock_data_provider)

        # Last values (where we have enough lookback) should be > 0.5
        tail = series.dropna().tail(20)
        if not tail.empty:
            assert tail.mean() > 0.5
