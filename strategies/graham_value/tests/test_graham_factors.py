"""Tests for Graham value strategy factors."""

import numpy as np
import pandas as pd
import pytest

from strategies.graham_value.factors import (
    CurrentRatioFactor,
    DividendYieldFactor,
    GrahamNumberFactor,
    GrahamPERatioFactor,
    PriceToBookFactor,
)


class TestGrahamFactors:
    def test_graham_pe_output_range(self, mock_data_provider, sample_ohlc):
        factor = GrahamPERatioFactor()
        series = factor.compute_series("AAPL", sample_ohlc.index, mock_data_provider)

        assert not series.empty
        assert series.dropna().between(0, 1).all()

    def test_graham_pe_low_pe_scores_high(self, mock_data_provider, sample_ohlc):
        """Low PE values should produce high scores (undervalued)."""
        mock_data_provider.get_ratios.return_value["pe_ratio"] = pd.Series(
            [10.0] * len(sample_ohlc), index=sample_ohlc.index
        )
        factor = GrahamPERatioFactor()
        series = factor.compute_series("AAPL", sample_ohlc.index, mock_data_provider)

        # PE=10 is below Graham threshold of 15 -> sigmoid should score high
        valid = series.dropna()
        assert valid.mean() > 0.5

    def test_price_to_book_output_range(self, mock_data_provider, sample_ohlc):
        factor = PriceToBookFactor()
        series = factor.compute_series("AAPL", sample_ohlc.index, mock_data_provider)

        assert not series.empty
        assert series.dropna().between(0, 1).all()

    def test_current_ratio_output_range(self, mock_data_provider, sample_ohlc):
        factor = CurrentRatioFactor()
        series = factor.compute_series("AAPL", sample_ohlc.index, mock_data_provider)

        assert not series.empty
        assert series.dropna().between(0, 1).all()

    def test_current_ratio_high_is_better(self, mock_data_provider, sample_ohlc):
        """High current ratio should score well (financial safety)."""
        # Use increasing values well above threshold (2.0) so both
        # sigmoid and percentile rank produce high scores
        n = len(sample_ohlc)
        mock_data_provider.get_ratios.return_value["current_ratio"] = pd.Series(
            np.linspace(2.5, 4.0, n), index=sample_ohlc.index
        )
        factor = CurrentRatioFactor()
        series = factor.compute_series("AAPL", sample_ohlc.index, mock_data_provider)

        valid = series.dropna()
        assert valid.mean() > 0.5

    def test_dividend_yield_output_range(self, mock_data_provider, sample_ohlc):
        factor = DividendYieldFactor()
        series = factor.compute_series("AAPL", sample_ohlc.index, mock_data_provider)

        assert not series.empty
        assert series.dropna().between(0, 1).all()

    def test_graham_number_output_range(self, mock_data_provider, sample_ohlc):
        factor = GrahamNumberFactor()
        series = factor.compute_series("AAPL", sample_ohlc.index, mock_data_provider)

        assert not series.empty
        assert series.dropna().between(0, 1).all()

    def test_graham_number_negative_eps_neutral(self, mock_data_provider, sample_ohlc):
        """Negative EPS should return neutral (0.5) score."""
        mock_data_provider.get_ratios.return_value["earnings_per_share"] = pd.Series(
            [-2.0] * len(sample_ohlc), index=sample_ohlc.index
        )
        factor = GrahamNumberFactor()
        series = factor.compute_series("AAPL", sample_ohlc.index, mock_data_provider)

        # All values should be 0.5 (neutral) since EPS is negative
        valid = series.dropna()
        assert (valid == 0.5).all()

    def test_graham_pe_single_date(self, mock_data_provider):
        from datetime import date

        factor = GrahamPERatioFactor()
        result = factor.compute("AAPL", date(2023, 6, 15), mock_data_provider)

        assert result is not None
        assert 0 <= result.value <= 1
        assert "raw" in result.metadata

    def test_missing_ratio_returns_empty(self, mock_data_provider, sample_ohlc):
        mock_data_provider.get_ratios.return_value = {}
        factor = GrahamPERatioFactor()
        series = factor.compute_series("AAPL", sample_ohlc.index, mock_data_provider)

        assert series.isna().all()

    def test_factor_registration(self):
        """All 5 Graham factors should be registered."""
        from alphalab.factors.registry import _REGISTRY

        graham_names = [
            "graham_pe", "price_to_book", "graham_number",
            "current_ratio", "dividend_yield",
        ]
        for name in graham_names:
            assert name in _REGISTRY, f"Factor '{name}' not registered"
