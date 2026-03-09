"""Tests for the business_resilience LLM factor."""

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from alphalab.data.cache import FileCache
from alphalab.factors.business_resilience import (
    JUDGE_MODEL,
    SUMMARIZER_MODEL,
    BusinessResilienceFactor,
    ResilienceScores,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_PROFILE = {
    "company_name": "Acme Corp",
    "description": "Acme Corp designs and sells widgets worldwide.",
    "sector": "Technology",
    "industry": "Widgets",
    "mkt_cap": 2_000_000_000_000,
    "full_time_employees": 100_000,
    "country": "US",
}

SAMPLE_REVENUE_SEG = {
    "by_product": {"Widgets": 60.0, "Services": 25.0, "Other": 15.0},
    "by_geography": {"Americas": 45.0, "Europe": 30.0, "Asia": 25.0},
}

SAMPLE_SCORES = ResilienceScores(
    business_quality=8.0,
    revenue_durability=7.0,
    ai_disruption_risk=6.0,
    fundamental_strength=7.5,
    explanations={
        "business_quality": "Strong moat",
        "revenue_durability": "Diversified revenue",
        "ai_disruption_risk": "Moderate AI exposure",
        "fundamental_strength": "Healthy balance sheet",
    },
)

SAMPLE_BRIEFING = "== Business Model ==\nCOMPANY_A is a technology company..." * 5


@pytest.fixture
def factor(tmp_path):
    f = BusinessResilienceFactor()
    f._cache = FileCache(cache_dir=tmp_path / "resilience", ttl_hours=24 * 90)
    return f


@pytest.fixture
def mock_data(mock_data_provider):
    """Extend mock_data_provider with resilience-specific mocks."""
    mock_data_provider.get_company_profile.return_value = SAMPLE_PROFILE
    mock_data_provider.get_revenue_segmentation.return_value = SAMPLE_REVENUE_SEG
    mock_data_provider.get_employee_count.return_value = 100_000
    mock_data_provider.get_available_transcript_dates.return_value = [(2023, 2)]
    mock_data_provider.get_earnings_transcripts.return_value = (
        "This is a sample earnings transcript with enough content. " * 20
    )
    return mock_data_provider


# ---------------------------------------------------------------------------
# Tests: ResilienceScores normalization
# ---------------------------------------------------------------------------


class TestResilienceScores:
    def test_normalization_midpoint(self):
        """All 5.5s should normalize to (5.5 - 1) / 9 = 0.5."""
        scores = ResilienceScores(
            business_quality=5.5,
            revenue_durability=5.5,
            ai_disruption_risk=5.5,
            fundamental_strength=5.5,
        )
        raw = [
            scores.business_quality,
            scores.revenue_durability,
            scores.ai_disruption_risk,
            scores.fundamental_strength,
        ]
        normalized = (np.mean(raw) - 1.0) / 9.0
        assert abs(normalized - 0.5) < 0.01

    def test_normalization_max(self):
        """All 10s should normalize to 1.0."""
        raw = [10.0, 10.0, 10.0, 10.0]
        normalized = (np.mean(raw) - 1.0) / 9.0
        assert abs(normalized - 1.0) < 0.01

    def test_normalization_min(self):
        """All 1s should normalize to 0.0."""
        raw = [1.0, 1.0, 1.0, 1.0]
        normalized = (np.mean(raw) - 1.0) / 9.0
        assert abs(normalized - 0.0) < 0.01

    def test_normalization_mixed(self):
        """Known mixed scores: (8+7+6+7.5)/4 = 7.125 -> (7.125-1)/9 = 0.680."""
        raw = [8.0, 7.0, 6.0, 7.5]
        normalized = (np.mean(raw) - 1.0) / 9.0
        assert abs(normalized - 0.680) < 0.01


# ---------------------------------------------------------------------------
# Tests: Factor computation
# ---------------------------------------------------------------------------


class TestBusinessResilienceFactor:
    @patch("alphalab.factors.business_resilience.LLMClient")
    def test_compute_returns_valid_range(self, MockLLMClient, factor, mock_data):
        mock_client = MagicMock()
        mock_client.generate.return_value = SAMPLE_BRIEFING
        mock_client.generate_structured.return_value = SAMPLE_SCORES
        MockLLMClient.return_value = mock_client

        result = factor.compute("AAPL", date(2023, 7, 1), mock_data)

        assert result is not None
        assert 0.0 <= result.value <= 1.0
        assert result.confidence == 0.7
        assert "scores" in result.metadata
        assert "briefing_preview" in result.metadata

    @patch("alphalab.factors.business_resilience.LLMClient")
    def test_compute_no_profile_returns_none(self, MockLLMClient, factor, mock_data):
        mock_data.get_company_profile.return_value = None

        result = factor.compute("AAPL", date(2023, 7, 1), mock_data)
        assert result is None

    @patch("alphalab.factors.business_resilience.LLMClient")
    def test_compute_uses_correct_models(self, MockLLMClient, factor, mock_data):
        mock_client = MagicMock()
        mock_client.generate.return_value = SAMPLE_BRIEFING
        mock_client.generate_structured.return_value = SAMPLE_SCORES
        MockLLMClient.return_value = mock_client

        factor.compute("AAPL", date(2023, 7, 1), mock_data)

        # Stage 1: summarizer should use flash model
        mock_client.generate.assert_called_once()
        call_kwargs = mock_client.generate.call_args
        assert call_kwargs.kwargs.get("model") == SUMMARIZER_MODEL

        # Stage 2: judge should use pro model
        mock_client.generate_structured.assert_called_once()
        call_kwargs = mock_client.generate_structured.call_args
        assert call_kwargs.kwargs.get("model") == JUDGE_MODEL

    @patch("alphalab.factors.business_resilience.LLMClient")
    def test_compute_graceful_without_optional_data(
        self, MockLLMClient, factor, mock_data
    ):
        """Factor should still work without revenue segmentation or employee count."""
        mock_data.get_revenue_segmentation.return_value = None
        mock_data.get_employee_count.return_value = None
        mock_data.get_available_transcript_dates.return_value = []

        mock_client = MagicMock()
        mock_client.generate.return_value = SAMPLE_BRIEFING
        mock_client.generate_structured.return_value = SAMPLE_SCORES
        MockLLMClient.return_value = mock_client

        result = factor.compute("AAPL", date(2023, 7, 1), mock_data)
        assert result is not None
        assert 0.0 <= result.value <= 1.0

    @patch("alphalab.factors.business_resilience.LLMClient")
    def test_caching_prevents_duplicate_calls(self, MockLLMClient, factor, mock_data):
        mock_client = MagicMock()
        mock_client.generate.return_value = SAMPLE_BRIEFING
        mock_client.generate_structured.return_value = SAMPLE_SCORES
        MockLLMClient.return_value = mock_client

        # First call
        result1 = factor.compute("AAPL", date(2023, 7, 1), mock_data)
        # Second call same year — should use cache
        result2 = factor.compute("AAPL", date(2023, 9, 1), mock_data)

        assert result1 is not None
        assert result2 is not None
        # LLM should only be called once (cached for second call)
        assert mock_client.generate.call_count == 1
        assert mock_client.generate_structured.call_count == 1

    @patch("alphalab.factors.business_resilience.LLMClient")
    def test_compute_series_forward_fills(self, MockLLMClient, factor, mock_data):
        mock_client = MagicMock()
        mock_client.generate.return_value = SAMPLE_BRIEFING
        mock_client.generate_structured.return_value = SAMPLE_SCORES
        MockLLMClient.return_value = mock_client

        date_range = pd.bdate_range("2023-01-01", "2023-12-31")
        series = factor.compute_series("AAPL", date_range, mock_data)

        assert len(series) == len(date_range)
        # Should have values (forward-filled from Jan 1)
        assert series.notna().sum() > 0
        # All values should be the same (one year, one evaluation)
        non_null = series.dropna()
        if len(non_null) > 1:
            assert non_null.nunique() == 1


# ---------------------------------------------------------------------------
# Tests: Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_factor_registered(self):
        from alphalab.factors.registry import list_factors

        assert "business_resilience" in list_factors()

    def test_factor_instantiation(self):
        from alphalab.factors.registry import get_factor

        f = get_factor("business_resilience")
        assert isinstance(f, BusinessResilienceFactor)
        assert f.category == "llm_judgment"
