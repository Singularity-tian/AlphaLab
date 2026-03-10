"""LLM judgment factor for earnings sentiment analysis."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from alphalab.anonymizer.pipeline import AnonymizationPipeline
from alphalab.config import AlphaLabConfig
from alphalab.data.cache import FileCache
from alphalab.factors.base import Factor, FactorResult
from alphalab.factors.registry import register_factor
from alphalab.llm.client import LLMClient
from alphalab.llm.prompts import EARNINGS_SENTIMENT_SYSTEM, EARNINGS_SENTIMENT_USER

if TYPE_CHECKING:
    from alphalab.data.providers import DataProvider

logger = logging.getLogger(__name__)


class SentimentScores(BaseModel):
    """Structured output from LLM sentiment analysis."""

    overall_sentiment: float = Field(ge=1, le=10)
    revenue_outlook: float = Field(ge=1, le=10)
    margin_trend: float = Field(ge=1, le=10)
    guidance_strength: float = Field(ge=1, le=10)
    risk_level: float = Field(ge=1, le=10)
    explanations: dict[str, str] = Field(default_factory=dict)


# Entity mappings for Phase 1 target stocks
COMPANY_ENTITIES: dict[str, dict] = {
    "AAPL": {
        "company_name": "Apple Inc.",
        "entities": {
            "Tim Cook": "person",
            "Luca Maestri": "person",
            "Kevan Parekh": "person",
            "iPhone": "product",
            "iPad": "product",
            "Mac": "product",
            "Apple Watch": "product",
            "AirPods": "product",
            "App Store": "product",
            "iCloud": "product",
            "Apple Silicon": "product",
            "Apple": "product",
        },
    },
}


@register_factor
class EarningsSentimentFactor(Factor):
    """LLM-judged sentiment from anonymized earnings call transcripts.

    Pipeline: fetch transcript -> anonymize -> LLM score -> normalize to [0,1].
    """

    name = "earnings_sentiment"
    description = "LLM-judged sentiment from anonymized earnings transcripts"
    category = "llm_judgment"

    def __init__(self):
        self._llm_client: Optional[LLMClient] = None
        self._anonymizer = AnonymizationPipeline()
        self._sentiment_cache = FileCache(
            cache_dir=Path("data/sp500_daily/cache/sentiment"),
            ttl_hours=24 * 30,
        )

    def _ensure_initialized(self, data: DataProvider) -> None:
        if self._llm_client is None:
            self._llm_client = LLMClient(data.config.llm)

    def compute(
        self,
        ticker: str,
        as_of_date: date,
        data: DataProvider,
    ) -> Optional[FactorResult]:
        self._ensure_initialized(data)

        transcript_dates = data.get_available_transcript_dates(ticker)
        if not transcript_dates:
            return None

        # Find most recent quarter before as_of_date
        relevant = [
            (y, q)
            for y, q in transcript_dates
            if date(y, q * 3, 1) <= as_of_date
        ]
        if not relevant:
            return None

        year, quarter = relevant[-1]

        # Check cache
        cache_key = f"sentiment_{ticker}_{year}_Q{quarter}"
        cached = self._sentiment_cache.get_json(cache_key)

        if cached is not None:
            scores = SentimentScores(**cached)
        else:
            transcript = data.get_earnings_transcripts(ticker, year, quarter)
            if not transcript or len(transcript) < 200:
                return None

            # Anonymize
            entity_info = COMPANY_ENTITIES.get(ticker, {})
            company_name = entity_info.get("company_name", ticker)
            additional = entity_info.get("entities", {})

            anon_result = self._anonymizer.anonymize(
                text=transcript,
                company_name=company_name,
                ticker=ticker,
                additional_entities=additional,
            )

            # Truncate to fit LLM context
            anon_text = anon_result.anonymized_text[:8000]

            prompt = EARNINGS_SENTIMENT_USER.format(transcript=anon_text)
            scores = self._llm_client.generate_structured(
                prompt=prompt,
                response_model=SentimentScores,
                system=EARNINGS_SENTIMENT_SYSTEM,
            )

            self._sentiment_cache.set_json(cache_key, scores.model_dump())

        # Normalize: average of dimensions, [1,10] -> [0,1]
        raw_scores = [
            scores.overall_sentiment,
            scores.revenue_outlook,
            scores.margin_trend,
            scores.guidance_strength,
            scores.risk_level,
        ]
        normalized = (np.mean(raw_scores) - 1.0) / 9.0

        return FactorResult(
            value=float(np.clip(normalized, 0.0, 1.0)),
            confidence=0.8,
            metadata={"scores": scores.model_dump(), "quarter": f"{year}Q{quarter}"},
        )

    def compute_series(
        self,
        ticker: str,
        date_range: pd.DatetimeIndex,
        data: DataProvider,
    ) -> pd.Series:
        self._ensure_initialized(data)

        transcript_dates = data.get_available_transcript_dates(ticker)
        if not transcript_dates:
            return pd.Series(dtype=float, index=date_range)

        quarter_scores: dict[pd.Timestamp, float] = {}
        for year, quarter in transcript_dates:
            # Approximate report date: quarter end + 45 days for earnings delay
            quarter_end = pd.Timestamp(year, quarter * 3, 28)
            report_date = quarter_end + pd.Timedelta(days=45)

            result = self.compute(ticker, report_date.date(), data)
            if result is not None:
                quarter_scores[report_date] = result.value

        if not quarter_scores:
            return pd.Series(dtype=float, index=date_range)

        quarterly = pd.Series(quarter_scores).sort_index()
        return quarterly.reindex(date_range, method="ffill")
