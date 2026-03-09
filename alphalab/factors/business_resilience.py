"""Business resilience factor — two-stage LLM pipeline for fundamental quality and AI disruption risk."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from alphalab.data.cache import FileCache
from alphalab.factors.base import Factor, FactorResult
from alphalab.factors.registry import register_factor
from alphalab.llm.client import LLMClient
from alphalab.llm.prompts import (
    RESILIENCE_JUDGE_SYSTEM,
    RESILIENCE_JUDGE_USER,
    RESILIENCE_SUMMARIZER_SYSTEM,
    RESILIENCE_SUMMARIZER_USER,
)

if TYPE_CHECKING:
    from alphalab.data.providers import DataProvider

logger = logging.getLogger(__name__)

# Models for the two-stage pipeline
SUMMARIZER_MODEL = "gemini-3-flash-preview"
JUDGE_MODEL = "gemini-3.1-pro-preview"


class ResilienceScores(BaseModel):
    """Structured output from the judge LLM."""

    business_quality: float = Field(ge=1, le=10)
    revenue_durability: float = Field(ge=1, le=10)
    ai_disruption_risk: float = Field(ge=1, le=10)
    fundamental_strength: float = Field(ge=1, le=10)
    explanations: dict[str, str] = Field(default_factory=dict)


@register_factor
class BusinessResilienceFactor(Factor):
    """LLM-judged business quality and AI disruption resilience.

    Two-stage pipeline:
      Stage 1 (Summarizer): Raw data -> anonymized structured briefing
      Stage 2 (Judge): Anonymized briefing -> scored on 4 dimensions
    """

    name = "business_resilience"
    description = "LLM-judged business fundamental quality and AI disruption resilience"
    category = "llm_judgment"

    def __init__(self):
        self._llm_client: Optional[LLMClient] = None
        self._cache = FileCache(
            cache_dir=Path("data/cache/resilience"),
            ttl_hours=24 * 90,  # 90-day TTL
        )

    def _ensure_initialized(self, data: DataProvider) -> None:
        if self._llm_client is None:
            self._llm_client = LLMClient(data.config.llm)

    # ------------------------------------------------------------------
    # Data gathering
    # ------------------------------------------------------------------

    def _build_raw_data_text(
        self,
        ticker: str,
        as_of_date: date,
        data: DataProvider,
    ) -> str | None:
        """Concatenate all available raw data into a single text block."""
        profile = data.get_company_profile(ticker)
        if not profile:
            return None

        sections: list[str] = []

        # Company profile
        sections.append("== Company Profile ==")
        sections.append(f"Company: {profile.get('company_name', ticker)}")
        sections.append(f"Ticker: {ticker}")
        sections.append(f"Sector: {profile.get('sector', 'N/A')}")
        sections.append(f"Industry: {profile.get('industry', 'N/A')}")
        sections.append(f"Country: {profile.get('country', 'N/A')}")
        desc = profile.get("description", "")
        if desc:
            sections.append(f"Description: {desc}")

        # Employee count
        employees = data.get_employee_count(ticker)
        if employees:
            sections.append(f"Employees: {employees:,}")

        # Revenue segmentation
        rev_seg = data.get_revenue_segmentation(ticker)
        if rev_seg:
            if "by_product" in rev_seg:
                sections.append("\n== Revenue by Product/Segment ==")
                for seg, pct in sorted(
                    rev_seg["by_product"].items(), key=lambda x: x[1], reverse=True
                ):
                    sections.append(f"  {seg}: {pct}%")
            if "by_geography" in rev_seg:
                sections.append("\n== Revenue by Geography ==")
                for geo, pct in sorted(
                    rev_seg["by_geography"].items(), key=lambda x: x[1], reverse=True
                ):
                    sections.append(f"  {geo}: {pct}%")

        # Financial ratios at as_of_date
        try:
            ratios = data.get_ratios(ticker)
            if ratios:
                sections.append(f"\n== Financial Metrics (as of {as_of_date}) ==")
                ratio_names = {
                    "pe_ratio": "PE Ratio",
                    "price_to_book": "Price/Book",
                    "roe": "ROE",
                    "fcf_yield": "FCF Yield",
                    "current_ratio": "Current Ratio",
                    "dividend_yield": "Dividend Yield",
                    "earnings_per_share": "EPS",
                }
                for key, label in ratio_names.items():
                    if key in ratios:
                        series = ratios[key]
                        # Get value at or before as_of_date
                        mask = series.index <= pd.Timestamp(as_of_date)
                        if mask.any():
                            val = series[mask].iloc[-1]
                            if key in ("roe", "fcf_yield", "dividend_yield"):
                                sections.append(f"  {label}: {val:.1%}")
                            else:
                                sections.append(f"  {label}: {val:.2f}")
        except Exception as e:
            logger.warning("Failed to get ratios for %s: %s", ticker, e)

        # Earnings transcript (most recent before as_of_date)
        try:
            transcript_dates = data.get_available_transcript_dates(ticker)
            if transcript_dates:
                relevant = [
                    (y, q)
                    for y, q in transcript_dates
                    if date(y, q * 3, 1) <= as_of_date
                ]
                if relevant:
                    year, quarter = relevant[-1]
                    transcript = data.get_earnings_transcripts(ticker, year, quarter)
                    if transcript and len(transcript) > 200:
                        # Truncate to keep total prompt reasonable
                        truncated = transcript[:6000]
                        sections.append(
                            f"\n== Earnings Call Transcript ({year} Q{quarter}) =="
                        )
                        sections.append(truncated)
        except Exception as e:
            logger.warning("Failed to get transcript for %s: %s", ticker, e)

        return "\n".join(sections)

    # ------------------------------------------------------------------
    # Two-stage LLM pipeline
    # ------------------------------------------------------------------

    def _get_briefing(self, ticker: str, year: int, raw_data: str) -> str:
        """Stage 1: Summarize and anonymize raw data via LLM."""
        cache_key = f"resilience_briefing_{ticker}_{year}"
        cached = self._cache.get_json(cache_key)
        if cached is not None:
            return cached.get("briefing", "")

        prompt = RESILIENCE_SUMMARIZER_USER.format(raw_data=raw_data)
        briefing = self._llm_client.generate(
            prompt=prompt,
            system=RESILIENCE_SUMMARIZER_SYSTEM,
            model=SUMMARIZER_MODEL,
        )

        self._cache.set_json(cache_key, {"briefing": briefing})
        return briefing

    def _get_scores(self, ticker: str, year: int, briefing: str) -> ResilienceScores:
        """Stage 2: Judge the anonymized briefing via LLM."""
        cache_key = f"resilience_score_{ticker}_{year}"
        cached = self._cache.get_json(cache_key)
        if cached is not None:
            return ResilienceScores(**cached)

        prompt = RESILIENCE_JUDGE_USER.format(briefing=briefing)
        scores = self._llm_client.generate_structured(
            prompt=prompt,
            response_model=ResilienceScores,
            system=RESILIENCE_JUDGE_SYSTEM,
            model=JUDGE_MODEL,
        )

        self._cache.set_json(cache_key, scores.model_dump())
        return scores

    # ------------------------------------------------------------------
    # Factor interface
    # ------------------------------------------------------------------

    def compute(
        self,
        ticker: str,
        as_of_date: date,
        data: DataProvider,
    ) -> Optional[FactorResult]:
        self._ensure_initialized(data)

        raw_text = self._build_raw_data_text(ticker, as_of_date, data)
        if not raw_text:
            return None

        year = as_of_date.year

        # Stage 1: Summarize & anonymize
        briefing = self._get_briefing(ticker, year, raw_text)
        if not briefing or len(briefing) < 100:
            logger.warning("Briefing too short for %s %d, skipping", ticker, year)
            return None

        # Stage 2: Judge & score
        scores = self._get_scores(ticker, year, briefing)

        # Normalize: mean of 4 dimensions, [1, 10] -> [0, 1]
        raw_scores = [
            scores.business_quality,
            scores.revenue_durability,
            scores.ai_disruption_risk,
            scores.fundamental_strength,
        ]
        normalized = (np.mean(raw_scores) - 1.0) / 9.0

        return FactorResult(
            value=float(np.clip(normalized, 0.0, 1.0)),
            confidence=0.7,
            metadata={
                "scores": scores.model_dump(),
                "briefing_preview": briefing[:200],
                "year": year,
            },
        )

    def compute_series(
        self,
        ticker: str,
        date_range: pd.DatetimeIndex,
        data: DataProvider,
    ) -> pd.Series:
        self._ensure_initialized(data)

        # Evaluate once per calendar year and forward-fill
        years = sorted(set(dt.year for dt in date_range))
        yearly_scores: dict[pd.Timestamp, float] = {}

        for year in years:
            # Use mid-year as evaluation date to have enough data
            eval_date = date(year, 7, 1)
            if eval_date > date_range[-1].date():
                eval_date = date_range[-1].date()

            result = self.compute(ticker, eval_date, data)
            if result is not None:
                yearly_scores[pd.Timestamp(year, 1, 1)] = result.value

        if not yearly_scores:
            return pd.Series(dtype=float, index=date_range)

        yearly = pd.Series(yearly_scores).sort_index()
        return yearly.reindex(date_range, method="ffill")
