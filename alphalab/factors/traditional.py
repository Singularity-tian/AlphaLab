"""Traditional fundamental factors wrapping FinanceToolkit ratios."""

from __future__ import annotations

import logging
from datetime import date
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

from alphalab.factors.base import Factor, FactorResult
from alphalab.factors.registry import register_factor

if TYPE_CHECKING:
    from alphalab.data.providers import DataProvider

logger = logging.getLogger(__name__)


class TraditionalFactor(Factor):
    """Base for fundamental ratio factors.

    Normalizes raw ratio values to [0, 1] using expanding-window
    percentile rank to prevent look-ahead bias.
    """

    category = "traditional"
    ratio_key: str
    higher_is_better: bool = True

    def compute(
        self,
        ticker: str,
        as_of_date: date,
        data: DataProvider,
    ) -> Optional[FactorResult]:
        ratios = data.get_ratios(ticker)
        if self.ratio_key not in ratios:
            return None

        series = ratios[self.ratio_key]
        ts = pd.Timestamp(as_of_date)

        available = series.loc[:ts].dropna()
        if available.empty:
            return None

        current_value = available.iloc[-1]
        if len(available) < 2:
            return FactorResult(value=0.5, metadata={"raw": current_value})

        percentile = (available < current_value).sum() / len(available)
        if not self.higher_is_better:
            percentile = 1.0 - percentile

        return FactorResult(
            value=float(np.clip(percentile, 0.0, 1.0)),
            metadata={"raw": current_value, "percentile": percentile},
        )

    def compute_series(
        self,
        ticker: str,
        date_range: pd.DatetimeIndex,
        data: DataProvider,
    ) -> pd.Series:
        ratios = data.get_ratios(ticker)
        if self.ratio_key not in ratios:
            return pd.Series(dtype=float, index=date_range)

        series = ratios[self.ratio_key].reindex(date_range, method="ffill")

        # Expanding-window percentile rank (no look-ahead)
        rank = series.expanding(min_periods=2).apply(
            lambda w: (w.iloc[:-1] < w.iloc[-1]).sum() / (len(w) - 1)
            if len(w) > 1
            else 0.5,
            raw=False,
        )

        if not self.higher_is_better:
            rank = 1.0 - rank

        return rank.clip(0.0, 1.0)


@register_factor
class PERatioFactor(TraditionalFactor):
    name = "pe_ratio"
    description = "Price-to-Earnings ratio (lower = more undervalued)"
    ratio_key = "pe_ratio"
    higher_is_better = False


@register_factor
class ROEFactor(TraditionalFactor):
    name = "roe"
    description = "Return on Equity (higher = more profitable)"
    ratio_key = "roe"
    higher_is_better = True


@register_factor
class FCFYieldFactor(TraditionalFactor):
    name = "fcf_yield"
    description = "Free Cash Flow Yield (higher = cheaper relative to cash generation)"
    ratio_key = "fcf_yield"
    higher_is_better = True
