"""Price momentum factor."""

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


@register_factor
class MomentumFactor(Factor):
    """12-month momentum with 1-month skip (12-1 momentum).

    Measures the return from 12 months ago to 1 month ago,
    skipping the most recent month to avoid short-term reversal.
    Normalized via sigmoid centered at 0% return.
    """

    name = "momentum_12_1"
    description = "12-month price momentum with 1-month reversal skip"
    category = "momentum"

    lookback_days: int = 252  # ~12 months
    skip_days: int = 21  # ~1 month

    def compute(
        self,
        ticker: str,
        as_of_date: date,
        data: DataProvider,
    ) -> Optional[FactorResult]:
        ohlc = data.get_ohlc(ticker)
        ts = pd.Timestamp(as_of_date)

        available = ohlc.loc[:ts, "Close"]
        if len(available) < self.lookback_days:
            return None

        price_recent = (
            available.iloc[-self.skip_days]
            if len(available) > self.skip_days
            else available.iloc[-1]
        )
        price_past = available.iloc[-self.lookback_days]

        if price_past <= 0 or np.isnan(price_past):
            return None

        raw_return = (price_recent / price_past) - 1.0
        score = 1.0 / (1.0 + np.exp(-raw_return * 5))

        return FactorResult(
            value=float(np.clip(score, 0.0, 1.0)),
            metadata={"raw_return": raw_return},
        )

    def compute_series(
        self,
        ticker: str,
        date_range: pd.DatetimeIndex,
        data: DataProvider,
    ) -> pd.Series:
        ohlc = data.get_ohlc(ticker)
        close = ohlc["Close"].reindex(date_range, method="ffill")

        price_past = close.shift(self.lookback_days)
        price_recent = close.shift(self.skip_days)

        raw_return = (price_recent / price_past) - 1.0
        score = 1.0 / (1.0 + np.exp(-raw_return * 5))

        return score.clip(0.0, 1.0)
