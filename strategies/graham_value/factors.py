"""Benjamin Graham value factors with hybrid normalization.

Combines absolute Graham thresholds (sigmoid) with expanding-window
percentile rank:

    score = 0.4 * sigmoid(raw, threshold) + 0.6 * percentile_rank

This anchors in Graham's value philosophy while remaining practical
for modern stocks whose absolute ratios may differ from Graham-era norms.
"""

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

# Default blending weights (overridden by config.factors.sigmoid_weight)
_DEFAULT_SIGMOID_WEIGHT = 0.4


def _sigmoid(x: float, center: float, steepness: float) -> float:
    """Sigmoid mapping: value at center -> 0.5, below -> higher, above -> lower."""
    return 1.0 / (1.0 + np.exp(steepness * (x - center)))


def _sigmoid_array(arr: pd.Series, center: float, steepness: float) -> pd.Series:
    """Vectorized sigmoid for Series."""
    return 1.0 / (1.0 + np.exp(steepness * (arr - center)))


class GrahamFactor(Factor):
    """Base for Graham-style factors using hybrid normalization.

    Subclasses set ratio_key, graham_threshold, steepness, and higher_is_better.
    """

    category = "traditional"
    ratio_key: str
    graham_threshold: float
    steepness: float
    higher_is_better: bool = False  # Graham factors mostly favor lower values

    def _hybrid_score(
        self, current_value: float, history: pd.Series,
        sigmoid_weight: float = _DEFAULT_SIGMOID_WEIGHT,
    ) -> float:
        """Compute blended sigmoid + percentile score."""
        percentile_weight = 1.0 - sigmoid_weight

        # Sigmoid component: maps raw value through Graham threshold
        if self.higher_is_better:
            sig = _sigmoid(current_value, self.graham_threshold, -self.steepness)
        else:
            sig = _sigmoid(current_value, self.graham_threshold, self.steepness)

        # Percentile component: expanding-window rank
        if len(history) < 2:
            pct = 0.5
        else:
            pct = float((history.iloc[:-1] < current_value).sum() / (len(history) - 1))
            if not self.higher_is_better:
                pct = 1.0 - pct

        return float(np.clip(
            sigmoid_weight * sig + percentile_weight * pct, 0.0, 1.0
        ))

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
        sw = getattr(getattr(data.config, "factors", None), "sigmoid_weight", _DEFAULT_SIGMOID_WEIGHT)
        score = self._hybrid_score(current_value, available, sigmoid_weight=sw)

        return FactorResult(
            value=score,
            metadata={"raw": current_value},
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

        series = ratios[self.ratio_key].reindex(date_range).ffill()

        # Sigmoid component (vectorized)
        if self.higher_is_better:
            sig = _sigmoid_array(series, self.graham_threshold, -self.steepness)
        else:
            sig = _sigmoid_array(series, self.graham_threshold, self.steepness)

        # Percentile component (expanding window, no look-ahead)
        pct = series.expanding(min_periods=2).apply(
            lambda w: (w.iloc[:-1] < w.iloc[-1]).sum() / (len(w) - 1)
            if len(w) > 1
            else 0.5,
            raw=False,
        )
        if not self.higher_is_better:
            pct = 1.0 - pct

        sw = getattr(getattr(data.config, "factors", None), "sigmoid_weight", _DEFAULT_SIGMOID_WEIGHT)
        score = sw * sig + (1.0 - sw) * pct
        return score.clip(0.0, 1.0)


@register_factor
class GrahamPERatioFactor(GrahamFactor):
    """PE < 15 is cheap by Graham standards."""

    name = "graham_pe"
    description = "Graham PE ratio (PE < 15 = undervalued)"
    ratio_key = "pe_ratio"
    graham_threshold = 15.0
    steepness = 0.3
    higher_is_better = False


@register_factor
class PriceToBookFactor(GrahamFactor):
    """P/B < 1.5 is undervalued by Graham standards."""

    name = "price_to_book"
    description = "Price-to-Book ratio (P/B < 1.5 = undervalued)"
    ratio_key = "price_to_book"
    graham_threshold = 1.5
    steepness = 2.0
    higher_is_better = False


@register_factor
class CurrentRatioFactor(GrahamFactor):
    """Current ratio > 2.0 indicates financial safety."""

    name = "current_ratio"
    description = "Current ratio (CR > 2.0 = financially safe)"
    ratio_key = "current_ratio"
    graham_threshold = 2.0
    steepness = 2.0
    higher_is_better = True


@register_factor
class DividendYieldFactor(GrahamFactor):
    """High dividend yield (> 3%) signals cheap stock."""

    name = "dividend_yield"
    description = "Dividend yield (DY > 3% = stock is cheap)"
    ratio_key = "dividend_yield"
    graham_threshold = 0.03
    steepness = 60.0
    higher_is_better = True


@register_factor
class GrahamNumberFactor(Factor):
    """Graham Number: sqrt(22.5 * EPS * BVPS).

    Score = sigmoid(Price / GrahamNumber) blended with percentile rank.
    Ratio < 1.0 means the stock trades below intrinsic value.
    Returns 0.5 (neutral) when EPS or BVPS is negative (formula undefined).
    """

    name = "graham_number"
    description = "Price vs Graham Number (Price < GN = undervalued)"
    category = "traditional"

    def _graham_ratio(self, price: float, eps: float, bvps: float) -> Optional[float]:
        """Compute Price / GrahamNumber. Returns None if undefined."""
        if eps <= 0 or bvps <= 0:
            return None
        gn = np.sqrt(22.5 * eps * bvps)
        if gn <= 0:
            return None
        return price / gn

    def compute(
        self,
        ticker: str,
        as_of_date: date,
        data: DataProvider,
    ) -> Optional[FactorResult]:
        ratios = data.get_ratios(ticker)
        ohlc = data.get_ohlc(ticker)

        for key in ("earnings_per_share", "book_value_per_share"):
            if key not in ratios:
                return None

        ts = pd.Timestamp(as_of_date)
        eps_avail = ratios["earnings_per_share"].loc[:ts].dropna()
        bvps_avail = ratios["book_value_per_share"].loc[:ts].dropna()
        price_avail = ohlc["Close"].loc[:ts].dropna()

        if eps_avail.empty or bvps_avail.empty or price_avail.empty:
            return None

        eps = eps_avail.iloc[-1]
        bvps = bvps_avail.iloc[-1]
        price = price_avail.iloc[-1]

        ratio = self._graham_ratio(price, eps, bvps)
        if ratio is None:
            return FactorResult(value=0.5, metadata={"raw": None, "reason": "negative EPS or BVPS"})

        # Sigmoid: ratio < 1.0 -> score > 0.5 (undervalued)
        sig = _sigmoid(ratio, 1.0, 3.0)
        return FactorResult(value=float(np.clip(sig, 0.0, 1.0)), metadata={"raw": ratio})

    def compute_series(
        self,
        ticker: str,
        date_range: pd.DatetimeIndex,
        data: DataProvider,
    ) -> pd.Series:
        ratios = data.get_ratios(ticker)
        ohlc = data.get_ohlc(ticker)

        for key in ("earnings_per_share", "book_value_per_share"):
            if key not in ratios:
                return pd.Series(dtype=float, index=date_range)

        eps = ratios["earnings_per_share"].reindex(date_range).ffill()
        bvps = ratios["book_value_per_share"].reindex(date_range).ffill()
        price = ohlc["Close"].reindex(date_range).ffill()

        # Graham Number = sqrt(22.5 * EPS * BVPS)
        product = 22.5 * eps * bvps
        # Only valid where both EPS and BVPS are positive
        valid = product > 0
        gn = pd.Series(np.nan, index=date_range)
        gn[valid] = np.sqrt(product[valid])

        ratio = price / gn  # NaN where gn is NaN

        # Sigmoid: ratio < 1.0 -> score > 0.5
        sig = _sigmoid_array(ratio, 1.0, 3.0)

        # Percentile component on ratio values (lower ratio = more bullish)
        pct = ratio.expanding(min_periods=2).apply(
            lambda w: (w.iloc[:-1] < w.iloc[-1]).sum() / (len(w) - 1)
            if len(w) > 1
            else 0.5,
            raw=False,
        )
        pct = 1.0 - pct  # Lower ratio is better

        sw = getattr(getattr(data.config, "factors", None), "sigmoid_weight", _DEFAULT_SIGMOID_WEIGHT)
        score = sw * sig + (1.0 - sw) * pct
        # Fill invalid (negative EPS/BVPS) with 0.5 (neutral)
        score = score.fillna(0.5)
        return score.clip(0.0, 1.0)
