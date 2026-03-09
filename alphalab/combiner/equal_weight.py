"""Equal-weight factor combiner."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from alphalab.config import AlphaLabConfig
from alphalab.factors.base import Factor

if TYPE_CHECKING:
    from alphalab.data.providers import DataProvider

logger = logging.getLogger(__name__)


class EqualWeightCombiner:
    """Combines factor scores into a trading signal using weighted average."""

    def __init__(self, config: AlphaLabConfig, data: DataProvider):
        self.config = config
        self.data = data

    def combine(
        self,
        ticker: str,
        factors: list[Factor],
        date_range: pd.DatetimeIndex,
    ) -> pd.Series:
        """Compute weighted average of factor scores in [0, 1]."""
        weights = self.config.factors.weights
        factor_scores: dict[str, pd.Series] = {}

        for factor in factors:
            try:
                scores = factor.compute_series(ticker, date_range, self.data)
                factor_scores[factor.name] = scores
                logger.info(
                    "Factor %s: mean=%.3f, non-null=%d/%d",
                    factor.name,
                    scores.mean(),
                    scores.notna().sum(),
                    len(scores),
                )
            except Exception as e:
                logger.error("Factor %s failed: %s", factor.name, e)

        if not factor_scores:
            raise ValueError("All factors failed. Cannot combine.")

        scores_df = pd.DataFrame(factor_scores)

        # Build weight vector
        if weights:
            w = pd.Series({name: weights.get(name, 1.0) for name in scores_df.columns})
        else:
            w = pd.Series(1.0, index=scores_df.columns)
        w = w / w.sum()

        # Weighted mean, re-weighting across available (non-NaN) factors per row
        def _row_avg(row: pd.Series) -> float:
            valid = row.dropna()
            if valid.empty:
                return np.nan
            row_w = w[valid.index]
            row_w = row_w / row_w.sum()
            return float(np.average(valid, weights=row_w))

        combined = scores_df.apply(_row_avg, axis=1)
        logger.info("Combined signal: mean=%.3f, std=%.3f", combined.mean(), combined.std())
        return combined

    def combine_to_signal(
        self,
        ticker: str,
        factors: list[Factor],
        date_range: pd.DatetimeIndex,
    ) -> pd.Series:
        """Convert combined scores into a trading signal.

        Returns pd.Series with values:
        - > 0: long signal (magnitude = position fraction)
        - 0: flat
        - < 0: short signal
        """
        combined = self.combine(ticker, factors, date_range)

        long_thresh = self.config.backtest.long_threshold
        short_thresh = self.config.backtest.short_threshold

        signal = pd.Series(0.0, index=combined.index)

        long_mask = combined > long_thresh
        signal[long_mask] = (combined[long_mask] - long_thresh) / (1.0 - long_thresh)

        short_mask = combined < short_thresh
        signal[short_mask] = -(short_thresh - combined[short_mask]) / short_thresh

        # Monthly rebalancing: only update signal on month boundaries
        if self.config.backtest.rebalance_frequency == "monthly":
            rebalance_mask = combined.index.is_month_start | (
                combined.index == combined.index[0]
            )
            rebalance_signal = signal.where(rebalance_mask)
            signal = rebalance_signal.ffill().fillna(0.0)

        return signal
