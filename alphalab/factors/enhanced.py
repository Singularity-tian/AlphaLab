"""Enhanced factors for high-alpha strategy.

Includes:
- Short-term momentum (3-month, 6-month)
- Earnings growth (YoY EPS growth)
- Revenue growth
- Composite quality score
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


@register_factor
class ShortMomentumFactor(Factor):
    """3-month momentum (63 trading days).
    
    Short-term momentum captures recent winners.
    Combined with 12-1 momentum, this creates a dual-momentum signal.
    """
    name = "momentum_3m"
    description = "3-month price momentum"
    category = "momentum"

    lookback_days: int = 63
    skip_days: int = 5  # skip last week to avoid microstructure noise

    def compute(self, ticker, as_of_date, data):
        ohlc = data.get_ohlc(ticker)
        ts = pd.Timestamp(as_of_date)
        available = ohlc.loc[:ts, "Close"]
        if len(available) < self.lookback_days:
            return None
        price_recent = available.iloc[-self.skip_days] if len(available) > self.skip_days else available.iloc[-1]
        price_past = available.iloc[-self.lookback_days]
        if price_past <= 0 or np.isnan(price_past):
            return None
        raw_return = (price_recent / price_past) - 1.0
        score = 1.0 / (1.0 + np.exp(-raw_return * 8))
        return FactorResult(value=float(np.clip(score, 0.0, 1.0)), metadata={"raw_return": raw_return})

    def compute_series(self, ticker, date_range, data):
        ohlc = data.get_ohlc(ticker)
        close = ohlc["Close"].reindex(date_range).ffill()
        price_past = close.shift(self.lookback_days)
        price_recent = close.shift(self.skip_days)
        raw_return = (price_recent / price_past) - 1.0
        score = 1.0 / (1.0 + np.exp(-raw_return * 8))
        return score.clip(0.0, 1.0)


@register_factor
class MediumMomentumFactor(Factor):
    """6-month momentum (126 trading days).
    
    Medium-term momentum - sweet spot between trend following and mean reversion.
    """
    name = "momentum_6m"
    description = "6-month price momentum"
    category = "momentum"

    lookback_days: int = 126
    skip_days: int = 10

    def compute(self, ticker, as_of_date, data):
        ohlc = data.get_ohlc(ticker)
        ts = pd.Timestamp(as_of_date)
        available = ohlc.loc[:ts, "Close"]
        if len(available) < self.lookback_days:
            return None
        price_recent = available.iloc[-self.skip_days] if len(available) > self.skip_days else available.iloc[-1]
        price_past = available.iloc[-self.lookback_days]
        if price_past <= 0 or np.isnan(price_past):
            return None
        raw_return = (price_recent / price_past) - 1.0
        score = 1.0 / (1.0 + np.exp(-raw_return * 6))
        return FactorResult(value=float(np.clip(score, 0.0, 1.0)), metadata={"raw_return": raw_return})

    def compute_series(self, ticker, date_range, data):
        ohlc = data.get_ohlc(ticker)
        close = ohlc["Close"].reindex(date_range).ffill()
        price_past = close.shift(self.lookback_days)
        price_recent = close.shift(self.skip_days)
        raw_return = (price_recent / price_past) - 1.0
        score = 1.0 / (1.0 + np.exp(-raw_return * 6))
        return score.clip(0.0, 1.0)


@register_factor
class VolatilityAdjustedMomentumFactor(Factor):
    """Momentum divided by volatility (Sharpe-like momentum).
    
    Favors stocks with strong, steady trends over volatile spikes.
    This is a key factor in many successful quant strategies.
    """
    name = "vol_adj_momentum"
    description = "Volatility-adjusted 6-month momentum (risk-adjusted trend)"
    category = "momentum"

    lookback_days: int = 126
    skip_days: int = 5

    def compute(self, ticker, as_of_date, data):
        ohlc = data.get_ohlc(ticker)
        ts = pd.Timestamp(as_of_date)
        available = ohlc.loc[:ts, "Close"]
        if len(available) < self.lookback_days:
            return None
        window = available.iloc[-self.lookback_days:-self.skip_days] if len(available) > self.skip_days else available.iloc[-self.lookback_days:]
        if len(window) < 20:
            return None
        returns = window.pct_change().dropna()
        if len(returns) < 10 or returns.std() <= 0:
            return None
        sharpe_like = returns.mean() / returns.std() * np.sqrt(252)
        # Sigmoid centered at 0, steepness 1.0
        score = 1.0 / (1.0 + np.exp(-sharpe_like * 0.8))
        return FactorResult(value=float(np.clip(score, 0.0, 1.0)), metadata={"sharpe_like": sharpe_like})

    def compute_series(self, ticker, date_range, data):
        ohlc = data.get_ohlc(ticker)
        close = ohlc["Close"].reindex(date_range).ffill()
        daily_ret = close.pct_change()
        
        # Rolling mean and std of returns
        roll_mean = daily_ret.rolling(window=self.lookback_days, min_periods=60).mean()
        roll_std = daily_ret.rolling(window=self.lookback_days, min_periods=60).std()
        
        sharpe_like = (roll_mean / roll_std) * np.sqrt(252)
        sharpe_like = sharpe_like.fillna(0)
        
        score = 1.0 / (1.0 + np.exp(-sharpe_like * 0.8))
        return score.clip(0.0, 1.0)


@register_factor
class EarningsGrowthFactor(Factor):
    """Year-over-year EPS growth rate.
    
    Companies with accelerating earnings tend to outperform.
    """
    name = "earnings_growth"
    description = "YoY EPS growth rate (higher = faster growing)"
    category = "traditional"

    def compute(self, ticker, as_of_date, data):
        ratios = data.get_ratios(ticker)
        if "earnings_per_share" not in ratios:
            return None
        ts = pd.Timestamp(as_of_date)
        eps = ratios["earnings_per_share"].loc[:ts].dropna()
        if len(eps) < 252:  # need at least 1 year
            return None
        current = eps.iloc[-1]
        year_ago = eps.iloc[-252] if len(eps) >= 252 else eps.iloc[0]
        if year_ago <= 0 or np.isnan(year_ago):
            return FactorResult(value=0.5, metadata={"raw": None})
        growth = (current / year_ago) - 1.0
        # Sigmoid: 0% growth -> 0.5, 50% growth -> ~0.85
        score = 1.0 / (1.0 + np.exp(-growth * 3))
        return FactorResult(value=float(np.clip(score, 0.0, 1.0)), metadata={"growth": growth})

    def compute_series(self, ticker, date_range, data):
        ratios = data.get_ratios(ticker)
        if "earnings_per_share" not in ratios:
            return pd.Series(dtype=float, index=date_range)
        eps = ratios["earnings_per_share"].reindex(date_range).ffill()
        eps_lagged = eps.shift(252)
        growth = (eps / eps_lagged) - 1.0
        # Handle division issues
        growth = growth.replace([np.inf, -np.inf], np.nan).fillna(0)
        score = 1.0 / (1.0 + np.exp(-growth * 3))
        return score.clip(0.0, 1.0)


@register_factor 
class PriceAccelerationFactor(Factor):
    """Price acceleration: is momentum accelerating or decelerating?
    
    Compares recent momentum to earlier momentum.
    Positive acceleration = trend is strengthening.
    """
    name = "price_acceleration"
    description = "Price momentum acceleration (strengthening trend)"
    category = "momentum"

    def compute(self, ticker, as_of_date, data):
        ohlc = data.get_ohlc(ticker)
        ts = pd.Timestamp(as_of_date)
        close = ohlc.loc[:ts, "Close"]
        if len(close) < 126:
            return None
        # Recent 3m return vs prior 3m return
        if len(close) >= 126:
            recent_3m = (close.iloc[-5] / close.iloc[-63]) - 1.0 if len(close) > 63 else 0
            prior_3m = (close.iloc[-63] / close.iloc[-126]) - 1.0
            acceleration = recent_3m - prior_3m
        else:
            return None
        score = 1.0 / (1.0 + np.exp(-acceleration * 5))
        return FactorResult(value=float(np.clip(score, 0.0, 1.0)), metadata={"acceleration": acceleration})

    def compute_series(self, ticker, date_range, data):
        ohlc = data.get_ohlc(ticker)
        close = ohlc["Close"].reindex(date_range).ffill()
        # 3m return ending 5 days ago
        recent_3m = (close.shift(5) / close.shift(63)) - 1.0
        # Prior 3m return
        prior_3m = (close.shift(63) / close.shift(126)) - 1.0
        acceleration = recent_3m - prior_3m
        acceleration = acceleration.fillna(0)
        score = 1.0 / (1.0 + np.exp(-acceleration * 5))
        return score.clip(0.0, 1.0)
