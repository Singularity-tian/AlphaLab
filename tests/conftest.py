"""Shared test fixtures."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from alphalab.config import AlphaLabConfig, BacktestConfig, DataConfig


@pytest.fixture
def config():
    return AlphaLabConfig(
        data=DataConfig(fmp_api_key="test_key", cache_dir="/tmp/alphalab_test_cache"),
        backtest=BacktestConfig(start_date="2023-01-01", end_date="2023-12-31"),
    )


@pytest.fixture
def sample_ohlc():
    """Synthetic OHLC data for testing."""
    dates = pd.bdate_range("2023-01-01", "2023-12-31")
    np.random.seed(42)
    n = len(dates)

    close = 150.0 + np.cumsum(np.random.randn(n) * 2)
    return pd.DataFrame(
        {
            "Open": close + np.random.randn(n) * 0.5,
            "High": close + abs(np.random.randn(n)) * 1.5,
            "Low": close - abs(np.random.randn(n)) * 1.5,
            "Close": close,
            "Volume": np.random.randint(1_000_000, 10_000_000, n),
        },
        index=dates,
    )


@pytest.fixture
def mock_data_provider(config, sample_ohlc):
    """Mock DataProvider returning synthetic data."""
    provider = MagicMock()
    provider.config = config

    provider.get_ohlc.return_value = sample_ohlc
    provider.get_ratios.return_value = {
        "pe_ratio": pd.Series(
            np.random.uniform(15, 35, len(sample_ohlc)),
            index=sample_ohlc.index,
        ),
        "roe": pd.Series(
            np.random.uniform(0.1, 0.4, len(sample_ohlc)),
            index=sample_ohlc.index,
        ),
        "fcf_yield": pd.Series(
            np.random.uniform(0.02, 0.08, len(sample_ohlc)),
            index=sample_ohlc.index,
        ),
        "price_to_book": pd.Series(
            np.random.uniform(0.8, 4.0, len(sample_ohlc)),
            index=sample_ohlc.index,
        ),
        "current_ratio": pd.Series(
            np.random.uniform(0.5, 3.5, len(sample_ohlc)),
            index=sample_ohlc.index,
        ),
        "dividend_yield": pd.Series(
            np.random.uniform(0.005, 0.06, len(sample_ohlc)),
            index=sample_ohlc.index,
        ),
        "earnings_per_share": pd.Series(
            np.random.uniform(2.0, 8.0, len(sample_ohlc)),
            index=sample_ohlc.index,
        ),
        "book_value_per_share": pd.Series(
            np.random.uniform(10.0, 40.0, len(sample_ohlc)),
            index=sample_ohlc.index,
        ),
    }
    return provider
