"""Configuration management for AlphaLab."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


class DataConfig(BaseSettings):
    model_config = {"populate_by_name": True}

    fmp_api_key: str = Field(default="", alias="FMP_API_KEY")
    cache_dir: Path = Path("data/cache")
    cache_ttl_hours: int = 24


class BacktestConfig(BaseSettings):
    initial_cash: float = 100_000.0
    commission: float = 0.001
    start_date: str = "2019-01-01"
    end_date: str = "2024-12-31"
    rebalance_frequency: Literal["monthly", "weekly", "daily"] = "monthly"
    long_threshold: float = 0.6
    short_threshold: float = 0.4

    # Position sizing
    signal_weighted: bool = True
    max_position_weight: float = 0.25
    max_positions: int = 15
    min_positions: int = 3
    cash_reserve_pct: float = 0.05

    # Fixed holding period
    holding_period_days: int = 252

    # Risk management
    trailing_stop_pct: float = 0.20

    # Scale-in
    scale_in: bool = True
    scale_in_pct: float = 0.50
    scale_in_days: int = 5


class LLMConfig(BaseSettings):
    model_config = {"populate_by_name": True}

    primary_provider: Literal["gemini", "claude"] = "gemini"
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = "gemini-2.0-flash"
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    anthropic_model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.1
    max_retries: int = 3


class FactorConfig(BaseSettings):
    factors: list[str] = [
        "graham_pe",
        "price_to_book",
        "graham_number",
        "current_ratio",
        "dividend_yield",
    ]
    weights: dict[str, float] = {}
    sigmoid_weight: float = 0.4  # Graham hybrid: sigmoid vs percentile blend


class AlphaLabConfig(BaseSettings):
    model_config = {"env_prefix": "ALPHALAB_", "env_nested_delimiter": "__", "populate_by_name": True}

    data: DataConfig = DataConfig()
    backtest: BacktestConfig = BacktestConfig()
    llm: LLMConfig = LLMConfig()
    factors: FactorConfig = FactorConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> AlphaLabConfig:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls(**raw)

    @classmethod
    def default(cls) -> AlphaLabConfig:
        default_path = Path("configs/default.yaml")
        if default_path.exists():
            return cls.from_yaml(default_path)
        return cls()
