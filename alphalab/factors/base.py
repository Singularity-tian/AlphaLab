"""Abstract base class for all factors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any, Literal, Optional

import pandas as pd

if TYPE_CHECKING:
    from alphalab.data.providers import DataProvider


@dataclass
class FactorResult:
    """Result of a single factor computation."""

    value: float  # Normalized score in [0, 1]
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


class Factor(ABC):
    """Base class for all factors.

    All factors output scores normalized to [0, 1] where:
    - 1.0 = strongly bullish
    - 0.5 = neutral
    - 0.0 = strongly bearish
    """

    name: str
    description: str
    category: Literal["traditional", "momentum", "llm_formula", "llm_judgment"]

    @abstractmethod
    def compute(
        self,
        ticker: str,
        as_of_date: date,
        data: DataProvider,
    ) -> Optional[FactorResult]:
        """Compute factor value for a single date."""
        ...

    def compute_series(
        self,
        ticker: str,
        date_range: pd.DatetimeIndex,
        data: DataProvider,
    ) -> pd.Series:
        """Compute factor over a date range. Default: loop over compute()."""
        results = {}
        for dt in date_range:
            result = self.compute(ticker, dt.date(), data)
            if result is not None:
                results[dt] = result.value
        series = pd.Series(results, dtype=float)
        return series.reindex(date_range)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
