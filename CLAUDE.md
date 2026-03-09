# CLAUDE.md — Agent Instructions for AlphaLab

## What This Project Is

AlphaLab is an **infrastructure for LLM-powered factor investing**. It's a research platform where AI agents and humans collaborate to discover, test, and deploy trading strategies. The main goal is to **leverage LLMs in factor creation** — both generating novel factor formulas and providing judgment-based scores on financial data.

Think of it as the R&D lab for an AI-native hedge fund. Many components (RL loops, new factor types, portfolio optimization, live trading) are not yet built — this is active research infrastructure.

## Project Layout

```
alphalab/                     # Core library
  factors/                    # Factor computation (THE CORE)
    base.py                   # Factor ABC, FactorResult(value, confidence, metadata)
    registry.py               # @register_factor, get_factor(), list_factors()
    graham.py                 # 5 Graham value factors (hybrid sigmoid + percentile)
    traditional.py            # PE, ROE, FCF Yield (expanding-window percentile rank)
    momentum.py               # 12-1 price momentum
    llm_judgment.py           # Anonymized earnings → LLM → sentiment score
  data/
    providers.py              # DataProvider: yfinance (OHLC) + FMP stable API (financials)
    cache.py                  # File cache with TTL (Parquet for DataFrames, JSON for dicts)
  combiner/
    equal_weight.py           # Factor scores → weighted avg → trading signal
  backtest/
    runner.py                 # BacktestRunner + FactorStrategy (backtesting.py)
    report.py                 # Charts, metrics, factor attribution
  anonymizer/
    pipeline.py               # Entity/temporal/product masking for LLM factors
    verification.py           # Leak detection
  llm/
    client.py                 # Gemini/Claude unified client
    prompts.py                # Structured prompts for sentiment scoring
  config.py                   # Pydantic settings (YAML + env vars)

configs/default.yaml          # Default configuration
examples/                     # Entry points and tutorials
tests/                        # pytest suite (31 tests)
reports/                      # Generated backtest outputs
data/cache/                   # Cached API responses (gitignored)
```

## Key Concepts

### Factor Pipeline
1. **Factors** compute normalized scores in [0, 1] where 1.0 = bullish, 0.0 = bearish
2. **Combiner** takes weighted average of factor scores → single combined score [0, 1]
3. **Signal generator** applies thresholds: score > 0.6 → long signal, score < 0.4 → short signal
4. **Strategy** executes: signal > 0 → buy (long-only), signal ≤ 0 → sell/stay flat
5. **Monthly rebalancing**: signals only update on month-start dates

### Factor Types
- **Traditional formula factors**: Compute ratios from financial statements (PE, ROE, etc.)
- **Graham factors**: Hybrid normalization = 40% sigmoid(raw, graham_threshold) + 60% percentile_rank
- **Momentum factors**: Price-based signals from OHLC data
- **LLM judgment factors**: Send anonymized text to LLM, get structured sentiment scores
- **LLM innovative factors**: (PLANNED) LLM generates novel factor formulas

### Data Sources
- **yfinance**: Free OHLC prices and dividends (reliable, no API key needed)
- **FMP stable API**: Quarterly financial statements — income, balance sheet, cash flow. Up to 80 quarters (20 years). Base URL: `https://financialmodelingprep.com/stable/`. Requires `FMP_API_KEY` env var. Starter plan ($22/mo).
- **Gemini/Claude**: LLM providers for judgment factors and future formula generation

### How Factors Are Normalized
- **TraditionalFactor** (traditional.py): expanding-window percentile rank. No look-ahead bias.
- **GrahamFactor** (graham.py): `0.4 * sigmoid(raw, threshold, steepness) + 0.6 * percentile_rank`. Anchors in absolute Graham thresholds while adapting to relative history.
- All factors MUST return values in [0, 1]. Factors return `FactorResult(value=..., confidence=..., metadata={...})`.

## How to Run

```bash
source .venv/bin/activate

# Single stock backtest
python examples/quickstart.py --no-llm

# Tests
pytest tests/ -v
```

Environment variables needed (put in `.env`):
```
FMP_API_KEY=...           # Required for financial data
GEMINI_API_KEY=...        # Optional, for LLM factors
ANTHROPIC_API_KEY=...     # Optional, LLM fallback
```

## How to Create a New Factor

```python
from alphalab.factors.base import Factor, FactorResult
from alphalab.factors.registry import register_factor

@register_factor
class MyFactor(Factor):
    name = "my_factor"            # Unique identifier
    description = "What it does"
    category = "traditional"       # or "momentum", "llm", etc.

    def compute(self, ticker, as_of_date, data):
        # data is a DataProvider instance
        ratios = data.get_ratios(ticker)        # dict[str, pd.Series]
        ohlc = data.get_ohlc(ticker)            # DataFrame with Open/High/Low/Close/Volume
        # ... your logic ...
        return FactorResult(value=score_0_to_1, metadata={"raw": raw_value})

    def compute_series(self, ticker, date_range, data):
        # Optional override for vectorized computation
        # Must return pd.Series with values in [0, 1]
        ...
```

Add the factor name to `configs/default.yaml` under `factors.factors` to include it in backtests.

## Available Data from DataProvider

```python
data.get_ohlc(ticker)       # pd.DataFrame: Open, High, Low, Close, Volume (daily)
data.get_ratios(ticker)     # dict[str, pd.Series] with keys:
                            #   earnings_per_share, book_value_per_share,
                            #   current_ratio, pe_ratio, price_to_book,
                            #   dividend_yield, roe, fcf_yield
                            # All forward-filled to daily frequency
data.get_earnings_transcripts(ticker, year, quarter)  # str or None
data.get_available_transcript_dates(ticker)            # list[(year, quarter)]
```

## Configuration (configs/default.yaml)

```yaml
data:
  cache_dir: "data/cache"
  cache_ttl_hours: 24

backtest:
  initial_cash: 100000
  commission: 0.001
  start_date: "2019-01-01"
  end_date: "2024-12-31"
  rebalance_frequency: "monthly"    # monthly | weekly | daily
  long_threshold: 0.6              # combined score > this → buy
  short_threshold: 0.4              # combined score < this → sell (if not long-only)

factors:
  factors: [graham_pe, price_to_book, graham_number, current_ratio, dividend_yield]
  weights: {}                       # empty = equal weight
```

## What's Not Built Yet (Opportunities)

- **LLM factor formula generation**: LLM proposes new ratio formulas, backtested automatically
- **Factor weight optimization**: Learn optimal factor weights from historical performance
- **Reinforcement learning loop**: Backtest results feed back to improve factors
- **Multi-stock portfolio mode**: Single portfolio across stock universe with capital allocation
- **Risk management**: Stop-loss, take-profit, max position size, sector limits
- **Factor decay detection**: Monitor when factors lose predictive power over time
- **Forward testing**: Paper trading with live data feed
- **Live execution**: Alpaca / IBKR integration
- **Alternative data factors**: News sentiment, social media, satellite, web traffic
- **Cross-sectional factors**: Rank stocks relative to each other, not just own history

## Code Conventions

- All factor scores are normalized to [0, 1]
- Factor registration uses `@register_factor` decorator
- DataProvider results are cached automatically (Parquet for DataFrames, JSON for metadata)
- Configuration flows from YAML → Pydantic settings → env var override
- Tests use mock DataProvider (no network calls). Fixtures in `tests/conftest.py`
- Factor modules must be imported to register — `alphalab/factors/__init__.py` handles this

## Testing

```bash
pytest tests/ -v          # 31 tests, ~0.5s
pytest tests/ -k graham   # just Graham factor tests
```

Tests use synthetic OHLC data and mock DataProvider. No API keys needed for tests.
