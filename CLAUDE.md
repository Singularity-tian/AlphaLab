# CLAUDE.md — Agent Instructions for AlphaLab

## What This Project Is

AlphaLab is an **infrastructure for LLM-powered factor investing**. It's a research platform where AI agents and humans collaborate to discover, test, and deploy trading strategies. The main goal is to **leverage LLMs in factor creation** — both generating novel factor formulas and providing judgment-based scores on financial data.

Think of it as the R&D lab for an AI-native hedge fund. Many components (RL loops, new factor types, portfolio optimization, live trading) are not yet built — this is active research infrastructure.

## Project Layout

```
data/                             # Shared data storage
  sp500_daily/                    # Daily OHLC + financials
    download.py                   # Download script (yfinance + FMP)
    cache/                        # Cached API responses (gitignored)
  sp500_hourly/                   # Hourly OHLC data
    download.py                   # Download script (FMP hourly API)
    cache/                        # Parquet files per ticker (gitignored)

alphalab/                         # Shared infrastructure (strategy-agnostic)
  factors/                        # Factor computation (THE CORE)
    base.py                       # Factor ABC, FactorResult(value, confidence, metadata)
    registry.py                   # @register_factor, get_factor(), list_factors()
    traditional.py                # PE, ROE, FCF Yield (expanding-window percentile rank)
    momentum.py                   # 12-1 price momentum
    llm_judgment.py               # Anonymized earnings → LLM → sentiment score
    business_resilience.py        # Two-stage LLM business resilience scoring
  data/
    providers.py                  # DataProvider: yfinance (OHLC) + FMP stable API (financials)
    cache.py                      # File cache with TTL (Parquet for DataFrames, JSON for dicts)
    tickers.py                    # Stock universes (SP500 list)
  combiner/
    equal_weight.py               # Factor scores → weighted avg → trading signal
  backtest/
    runner.py                     # BacktestRunner + FactorStrategy (backtesting.py)
    portfolio_runner.py           # PortfolioBacktestRunner (multi-stock)
    report.py                     # Charts, metrics, factor attribution
    portfolio_report.py           # Portfolio HTML report
  anonymizer/
    pipeline.py                   # Entity/temporal/product masking for LLM factors
    verification.py               # Leak detection
  llm/
    client.py                     # Gemini/Claude unified client
    prompts.py                    # Structured prompts for sentiment scoring
  optimizer/                      # Generic hyperparameter optimization
    runner.py                     # Optuna study orchestration
    objective.py                  # Backtest reward function (accepts strategy search space)
    apply.py                      # Write best params to YAML
    report.py                     # Print optimization results
  config.py                       # Pydantic settings (YAML + env vars)

strategies/                       # Independent strategy modules
  graham_value/                   # Benjamin Graham value investing strategy
    factors.py                    # 5 Graham factors (hybrid sigmoid + percentile)
    configs/                      # Strategy configs
      default.yaml                # Default parameters
      optimized.yaml              # Optimized parameters (from optimizer)
    run.py                        # Single-stock backtest entry point
    portfolio_run.py              # Full S&P 500 portfolio backtest
    batch_run.py                  # Quick summary on top 100 stocks
    batch_run_detailed.py         # Detailed trade-level analysis
    optimize.py                   # Bayesian parameter tuning (owns search space)
    reports/                      # Generated backtest outputs (gitignored)
    tests/                        # Strategy-specific tests
      test_graham_factors.py
  expma_kdj/                      # EXPMA(12) & KDJ hourly technical strategy
    strategy.py                   # Core strategy logic
    run.py                        # Entry point
    report.py                     # HTML report generator
    configs/                      # Strategy configs
      default.yaml                # Backtest parameters
    reports/                      # Backtest output (gitignored)

tests/                            # Shared infrastructure tests
  conftest.py                     # Pytest fixtures & mock DataProvider
  test_factors.py                 # Traditional + momentum factor tests
  test_backtest.py                # Backtesting infrastructure tests
  test_anonymizer.py              # Anonymization tests
  test_business_resilience.py     # Business resilience factor tests
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
- **GrahamFactor** (strategies/graham_value/factors.py): `0.4 * sigmoid(raw, threshold, steepness) + 0.6 * percentile_rank`. Anchors in absolute Graham thresholds while adapting to relative history.
- All factors MUST return values in [0, 1]. Factors return `FactorResult(value=..., confidence=..., metadata={...})`.

## How to Run

```bash
source .venv/bin/activate

# Graham value strategy
python strategies/graham_value/run.py                # single stock
python strategies/graham_value/run.py --ticker MSFT
python strategies/graham_value/portfolio_run.py      # full S&P 500 portfolio
python strategies/graham_value/batch_run.py          # quick summary on 100 stocks
python strategies/graham_value/optimize.py           # Bayesian parameter tuning

# EXPMA+KDJ hourly strategy
python strategies/expma_kdj/run.py

# Download data for offline use
python data/sp500_daily/download.py
python data/sp500_hourly/download.py

# Tests
pytest tests/ strategies/ -v
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

For basic/generic factors, put them in `alphalab/factors/`. For cohesive strategy-specific factor groups, create a new folder under `strategies/` (e.g., `strategies/my_strategy/factors.py`) with its own `configs/default.yaml` and `run.py`. Each strategy entry point imports its own factors to trigger registration.

Add the factor name to your strategy's `configs/default.yaml` under `factors.factors` to include it in backtests.

## Strategy Template

Each strategy under `strategies/` should follow this structure:
```
strategies/my_strategy/
  __init__.py
  factors.py              # Factor definitions with @register_factor
  configs/
    default.yaml          # Default parameters
    optimized.yaml        # Optimized parameters (optional)
  run.py                  # Entry point (imports own factors)
  optimize.py             # Bayesian tuning (defines own search space)
  reports/                # Generated outputs (gitignored)
  tests/                  # Strategy-specific tests
```

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

## Configuration (strategies/graham_value/configs/default.yaml)

```yaml
data:
  cache_dir: "data/sp500_daily/cache"
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
- **Risk management**: Stop-loss, take-profit, max position size, sector limits
- **Factor decay detection**: Monitor when factors lose predictive power over time
- **Forward testing**: Paper trading with live data feed
- **Live execution**: Alpaca / IBKR integration
- **Alternative data factors**: News sentiment, social media, satellite, web traffic
- **Cross-sectional factors**: Rank stocks relative to each other, not just own history

## Code Conventions

- All factor scores are normalized to [0, 1]
- Factor registration uses `@register_factor` decorator
- Strategy-specific factors are imported by their own entry points (not by alphalab/)
- DataProvider results are cached automatically (Parquet for DataFrames, JSON for metadata)
- Configuration flows from YAML → Pydantic settings → env var override
- Tests use mock DataProvider (no network calls). Fixtures in `tests/conftest.py`
- Shared infra tests in `tests/`, strategy tests in `strategies/*/tests/`

## Testing

```bash
pytest tests/ strategies/ -v    # All tests
pytest tests/ -v                # Shared infra tests only
pytest strategies/graham_value/tests/ -v   # Graham strategy tests only
```

Tests use synthetic OHLC data and mock DataProvider. No API keys needed for tests.
