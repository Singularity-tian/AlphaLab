# AlphaLab

**Infrastructure for LLM-powered factor investing — research, backtest, and deploy trading strategies where LLMs are first-class participants in alpha generation.**

AlphaLab is not just another backtester. It's a research platform designed for a future where LLMs actively contribute to investment decisions — generating novel factor formulas, analyzing earnings transcripts without memorization bias, and collaborating with human portfolio managers. Think of it as the R&D lab for an AI-native hedge fund.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Factor Layer                          │
│  ┌──────────────────┐  ┌──────────────────┐             │
│  │ Traditional       │  │ LLM Innovative   │             │
│  │ Formula Factors   │  │ Formula Factors   │  ◄── LLM   │
│  │ (PE, P/B, Graham) │  │ (LLM-generated)  │    designs  │
│  └────────┬─────────┘  └────────┬─────────┘    new      │
│           │                     │               formulas │
│  ┌────────┴─────────────────────┘                       │
│  │  ┌──────────────────┐                                │
│  │  │ LLM Judgment      │                               │
│  │  │ Factors            │ ◄── Anonymized transcripts    │
│  │  │ (Sentiment scores) │     prevent memorization      │
│  │  └────────┬──────────┘                               │
│  └───────────┤                                          │
└──────────────┼──────────────────────────────────────────┘
               ▼
      ┌────────────────┐       ┌──────────────────┐
      │ Weighted        │ ◄───▶│ Backtest +        │
      │ Combination     │      │ Optimization      │
      └────────┬───────┘       └──────────────────┘
               ▼                  ▲  (feedback loop)
      ┌────────────────┐          │
      │ LLM + Human    ├──────────┘
      │ Decision        │
      └────────┬───────┘
               ▼
      ┌────────────────┐
      │ Live Trading    │  (planned)
      └────────────────┘
```

## Three Types of Factors

| Type | What it does | Example | Status |
|------|-------------|---------|--------|
| **Traditional Formula** | Classic quant factors from financial data | Graham PE < 15, Price-to-Book, Current Ratio | Implemented |
| **LLM Innovative Formula** | LLM generates novel factor formulas from data patterns | LLM proposes new ratio combinations, screens | Planned |
| **LLM Judgment (Anonymized)** | LLM scores anonymized earnings transcripts | Sentiment, guidance strength, risk assessment | Implemented |

The anonymization pipeline prevents LLM memorization: "Apple Inc." becomes "COMPANY_A", "Tim Cook" becomes "PERSON_A", "Q3 2023" becomes "QUARTER_N". The LLM evaluates content on its merits, not its memory.


## Quick Start

```bash
git clone <repo-url> && cd AlphaLab
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env  # add your FMP_API_KEY
```

Run a single-stock backtest:
```bash
python strategies/graham_value/run.py
python strategies/graham_value/run.py --ticker MSFT
```

Run a portfolio backtest:
```bash
python strategies/graham_value/portfolio_run.py      # S&P 500 portfolio
python strategies/expma_kdj/run.py                   # EXPMA+KDJ hourly strategy
```


## Project Structure

```
data/                             # Shared data storage
  sp500_daily/                    # Daily OHLC + financials
    download.py                   # Download script (yfinance + FMP)
    cache/                        # Cached API responses (gitignored)
  sp500_hourly/                   # Hourly OHLC data
    download.py                   # Download script (FMP hourly API)
    cache/                        # Parquet files per ticker (gitignored)

alphalab/                         # Shared infrastructure (strategy-agnostic)
  factors/                        # Factor computation
    base.py                       # Factor ABC, FactorResult dataclass
    registry.py                   # @register_factor decorator, factor lookup
    traditional.py                # PE, ROE, FCF Yield (percentile rank)
    momentum.py                   # 12-1 momentum factor
    llm_judgment.py               # LLM sentiment on anonymized transcripts
    business_resilience.py        # Two-stage LLM business resilience scoring
  data/
    providers.py                  # DataProvider: yfinance OHLC + FMP financials
    cache.py                      # File-based cache (Parquet + JSON, TTL expiry)
    tickers.py                    # Stock universes (SP500 list)
  combiner/
    equal_weight.py               # Weighted avg → threshold → trading signal
  backtest/
    runner.py                     # BacktestRunner + FactorStrategy (backtesting.py)
    portfolio_runner.py           # PortfolioBacktestRunner (multi-stock)
    report.py                     # Charts, metrics, factor attribution
    portfolio_report.py           # Portfolio HTML report
  anonymizer/
    pipeline.py                   # 3-layer entity/temporal/product masking
    verification.py               # Leak detection on anonymized text
  llm/
    client.py                     # Gemini/Claude unified client with retries
    prompts.py                    # Structured prompts for sentiment scoring
  optimizer/                      # Generic hyperparameter optimization (Optuna)
    runner.py                     # Study orchestration
    objective.py                  # Backtest reward function
    apply.py                      # Write best params to YAML
    report.py                     # Print optimization results
  config.py                       # Pydantic settings (YAML + env vars)

strategies/                       # Independent strategy modules
  graham_value/                   # Benjamin Graham value investing strategy
    factors.py                    # 5 Graham factors (hybrid sigmoid + percentile)
    configs/                      # Strategy configs (default.yaml, optimized.yaml)
    run.py                        # Single-stock backtest entry point
    portfolio_run.py              # Full S&P 500 portfolio backtest
    batch_run.py                  # Quick summary on top 100 stocks
    batch_run_detailed.py         # Detailed trade-level analysis
    optimize.py                   # Bayesian parameter tuning (owns search space)
    reports/                      # Generated outputs (gitignored)
    tests/                        # Strategy-specific tests
  expma_kdj/                      # EXPMA(12) & KDJ hourly technical strategy
    strategy.py                   # Core strategy logic
    run.py                        # Entry point
    report.py                     # HTML report generator
    configs/                      # Strategy configs (default.yaml)
    reports/                      # Backtest output (gitignored)

tests/                            # Shared infrastructure tests
  conftest.py                     # Shared fixtures (mock data, synthetic OHLC)
  test_factors.py                 # Traditional + momentum factor tests
  test_backtest.py                # Combiner + strategy integration tests
  test_anonymizer.py              # Anonymization pipeline tests
  test_business_resilience.py     # Business resilience factor tests
```

## Add Your Own Factor

```python
from alphalab.factors.base import Factor, FactorResult
from alphalab.factors.registry import register_factor

@register_factor
class MyFactor(Factor):
    name = "my_factor"
    description = "My custom factor"
    category = "traditional"

    def compute(self, ticker, as_of_date, data):
        ratios = data.get_ratios(ticker)
        raw = ratios.get("pe_ratio")
        # your logic here — return normalized [0,1] score
        return FactorResult(value=0.7, metadata={"raw": raw})
```

For generic factors, add them to `alphalab/factors/`. For strategy-specific factor groups, create a new folder under `strategies/` with its own `factors.py`, `configs/`, and `run.py`. Each strategy entry point imports its own factors to trigger registration.

## Data Sources

| Source | What | Plan |
|--------|------|------|
| **yfinance** | Daily OHLC prices, dividends | Free |
| **FMP Stable API** | Quarterly financials (80 quarters / 20 years) — income, balance sheet, cash flow | Starter ($22/mo) |
| **Gemini / Claude** | LLM judgment scoring | API keys |

## API Keys

| Key | Purpose | Required? |
|-----|---------|-----------|
| `FMP_API_KEY` | Financial statements (20yr quarterly data) | Yes |
| `GEMINI_API_KEY` | LLM sentiment factor | Optional |
| `ANTHROPIC_API_KEY` | LLM fallback provider | Optional |

## What's Built vs What's Planned

### Built
- [x] Factor framework with registry, normalization, and extensible base classes
- [x] Graham value strategy (5 factors, hybrid sigmoid + percentile normalization)
- [x] EXPMA(12) & KDJ hourly technical strategy
- [x] Traditional factors (PE, ROE, FCF Yield) with expanding-window percentile rank
- [x] Momentum factor (12-1 price momentum)
- [x] LLM judgment factor (anonymized earnings transcript sentiment)
- [x] LLM business resilience factor (two-stage scoring)
- [x] Anonymization pipeline (entity, temporal, product masking + verification)
- [x] Data pipeline (yfinance OHLC + FMP 20-year quarterly financials + caching)
- [x] Backtesting engine (single-stock and portfolio, long-only, monthly rebalance)
- [x] Bayesian hyperparameter optimization (Optuna, walk-forward validation)
- [x] Report generation (charts, metrics, factor attribution, HTML reports)

### Planned
- [ ] **LLM factor formula generation** — LLM proposes novel factor formulas from data patterns
- [ ] **Reinforcement learning loop** — backtest feedback → factor improvement
- [ ] **Factor decay monitoring** — detect when factors lose predictive power
- [ ] **Forward testing** — paper trading with live data
- [ ] **Live trading integration** — Alpaca / IBKR execution
- [ ] **More factor categories** — quality, growth, sentiment, macro, alternative data

## Development

```bash
pip install -e ".[dev]"
pytest tests/ strategies/ -v               # all 43 tests
python strategies/graham_value/run.py      # single stock backtest
python strategies/graham_value/optimize.py # Bayesian parameter tuning
```

Download data for offline use:
```bash
python data/sp500_daily/download.py
python data/sp500_hourly/download.py
```

## License

MIT
