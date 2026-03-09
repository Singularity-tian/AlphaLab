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
python examples/quickstart.py --no-llm
```


## Project Structure

```
alphalab/
├── factors/              # Factor computation
│   ├── base.py           # Factor ABC, FactorResult dataclass
│   ├── registry.py       # @register_factor decorator, factor lookup
│   ├── graham.py         # 5 Graham value factors (hybrid normalization)
│   ├── traditional.py    # PE, ROE, FCF Yield (percentile rank)
│   ├── momentum.py       # 12-1 momentum factor
│   └── llm_judgment.py   # LLM sentiment on anonymized transcripts
├── data/
│   ├── providers.py      # DataProvider: yfinance OHLC + FMP financials
│   └── cache.py          # File-based cache (Parquet + JSON, TTL expiry)
├── combiner/
│   └── equal_weight.py   # Weighted avg → threshold → trading signal
├── backtest/
│   ├── runner.py         # BacktestRunner + FactorStrategy (backtesting.py)
│   └── report.py         # Charts, metrics, factor attribution
├── anonymizer/
│   ├── pipeline.py       # 3-layer entity/temporal/product masking
│   └── verification.py   # Leak detection on anonymized text
├── llm/
│   ├── client.py         # Gemini/Claude unified client with retries
│   └── prompts.py        # Structured prompts for sentiment scoring
└── config.py             # Pydantic settings (YAML + env vars)

examples/
├── quickstart.py         # Single-stock backtest entry point
├── batch_backtest.py     # 100-stock summary scan
├── batch_detailed.py     # 100-stock with full trade accounting
└── add_custom_factor.py  # Tutorial: create a custom RSI factor

tests/
├── conftest.py           # Shared fixtures (mock data, synthetic OHLC)
├── test_factors.py       # Factor output validation (31 tests)
├── test_backtest.py      # Combiner + strategy integration tests
└── test_anonymizer.py    # Anonymization pipeline tests

configs/
└── default.yaml          # Default backtest/factor/LLM configuration
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

See `examples/add_custom_factor.py` for a complete RSI example.

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
- [x] Traditional factors (PE, ROE, FCF Yield) with expanding-window percentile rank
- [x] Momentum factor (12-1 price momentum)
- [x] LLM judgment factor (anonymized earnings transcript sentiment)
- [x] Anonymization pipeline (entity, temporal, product masking + verification)
- [x] Data pipeline (yfinance OHLC + FMP 20-year quarterly financials + caching)
- [x] Backtesting engine (backtesting.py integration, long-only, monthly rebalance)
- [x] Batch backtesting across 100 stocks with full trade accounting
- [x] Report generation (charts, metrics, factor attribution)

### Planned
- [ ] **LLM factor formula generation** — LLM proposes novel factor formulas from data patterns
- [ ] **Factor weight optimization** — learn optimal weights from backtest results
- [ ] **Reinforcement learning loop** — backtest feedback → factor improvement
- [ ] **Multi-stock portfolio** — single portfolio across universe (not per-stock)
- [ ] **Stop-loss / take-profit** — risk management in strategy
- [ ] **Factor decay monitoring** — detect when factors lose predictive power
- [ ] **Forward testing** — paper trading with live data
- [ ] **Live trading integration** — Alpaca / IBKR execution
- [ ] **More factor categories** — quality, growth, sentiment, macro, alternative data

## Development

```bash
pip install -e ".[dev]"
pytest                          # 31 tests
python examples/quickstart.py --no-llm  # single stock
python examples/batch_detailed.py       # 100 stocks
```

## License

MIT
