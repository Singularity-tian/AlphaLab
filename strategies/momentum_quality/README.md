# Momentum+Quality Strategy

**Concentrated multi-momentum portfolio with quality filters on S&P 500.**

## Performance

| Period | Ann. Return | Sharpe | Sortino | Max DD | Total Return |
|--------|------------|--------|---------|--------|-------------|
| 2019-2024 (6yr) | **50.9%** | 1.41 | 1.90 | -33.5% | +1,078% |
| 2023-2025 (3yr) | **45.1%** | 1.29 | 1.68 | -35.5% | +205% |

## Strategy Overview

**Core idea:** Buy stocks with strong, steady price momentum backed by fundamental quality (earnings growth + ROE). Hold a concentrated 5-stock portfolio, rotate every ~42 days.

### 6 Factors (weighted)

| Factor | Weight | Description |
|--------|--------|-------------|
| `vol_adj_momentum` | 5.0 | Sharpe-like rolling momentum (return/volatility × √252) — rewards steady trends |
| `price_acceleration` | 3.0 | Is momentum speeding up? (recent 3m return - prior 3m return) |
| `earnings_growth` | 3.0 | YoY EPS growth via sigmoid normalization |
| `momentum_6m` | 3.0 | 6-month price return (126 days, skip 10) |
| `momentum_3m` | 2.5 | 3-month price return (63 days, skip 5) |
| `roe` | 2.0 | Return on equity — expanding-window percentile rank |

### Key Parameters

- **Max positions:** 5 (concentrated, high conviction)
- **Holding period:** 42 trading days (~2 months)
- **Entry threshold:** Combined score > 0.52
- **Trailing stop:** 25% from peak
- **Max weight per stock:** 35%
- **Signal-weighted sizing:** stronger signals get larger allocations

## Files

```
momentum_quality/
├── README.md              ← this file
├── __init__.py
├── factors.py             ← factor registration
├── portfolio_run.py       ← full backtest runner
├── optimize.py            ← Optuna-based parameter optimization
├── configs/
│   ├── default.yaml       ← baseline config
│   └── optimized.yaml     ← best config (50.9% ann.)
└── reports/               ← timestamped HTML reports + CSVs
```

## Usage

```bash
cd ~/Desktop/alphalab
source .venv/bin/activate
PYTHONPATH=. python strategies/momentum_quality/portfolio_run.py
```

## Evolution from Graham Value

The original Graham Value strategy (PE, P/B, Graham Number, Current Ratio, Dividend Yield) produced only **7.9% annualized** — barely beating a savings account. The key insight was:

1. **Momentum > Value** in the 2019-2025 regime
2. **Volatility-adjusted momentum** is the single most important factor
3. **Concentration amplifies alpha** — 5 picks >> 20 picks
4. **Quality filters prevent disasters** — ROE + earnings growth filter out junk rallies
5. **Wider stops + shorter holds** — let trends run, but force periodic re-evaluation

## Top Contributors (2019-2024)

| Ticker | P&L | Description |
|--------|-----|-------------|
| PLTR | +$243k | Palantir — caught the AI rally |
| APP | +$217k | AppLovin — mobile ad tech momentum |
| SMCI | +$216k | Super Micro — AI infrastructure |
| NVDA | +$97k | NVIDIA — the AI kingpin |
| MRNA | +$37k | Moderna — COVID momentum |
