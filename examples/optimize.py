"""Run Bayesian parameter optimization using Optuna.

Usage:
    python examples/optimize.py                    # 200 trials, single-period
    python examples/optimize.py --walk-forward     # 200 trials, walk-forward validation
    python examples/optimize.py --n-trials 5       # Quick smoke test
"""

from __future__ import annotations

import argparse
import logging

from dotenv import load_dotenv

load_dotenv()

from alphalab.optimizer.runner import run_optimization
from alphalab.optimizer.report import print_optimization_report
from alphalab.optimizer.apply import apply_best_params

# Top 30 stocks for faster optimization (~30s per trial vs ~77s with 100)
OPTIMIZE_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "JPM",
    "V", "MA", "PG", "JNJ", "HD", "WMT", "BAC", "CVX", "MRK", "KO",
    "PEP", "CSCO", "ABT", "WFC", "PM", "IBM", "GE", "CAT", "VZ", "T",
    "GS", "MMM",
]

logging.basicConfig(level=logging.WARNING, format="%(message)s")


def main():
    parser = argparse.ArgumentParser(description="Optimize strategy parameters")
    parser.add_argument("--n-trials", type=int, default=200, help="Number of trials")
    parser.add_argument("--walk-forward", action="store_true", help="Use walk-forward validation")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel workers")
    parser.add_argument("--study-name", default="alphalab_v1", help="Optuna study name")
    args = parser.parse_args()

    print("=" * 80)
    print("  BAYESIAN PARAMETER OPTIMIZATION")
    print(f"  Trials: {args.n_trials}")
    print(f"  Universe: {len(OPTIMIZE_UNIVERSE)} stocks")
    print(f"  Walk-forward: {args.walk_forward}")
    if args.walk_forward:
        print(f"  Train: 2019-01-01 to 2022-12-31")
        print(f"  Test:  2023-01-01 to 2024-12-31")
    print(f"  Storage: sqlite:///data/optuna_study.db")
    print("=" * 80)
    print()

    study = run_optimization(
        tickers=OPTIMIZE_UNIVERSE,
        n_trials=args.n_trials,
        study_name=args.study_name,
        n_jobs=args.n_jobs,
        walk_forward=args.walk_forward,
    )

    print_optimization_report(study)

    # Save best params
    output = apply_best_params(study)
    print(f"\nTo run backtest with optimized params:")
    print(f"  Edit examples/portfolio_backtest.py to use '{output}'")
    print("=" * 80)


if __name__ == "__main__":
    main()
