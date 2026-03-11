"""Run Bayesian parameter optimization using Optuna.

Usage:
    python strategies/momentum_quality/optimize.py                    # 200 trials, single-period
    python strategies/momentum_quality/optimize.py --walk-forward     # 200 trials, walk-forward validation
    python strategies/momentum_quality/optimize.py --n-trials 5       # Quick smoke test
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import optuna
from dotenv import load_dotenv

load_dotenv()

import strategies.momentum_quality.factors  # noqa: F401  — register Graham factors

from alphalab.optimizer.runner import run_optimization
from alphalab.optimizer.report import print_optimization_report
from alphalab.optimizer.apply import apply_best_params
from alphalab.data.tickers import SP500

logging.basicConfig(level=logging.WARNING, format="%(message)s")

STRATEGY_DIR = Path(__file__).resolve().parent
CONFIG_PATH = str(STRATEGY_DIR / "configs" / "default.yaml")
OUTPUT_PATH = str(STRATEGY_DIR / "configs" / "optimized.yaml")
STORAGE = f"sqlite:///{STRATEGY_DIR / 'optuna_study.db'}"


def mq_search_space(trial: optuna.Trial) -> dict:
    """Define Graham-specific hyperparameter search space.

    Returns a dict of config overrides (dot-separated keys).
    """
    # Tier 1: Factor weights (5 params)
    weights = {
        "mq_pe": trial.suggest_float("w_mq_pe", 0.0, 3.0),
        "price_to_book": trial.suggest_float("w_price_to_book", 0.0, 3.0),
        "mq_number": trial.suggest_float("w_mq_number", 0.0, 3.0),
        "current_ratio": trial.suggest_float("w_current_ratio", 0.0, 3.0),
        "dividend_yield": trial.suggest_float("w_dividend_yield", 0.0, 3.0),
    }

    # Tier 2: Signal thresholds (2 params)
    long_threshold = trial.suggest_float("long_threshold", 0.45, 0.75)
    sigmoid_weight = trial.suggest_float("sigmoid_weight", 0.1, 0.9)

    # Tier 3: Risk management (1 param)
    trailing_stop = trial.suggest_float("trailing_stop_pct", 0.10, 0.35)

    return {
        "factors.weights": weights,
        "factors.sigmoid_weight": sigmoid_weight,
        "backtest.long_threshold": long_threshold,
        "backtest.trailing_stop_pct": trailing_stop,
    }


def main():
    parser = argparse.ArgumentParser(description="Optimize strategy parameters")
    parser.add_argument("--n-trials", type=int, default=200, help="Number of trials")
    parser.add_argument("--walk-forward", action="store_true", help="Use walk-forward validation")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel workers")
    parser.add_argument("--study-name", default="mq_v1", help="Optuna study name")
    args = parser.parse_args()

    print("=" * 80)
    print("  BAYESIAN PARAMETER OPTIMIZATION — Momentum+Quality Strategy")
    print(f"  Trials: {args.n_trials}")
    print(f"  Universe: {len(SP500)} stocks (S&P 500)")
    print(f"  Walk-forward: {args.walk_forward}")
    if args.walk_forward:
        print(f"  Train: 2019-01-01 to 2023-12-31")
        print(f"  Test:  2024-01-01 to 2025-12-31")
    print(f"  Storage: {STORAGE}")
    print("=" * 80)
    print()

    study = run_optimization(
        tickers=SP500,
        config_path=CONFIG_PATH,
        suggest_params=mq_search_space,
        n_trials=args.n_trials,
        study_name=args.study_name,
        storage=STORAGE,
        n_jobs=args.n_jobs,
        walk_forward=args.walk_forward,
        train_period=("2019-01-01", "2023-12-31"),
        test_period=("2024-01-01", "2025-12-31"),
    )

    print_optimization_report(study)

    # Save best params
    output = apply_best_params(study, CONFIG_PATH, OUTPUT_PATH)
    print(f"\nTo run backtest with optimized params:")
    print(f"  python strategies/momentum_quality/portfolio_run.py  (uses '{output}')")
    print("=" * 80)


if __name__ == "__main__":
    main()
