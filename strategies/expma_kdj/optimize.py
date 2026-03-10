"""Bayesian parameter optimization for EXPMA & KDJ hourly strategy.

Usage:
    python strategies/expma_kdj/optimize.py                 # 100 trials
    python strategies/expma_kdj/optimize.py --n-trials 10   # Quick test
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import optuna
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from alphalab.data.tickers import SP500
from alphalab.optimizer.objective import compute_reward
from alphalab.optimizer.report import print_optimization_report
from strategies.expma_kdj.strategy import run_portfolio

STRATEGY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STRATEGY_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "sp500_hourly" / "cache"
CONFIG_PATH = STRATEGY_DIR / "configs" / "default.yaml"
OUTPUT_PATH = STRATEGY_DIR / "configs" / "optimized.yaml"
STORAGE = f"sqlite:///{STRATEGY_DIR / 'optuna_study.db'}"

logging.basicConfig(level=logging.WARNING, format="%(message)s")


def load_fixed_params() -> dict:
    """Load non-optimized params from default config."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f) or {}
    bt = cfg.get("backtest", {})
    return {
        "initial_cash": bt.get("initial_cash", 100_000.0),
        "commission": bt.get("commission", 0.001),
        "max_positions": bt.get("max_positions", 20),
    }


def suggest_params(trial: optuna.Trial) -> dict:
    """Define the search space. Returns kwargs for run_portfolio()."""
    return {
        # Indicator params
        "expma_fast": trial.suggest_int("expma_fast", 6, 24, step=2),
        "expma_slow": trial.suggest_int("expma_slow", 30, 100, step=5),
        "kdj_period": trial.suggest_int("kdj_period", 5, 21),
        "j_threshold": trial.suggest_float("j_threshold", -20.0, 20.0, step=2.0),
        # Position management
        "position_pct": trial.suggest_float("position_pct", 0.02, 0.15, step=0.01),
        "stop_loss_pct": trial.suggest_float("stop_loss_pct", 0.02, 0.15, step=0.01),
        # Entry refinement
        "initial_buy_frac": trial.suggest_float("initial_buy_frac", 0.3, 1.0, step=0.1),
        "addon_enabled": trial.suggest_categorical("addon_enabled", [True, False]),
    }


def objective(trial: optuna.Trial, fixed: dict) -> float:
    """Run one backtest trial and return the reward score."""
    params = suggest_params(trial)

    # Constraint: fast EMA must be shorter than slow EMA
    if params["expma_fast"] >= params["expma_slow"]:
        return -10.0

    try:
        results = run_portfolio(
            data_dir=DATA_DIR,
            tickers=SP500,
            **fixed,
            **params,
        )
    except Exception as e:
        logging.warning(f"Trial {trial.number} failed: {e}")
        return -10.0

    stats = results["stats"]

    # Store metrics for reporting
    trial.set_user_attr("sharpe", stats.get("Sharpe Ratio", 0.0))
    trial.set_user_attr("sortino", stats.get("Sortino Ratio", 0.0))
    trial.set_user_attr("return", stats.get("Annualized Return [%]", 0.0))
    trial.set_user_attr("max_dd", stats.get("Max Drawdown [%]", 0.0))
    trial.set_user_attr("profit_factor", stats.get("Profit Factor", 0.0))
    trial.set_user_attr("num_trades", stats.get("# Trades", 0))
    trial.set_user_attr("total_commission", stats.get("Total Commission [$]", 0.0))

    return compute_reward(stats)


def apply_best_params(study: optuna.Study) -> str:
    """Write best trial params to optimized.yaml."""
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f) or {}

    best = study.best_trial.params

    if "backtest" not in config:
        config["backtest"] = {}
    if "indicators" not in config:
        config["indicators"] = {}

    backtest_keys = ["position_pct", "stop_loss_pct", "initial_buy_frac", "addon_enabled"]
    indicator_keys = ["expma_fast", "expma_slow", "kdj_period", "j_threshold"]

    for k in backtest_keys:
        if k in best:
            val = best[k]
            config["backtest"][k] = round(val, 4) if isinstance(val, float) else val

    for k in indicator_keys:
        if k in best:
            val = best[k]
            config["indicators"][k] = round(val, 4) if isinstance(val, float) else val

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\nBest parameters written to {OUTPUT_PATH}")
    return str(OUTPUT_PATH)


def main():
    parser = argparse.ArgumentParser(description="Optimize EXPMA & KDJ parameters")
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--study-name", default="expma_kdj_v1")
    args = parser.parse_args()

    fixed = load_fixed_params()

    print("=" * 70)
    print("  BAYESIAN OPTIMIZATION — EXPMA & KDJ Hourly Strategy")
    print(f"  Trials: {args.n_trials}")
    print(f"  Universe: {len(SP500)} stocks")
    print(f"  Fixed: commission={fixed['commission']}, max_positions={fixed['max_positions']}")
    print(f"  Storage: {STORAGE}")
    print("=" * 70)
    print()

    study = optuna.create_study(
        study_name=args.study_name,
        storage=STORAGE,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    start = time.time()
    study.optimize(
        lambda trial: objective(trial, fixed),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True,
    )
    elapsed = time.time() - start

    print_optimization_report(study)
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed / args.n_trials:.1f}s/trial)")

    apply_best_params(study)

    print(f"\nTo run with optimized params:")
    print(f"  Edit run.py CONFIG_PATH to point to configs/optimized.yaml")
    print("=" * 70)


if __name__ == "__main__":
    main()
