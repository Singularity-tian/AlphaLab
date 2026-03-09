"""Optuna objective function: maps trial parameters to backtest reward."""

from __future__ import annotations

import logging

import optuna

from alphalab.config import AlphaLabConfig
from alphalab.backtest.portfolio_runner import PortfolioBacktestRunner

logger = logging.getLogger(__name__)


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def compute_reward(stats: dict) -> float:
    """Multi-metric reward: normalized blend of Sortino, Sharpe, Calmar, Profit Factor.

    Returns ~[0, 1] for viable strategies, -10.0 for hard rejections.
    """
    sharpe = _safe_float(stats.get("Sharpe Ratio", 0.0))
    sortino = _safe_float(stats.get("Sortino Ratio", 0.0))
    ann_return = _safe_float(stats.get("Annualized Return [%]", 0.0))
    max_dd = abs(_safe_float(stats.get("Max Drawdown [%]", 0.0)))
    profit_factor = _safe_float(stats.get("Profit Factor", 0.0))
    num_trades = _safe_float(stats.get("# Trades", 0))

    calmar = ann_return / max_dd if max_dd > 0 else 0.0

    # --- Hard rejections ---
    if max_dd > 40:
        return -10.0
    if num_trades < 10:
        return -10.0
    if profit_factor < 1.0:
        return -10.0

    # --- Normalize each metric to ~[0, 1] ---
    sharpe_n = max(min(sharpe / 2.5, 1.0), 0.0)
    sortino_n = max(min(sortino / 3.5, 1.0), 0.0)
    calmar_n = max(min(calmar / 1.5, 1.0), 0.0)
    pf_n = max(min((profit_factor - 1.0) / 2.0, 1.0), 0.0)

    # --- Weighted composite ---
    score = (
        0.30 * sortino_n
        + 0.25 * sharpe_n
        + 0.25 * calmar_n
        + 0.20 * pf_n
    )

    return score


def objective(
    trial: optuna.Trial,
    tickers: list[str],
    config_path: str = "configs/default.yaml",
    train_period: tuple[str, str] | None = None,
    test_period: tuple[str, str] | None = None,
) -> float:
    """Single backtest evaluation with trial-suggested parameters.

    If train_period and test_period are provided, runs walk-forward validation:
    optimizes on train, validates on test, penalizes train/test gap.
    """
    # --- Sample parameters ---

    # Tier 1: Factor weights
    weights = {
        "graham_pe": trial.suggest_float("w_graham_pe", 0.0, 3.0),
        "price_to_book": trial.suggest_float("w_price_to_book", 0.0, 3.0),
        "graham_number": trial.suggest_float("w_graham_number", 0.0, 3.0),
        "current_ratio": trial.suggest_float("w_current_ratio", 0.0, 3.0),
        "dividend_yield": trial.suggest_float("w_dividend_yield", 0.0, 3.0),
    }

    # Tier 2: Signal thresholds
    long_threshold = trial.suggest_float("long_threshold", 0.45, 0.75)
    sigmoid_weight = trial.suggest_float("sigmoid_weight", 0.1, 0.9)
    holding_period = trial.suggest_int("holding_period", 63, 504, step=21)

    # Tier 3: Risk management
    trailing_stop = trial.suggest_float("trailing_stop_pct", 0.10, 0.35)
    max_pos_weight = trial.suggest_float("max_position_weight", 0.10, 0.40)
    max_positions = trial.suggest_int("max_positions", 5, 25)
    scale_in_pct = trial.suggest_float("scale_in_pct", 0.3, 1.0)

    def _build_config(start: str, end: str) -> AlphaLabConfig:
        config = AlphaLabConfig.from_yaml(config_path)
        config.factors.weights = weights
        config.factors.sigmoid_weight = sigmoid_weight
        config.backtest.long_threshold = long_threshold
        config.backtest.trailing_stop_pct = trailing_stop
        config.backtest.max_position_weight = max_pos_weight
        config.backtest.max_positions = max_positions
        config.backtest.holding_period_days = holding_period
        config.backtest.scale_in_pct = scale_in_pct
        config.backtest.start_date = start
        config.backtest.end_date = end
        return config

    # --- Walk-forward validation ---
    if train_period and test_period:
        # Train
        train_config = _build_config(*train_period)
        train_runner = PortfolioBacktestRunner(train_config)
        train_result = train_runner.run(tickers)
        train_stats = train_result["stats"]
        train_reward = compute_reward(train_stats)

        # Test
        test_config = _build_config(*test_period)
        test_runner = PortfolioBacktestRunner(test_config)
        test_result = test_runner.run(tickers)
        test_stats = test_result["stats"]
        test_reward = compute_reward(test_stats)

        # Store both for analysis
        trial.set_user_attr("train_reward", train_reward)
        trial.set_user_attr("test_reward", test_reward)
        trial.set_user_attr("train_sharpe", train_stats.get("Sharpe Ratio", 0.0))
        trial.set_user_attr("test_sharpe", test_stats.get("Sharpe Ratio", 0.0))

        # Penalize overfitting (large train/test gap)
        gap_penalty = max(0, train_reward - test_reward) * 0.5
        return test_reward - gap_penalty

    # --- Single-period evaluation ---
    config = _build_config(
        AlphaLabConfig.from_yaml(config_path).backtest.start_date,
        AlphaLabConfig.from_yaml(config_path).backtest.end_date,
    )
    runner = PortfolioBacktestRunner(config)
    result = runner.run(tickers)
    stats = result["stats"]

    trial.set_user_attr("sharpe", stats.get("Sharpe Ratio", 0.0))
    trial.set_user_attr("sortino", stats.get("Sortino Ratio", 0.0))
    trial.set_user_attr("return", stats.get("Annualized Return [%]", 0.0))
    trial.set_user_attr("max_dd", stats.get("Max Drawdown [%]", 0.0))
    trial.set_user_attr("calmar", stats.get("Calmar Ratio", 0.0))
    trial.set_user_attr("profit_factor", stats.get("Profit Factor", 0.0))

    return compute_reward(stats)
