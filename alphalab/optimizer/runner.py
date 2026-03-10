"""Optuna study orchestration."""

from __future__ import annotations

import logging

import optuna

from alphalab.optimizer.objective import objective

logger = logging.getLogger(__name__)


def run_optimization(
    tickers: list[str],
    n_trials: int = 200,
    study_name: str = "alphalab_v1",
    storage: str = "sqlite:///data/optuna_study.db",
    n_jobs: int = 1,
    config_path: str = "configs/optimized.yaml",
    walk_forward: bool = False,
    train_period: tuple[str, str] = ("2019-01-01", "2022-12-31"),
    test_period: tuple[str, str] = ("2023-01-01", "2024-12-31"),
) -> optuna.Study:
    """Run Bayesian optimization over strategy parameters.

    Args:
        tickers: Stock universe for backtesting.
        n_trials: Number of optimization trials.
        study_name: Optuna study name (for resuming).
        storage: SQLite URL for persistent study storage.
        n_jobs: Parallel workers (1 = sequential).
        config_path: Base config YAML to override.
        walk_forward: If True, use train/test split to prevent overfitting.
        train_period: (start, end) dates for training period.
        test_period: (start, end) dates for test period.

    Returns:
        Completed Optuna study.
    """
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    train = train_period if walk_forward else None
    test = test_period if walk_forward else None

    study.optimize(
        lambda trial: objective(
            trial,
            tickers,
            config_path=config_path,
            train_period=train,
            test_period=test,
        ),
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )

    return study
