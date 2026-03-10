"""Apply best optimization params to a YAML config file."""

from __future__ import annotations

from pathlib import Path

import optuna
import yaml


def apply_best_params(
    study: optuna.Study,
    base_config_path: str,
    output_path: str,
) -> str:
    """Write best trial parameters to a new YAML config file.

    Returns the output path.
    """
    with open(base_config_path) as f:
        config = yaml.safe_load(f) or {}

    best = study.best_trial.params

    # Factor weights — any param starting with "w_" maps to factors.weights
    weights = {}
    for k, v in best.items():
        if k.startswith("w_"):
            factor_name = k[2:]  # strip "w_" prefix
            weights[factor_name] = round(v, 4)

    if weights:
        if "factors" not in config:
            config["factors"] = {}
        config["factors"]["weights"] = weights

    if "sigmoid_weight" in best:
        if "factors" not in config:
            config["factors"] = {}
        config["factors"]["sigmoid_weight"] = round(best["sigmoid_weight"], 4)

    # Backtest params
    if "backtest" not in config:
        config["backtest"] = {}

    param_map = {
        "long_threshold": "long_threshold",
        "trailing_stop_pct": "trailing_stop_pct",
        "max_position_weight": "max_position_weight",
        "max_positions": "max_positions",
        "holding_period": "holding_period_days",
        "scale_in_pct": "scale_in_pct",
    }
    for trial_key, yaml_key in param_map.items():
        if trial_key in best:
            val = best[trial_key]
            config["backtest"][yaml_key] = round(val, 4) if isinstance(val, float) else val

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Best parameters written to {output_path}")
    return output_path
