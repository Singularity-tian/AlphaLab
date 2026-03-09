"""Optimization results reporting."""

from __future__ import annotations

import optuna


def print_optimization_report(study: optuna.Study) -> None:
    """Print best params, parameter importance, and top trials."""
    print("=" * 80)
    print("  OPTIMIZATION RESULTS")
    print("=" * 80)

    best = study.best_trial
    print(f"\n  Best Trial: #{best.number}")
    print(f"  Best Reward: {best.value:.4f}")

    # User attributes (Sharpe, return, etc.)
    for key, val in best.user_attrs.items():
        print(f"  {key}: {val}")

    print(f"\n  Best Parameters:")
    for k, v in sorted(best.params.items()):
        if isinstance(v, float):
            print(f"    {k:<25} {v:.4f}")
        else:
            print(f"    {k:<25} {v}")

    # Parameter importance
    try:
        importance = optuna.importance.get_param_importances(study)
        print(f"\n  Parameter Importance (fANOVA):")
        for k, v in importance.items():
            bar = "#" * int(v * 40)
            print(f"    {k:<25} {v:.3f}  {bar}")
    except Exception:
        print("\n  (Parameter importance requires >= 10 completed trials)")

    # Top 10 trials
    completed = [t for t in study.trials if t.value is not None]
    top = sorted(completed, key=lambda t: t.value, reverse=True)[:10]

    print(f"\n  Top 10 Trials (of {len(completed)} completed):")
    print(f"  {'#':<6} {'Reward':>8} {'Sharpe':>8} {'Return%':>8} {'MaxDD%':>8}")
    print("  " + "-" * 42)
    for t in top:
        sharpe = t.user_attrs.get("sharpe", t.user_attrs.get("test_sharpe", ""))
        ret = t.user_attrs.get("return", "")
        dd = t.user_attrs.get("max_dd", "")
        sharpe_s = f"{sharpe:.2f}" if isinstance(sharpe, (int, float)) else str(sharpe)
        ret_s = f"{ret:.1f}" if isinstance(ret, (int, float)) else str(ret)
        dd_s = f"{dd:.1f}" if isinstance(dd, (int, float)) else str(dd)
        print(f"  {t.number:<6} {t.value:>8.4f} {sharpe_s:>8} {ret_s:>8} {dd_s:>8}")

    print("=" * 80)
