"""Backtest report generation with metrics, charts, and factor attribution."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates backtest reports with metrics and charts."""

    def __init__(self, output_dir: Path = Path("reports"), method: str = "graham"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = output_dir / f"{method}_{ts}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        results: dict[str, Any],
        ticker: str,
        save_html: bool = True,
    ) -> dict[str, Any]:
        stats = results["stats"]
        equity_curve = results["equity_curve"]
        factor_scores = results["factor_scores"]
        combined_signal = results["combined_signal"]
        trades = results["trades"]

        metrics = self._format_metrics(stats)
        factor_corr = self._factor_correlations(factor_scores)
        chart_paths = self._generate_charts(
            ticker, equity_curve, factor_scores, combined_signal
        )

        if save_html:
            html_path = self.output_dir / f"{ticker}_backtest.html"
            results["bt"].plot(filename=str(html_path), open_browser=False)
            chart_paths["interactive_html"] = str(html_path)

        report = {
            "ticker": ticker,
            "metrics": metrics,
            "factor_correlations": factor_corr,
            "chart_paths": chart_paths,
            "trade_count": len(trades),
        }

        self._print_summary(report)
        return report

    def _format_metrics(self, stats: pd.Series) -> dict[str, str]:
        keys = [
            "Return [%]",
            "Return (Ann.) [%]",
            "Sharpe Ratio",
            "Sortino Ratio",
            "Calmar Ratio",
            "Max. Drawdown [%]",
            "Win Rate [%]",
            "# Trades",
            "Equity Final [$]",
            "Best Trade [%]",
            "Worst Trade [%]",
            "Avg. Trade [%]",
        ]
        metrics = {}
        for key in keys:
            if key in stats.index:
                val = stats[key]
                metrics[key] = f"{val:.2f}" if isinstance(val, float) else str(val)
        return metrics

    def _factor_correlations(self, factor_scores: pd.DataFrame) -> pd.DataFrame:
        return factor_scores.corr().round(3)

    def _generate_charts(
        self,
        ticker: str,
        equity_curve: pd.DataFrame,
        factor_scores: pd.DataFrame,
        combined_signal: pd.Series,
    ) -> dict[str, str]:
        paths: dict[str, str] = {}

        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

        # Equity curve
        ax1 = axes[0]
        ax1.plot(equity_curve.index, equity_curve["Equity"], color="steelblue")
        ax1.set_title(f"{ticker} — Equity Curve")
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.grid(True, alpha=0.3)

        # Factor scores
        ax2 = axes[1]
        for col in factor_scores.columns:
            ax2.plot(factor_scores.index, factor_scores[col], label=col, alpha=0.7)
        ax2.set_title("Individual Factor Scores")
        ax2.set_ylabel("Score [0, 1]")
        ax2.legend(loc="upper left", fontsize=8)
        ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        ax2.grid(True, alpha=0.3)

        # Combined signal
        ax3 = axes[2]
        ax3.fill_between(
            combined_signal.index,
            combined_signal,
            where=combined_signal > 0,
            color="green",
            alpha=0.4,
            label="Long",
        )
        ax3.fill_between(
            combined_signal.index,
            combined_signal,
            where=combined_signal < 0,
            color="red",
            alpha=0.4,
            label="Short",
        )
        ax3.set_title("Combined Trading Signal")
        ax3.set_ylabel("Signal Strength")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        chart_path = self.output_dir / f"{ticker}_analysis.png"
        fig.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths["analysis_chart"] = str(chart_path)

        return paths

    def _print_summary(self, report: dict[str, Any]) -> None:
        print()
        print("=" * 60)
        print(f"  BACKTEST REPORT: {report['ticker']}")
        print("=" * 60)
        for key, value in report["metrics"].items():
            print(f"  {key:<30} {value:>15}")
        print("-" * 60)
        print(f"  Total Trades: {report['trade_count']}")
        if report.get("chart_paths"):
            print()
            print("  Charts saved to:")
            for name, path in report["chart_paths"].items():
                print(f"    {name}: {path}")
        print("=" * 60)
        print()
