"""Run Graham value strategy backtest.

Usage:
    source .venv/bin/activate
    python strategies/graham_value/run.py
    python strategies/graham_value/run.py --ticker MSFT
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv

load_dotenv()

# Import Graham factors to trigger @register_factor decorators
import strategies.graham_value.factors  # noqa: F401

from alphalab.backtest.report import ReportGenerator
from alphalab.backtest.runner import BacktestRunner
from alphalab.config import AlphaLabConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s — %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="Graham Value Strategy Backtest")
    parser.add_argument("--ticker", default="AAPL", help="Stock ticker (default: AAPL)")
    args = parser.parse_args()

    config = AlphaLabConfig.from_yaml(Path(__file__).parent / "configs" / "default.yaml")

    runner = BacktestRunner(config)
    results = runner.run(args.ticker)

    reporter = ReportGenerator(output_dir=Path(__file__).parent / "reports")
    reporter.generate(results, args.ticker)

    results["bt"].plot(open_browser=True)


if __name__ == "__main__":
    main()
