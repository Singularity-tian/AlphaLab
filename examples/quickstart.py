"""AlphaLab Quickstart: Run 5 factors on AAPL and see backtest results.

Usage:
    1. Copy .env.example to .env and add your API keys
    2. pip install -e .
    3. python examples/quickstart.py

For traditional-factors-only mode (no LLM keys needed):
    python examples/quickstart.py --no-llm
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

# Import factor modules to trigger @register_factor decorators
import alphalab.factors.graham
import alphalab.factors.momentum
import alphalab.factors.traditional
from alphalab.backtest.report import ReportGenerator
from alphalab.backtest.runner import BacktestRunner
from alphalab.config import AlphaLabConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s — %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="AlphaLab Quickstart")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM factors")
    parser.add_argument("--ticker", default="AAPL", help="Stock ticker (default: AAPL)")
    args = parser.parse_args()

    config = AlphaLabConfig.default()

    if args.no_llm:
        config.factors.factors = [
            "graham_pe", "price_to_book", "graham_number",
            "current_ratio", "dividend_yield",
        ]
    else:
        import alphalab.factors.llm_judgment  # noqa: F401

    runner = BacktestRunner(config)
    results = runner.run(args.ticker)

    reporter = ReportGenerator()
    reporter.generate(results, args.ticker)

    # Open interactive chart
    results["bt"].plot(open_browser=True)


if __name__ == "__main__":
    main()
