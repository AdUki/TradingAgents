"""Run TradingAgents analysis over a list of tickers you hold and print a
summary table of the decisions.

TradingAgents has no concept of "your portfolio" — each ticker is analyzed
independently, with no knowledge of your position size or entry price. This
script just loops ``TradingAgentsGraph.propagate()`` over a ticker list so
you can check your whole eToro (or any) portfolio in one pass instead of
running the CLI once per stock.

Usage:
    # tickers as arguments
    python scripts/portfolio_review.py AAPL TSLA BTC-USD

    # or from a file, one ticker per line (# comments allowed)
    python scripts/portfolio_review.py --file my_portfolio.txt

    # pin the analysis date (defaults to today)
    python scripts/portfolio_review.py AAPL TSLA --date 2026-09-01

Requires the API key for whichever provider is configured (see .env /
TRADINGAGENTS_LLM_PROVIDER) to be set. Analyzing many tickers makes that
many full agent runs, which costs real LLM tokens and takes a few minutes
per ticker — start with a short list.
"""

from __future__ import annotations

import argparse
import datetime
import sys

from cli.utils import detect_asset_type
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph


def load_tickers(args: argparse.Namespace) -> list[str]:
    tickers = list(args.tickers)
    if args.file:
        with open(args.file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    tickers.append(line)
    # de-dupe while preserving order
    seen: set[str] = set()
    ordered = []
    for t in tickers:
        if t.upper() not in seen:
            seen.add(t.upper())
            ordered.append(t)
    return ordered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tickers", nargs="*", help="Tickers to analyze, e.g. AAPL TSLA BTC-USD")
    parser.add_argument("--file", help="Path to a file with one ticker per line")
    parser.add_argument("--date", default=datetime.date.today().isoformat(), help="Analysis date, YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    tickers = load_tickers(args)
    if not tickers:
        parser.error("no tickers given — pass them as arguments or via --file")

    config = DEFAULT_CONFIG.copy()
    ta = TradingAgentsGraph(debug=False, config=config)

    results: list[tuple[str, str]] = []
    for ticker in tickers:
        asset_type = detect_asset_type(ticker).value
        print(f"\n=== {ticker} ({args.date}, {asset_type}) ===", flush=True)
        try:
            _, decision = ta.propagate(ticker, args.date, asset_type=asset_type)
        except Exception as exc:  # noqa: BLE001 - keep going, report every ticker
            print(f"  ERROR: {exc}")
            results.append((ticker, f"ERROR: {exc}"))
            continue
        print(f"  Decision: {decision}")
        results.append((ticker, decision))

    print("\n" + "=" * 40)
    print(f"Portfolio review — {args.date}")
    print("=" * 40)
    width = max(len(t) for t, _ in results)
    for ticker, decision in results:
        print(f"  {ticker:<{width}}  {decision}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
