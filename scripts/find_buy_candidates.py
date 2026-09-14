#!/usr/bin/env python3
"""Pick new stocks worth analyzing as buy candidates.

Takes Yahoo Finance screener results and keeps liquid US stocks that analysts
rate Buy, that you don't hold, that weren't analyzed in the last week, and that
eToro lists. The best few go to candidates.txt, which the nightly review
analyzes after your holdings; its rating decides whether one is suggested.

Usage:
    scripts/find_buy_candidates.py             # write candidates.txt (3 candidates)
    scripts/find_buy_candidates.py --count 5
    scripts/find_buy_candidates.py --dry-run   # only print

Requires ETORO_API_KEY / ETORO_USER_KEY in .env and pip install ".[etoro]".
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"

# Run directly, the shebang picks the system python, which lacks the project's
# dependencies; switch to the repo's venv.
if VENV_PYTHON.exists() and Path(sys.prefix).resolve() != VENV_PYTHON.parent.parent.resolve():
    os.execv(VENV_PYTHON, [str(VENV_PYTHON), *sys.argv])

import yfinance as yf  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from etoropy import EToroTrading  # noqa: E402
from etoropy.errors.exceptions import EToroValidationError  # noqa: E402

from tradingagents.agents.utils.memory import TradingMemoryLog  # noqa: E402
from tradingagents.default_config import DEFAULT_CONFIG  # noqa: E402

PORTFOLIO = REPO_ROOT / "portfolio.txt"
CANDIDATES_TXT = REPO_ROOT / "candidates.txt"
CANDIDATES_JSON = Path.home() / ".tradingagents" / "etoro" / "candidates.json"

SCREENS = ("undervalued_growth_stocks", "undervalued_large_caps", "growth_technology_stocks")
SCREEN_SIZE = 25
# Yahoo's analyst scale runs 1 (strong buy) to 5 (sell); 2.0 and better reads "Buy".
MAX_ANALYST_SCORE = 2.0
MIN_PRICE = 5.0
MIN_MARKET_CAP = 2e9
REANALYZE_AFTER_DAYS = 7


def analyst_score(rating: str | None) -> float | None:
    """1.6 from Yahoo's "1.6 - Buy"."""
    match = re.match(r"\s*(\d+(?:\.\d+)?)", rating or "")
    return float(match.group(1)) if match else None


def screen_candidates(quotes: list[dict], exclude: set[str]) -> list[dict]:
    """Qualifying screener quotes, best analyst score first, one per symbol."""
    picked: dict[str, dict] = {}
    for quote in quotes:
        symbol = (quote.get("symbol") or "").upper()
        score = analyst_score(quote.get("averageAnalystRating"))
        if (
            not symbol
            or symbol in exclude
            or symbol in picked
            or quote.get("quoteType") != "EQUITY"
            or quote.get("currency") != "USD"
            or score is None
            or score > MAX_ANALYST_SCORE
            or (quote.get("regularMarketPrice") or 0) < MIN_PRICE
            or (quote.get("marketCap") or 0) < MIN_MARKET_CAP
        ):
            continue
        picked[symbol] = {
            "symbol": symbol,
            "name": quote.get("shortName") or quote.get("longName") or symbol,
            "analyst_rating": quote.get("averageAnalystRating"),
            "analyst_score": score,
            "price": quote.get("regularMarketPrice"),
            "market_cap": quote.get("marketCap"),
            "screen": quote.get("screen"),
        }
    return sorted(picked.values(), key=lambda c: (c["analyst_score"], -(c["market_cap"] or 0)))


def held_tickers() -> set[str]:
    if not PORTFOLIO.exists():
        return set()
    lines = (line.strip() for line in PORTFOLIO.read_text().splitlines())
    return {line.upper() for line in lines if line and not line.startswith("#")}


def recently_analyzed(days: int) -> set[str]:
    cutoff = (datetime.date.today() - datetime.timedelta(days=days)).isoformat()
    return {e["ticker"].upper() for e in TradingMemoryLog(DEFAULT_CONFIG).load_entries() if e["date"] >= cutoff}


def fetch_quotes() -> list[dict]:
    quotes = []
    for screen in SCREENS:
        try:
            result = yf.screen(screen, count=SCREEN_SIZE)
        except Exception as exc:  # noqa: BLE001 - one broken screen shouldn't drop the others
            print(f"  screen {screen} failed: {exc}")
            continue
        quotes.extend({**q, "screen": screen} for q in result.get("quotes", []))
    return quotes


async def keep_etoro_listed(candidates: list[dict], count: int) -> list[dict]:
    """The first ``count`` candidates eToro lists as stocks under the same symbol."""
    chosen: list[dict] = []
    async with EToroTrading() as etoro:
        etoro.resolver.load_bundled_csv()
        types = await etoro.rest.market_data.get_instrument_types()
        type_names = {t.instrument_type_id: t.instrument_type_description.lower() for t in types.instrument_types}
        for candidate in candidates:
            try:
                info = await etoro.resolver.get_instrument_info(candidate["symbol"])
            except EToroValidationError:
                continue
            # The resolver falls back to fuzzy text search, so require an exact symbol.
            if info.symbol_full.upper() != candidate["symbol"]:
                continue
            if "stock" not in type_names.get(info.instrument_type_id, ""):
                continue
            chosen.append({**candidate, "etoro_instrument_id": info.instrument_id})
            if len(chosen) == count:
                break
    return chosen


def write_outputs(chosen: list[dict]) -> None:
    now = datetime.datetime.now()
    header = (
        f"# Buy candidates picked by scripts/find_buy_candidates.py at {now:%Y-%m-%d %H:%M}.\n"
        "# Rewritten on every run.\n"
    )
    CANDIDATES_TXT.write_text(header + "".join(f"{c['symbol']}\n" for c in chosen))
    CANDIDATES_JSON.parent.mkdir(parents=True, exist_ok=True)
    tmp = CANDIDATES_JSON.with_name(CANDIDATES_JSON.name + ".tmp")
    tmp.write_text(json.dumps({"generated_at": now.isoformat(timespec="minutes"), "candidates": chosen}, indent=2))
    tmp.replace(CANDIDATES_JSON)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=3, help="How many candidates to pick (default: 3)")
    parser.add_argument("--dry-run", action="store_true", help="Print candidates without writing files")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    missing = [v for v in ("ETORO_API_KEY", "ETORO_USER_KEY") if not os.environ.get(v)]
    if missing:
        print(f"Missing {', '.join(missing)} in {REPO_ROOT / '.env'}; candidates left unchanged.")
        return 1

    exclude = held_tickers() | recently_analyzed(REANALYZE_AFTER_DAYS)
    candidates = screen_candidates(fetch_quotes(), exclude)
    try:
        chosen = asyncio.run(keep_etoro_listed(candidates, args.count))
    except Exception as exc:  # noqa: BLE001 - keep the last candidates if eToro is unreachable
        print(f"eToro lookup failed ({type(exc).__name__}: {exc}); candidates left unchanged.")
        return 1

    print(f"{len(candidates)} screener stocks qualify; picked {len(chosen)}:")
    for c in chosen:
        print(f"  {c['symbol']:<6} {c['name']} — analysts {c['analyst_rating']}, ${c['price']:,.2f} ({c['screen']})")
    if args.dry_run:
        return 0
    write_outputs(chosen)
    print(f"Wrote {CANDIDATES_TXT} and {CANDIDATES_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
