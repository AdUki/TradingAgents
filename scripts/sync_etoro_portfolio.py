#!/usr/bin/env python3
"""Sync portfolio.txt with the open positions in your eToro account.

Reads positions through the eToro Public API (via the etoropy SDK) and writes
their Yahoo Finance tickers to portfolio.txt, which scripts/portfolio_review.py
analyzes. Only stocks, ETFs and crypto are kept: eToro names currencies,
commodities and indices like unrelated Yahoo stocks (GOLD, OIL, SPX500).
Copy-trading (mirror) positions are not included.

Also saves an account snapshot (cash, pending orders, every position with its
stop loss / take profit) to ~/.tradingagents/etoro/account.json for
scripts/etoro_account.py.

Setup, in .env:
    ETORO_API_KEY=...   # Public API Key
    ETORO_USER_KEY=...  # User Key, shown once when the key is created
    ETORO_MODE=real     # must match the key's environment (Real or Demo)

Create the key in eToro under Settings > Trading > API Key Management with
permission "Read"; this script never trades.

Usage:
    scripts/sync_etoro_portfolio.py            # rewrite portfolio.txt and the snapshot
    scripts/sync_etoro_portfolio.py --dry-run  # only print the tickers

Requires the optional dependency: pip install ".[etoro]"
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
DEFAULT_PORTFOLIO = REPO_ROOT / "portfolio.txt"
ACCOUNT_SNAPSHOT = Path.home() / ".tradingagents" / "etoro" / "account.json"

# Run directly, the shebang picks the system python, which lacks the project's
# dependencies; switch to the repo's venv.
if VENV_PYTHON.exists() and Path(sys.prefix).resolve() != VENV_PYTHON.parent.parent.resolve():
    os.execv(VENV_PYTHON, [str(VENV_PYTHON), *sys.argv])

from dotenv import load_dotenv  # noqa: E402
from etoropy import EToroTrading  # noqa: E402

# eToro exchange suffixes Yahoo spells differently; "" drops the suffix
# (.US, .RTH regular hours and .EXT extended hours are ordinary US listings).
_SUFFIX_TO_YAHOO = {".ZU": ".SW", ".NV": ".AS", ".ASX": ".AX", ".US": "", ".RTH": "", ".EXT": ""}
_SHARED_SUFFIXES = {
    ".L", ".DE", ".PA", ".HK", ".MI", ".MC", ".ST", ".OL", ".CO", ".HE",
    ".VI", ".WA", ".BR", ".BD",
}
# Quote currencies on eToro crypto pairs such as BTCEUR.
_CRYPTO_QUOTES = ("USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF", "CNH")


def to_yahoo_symbol(symbol_full: str, type_description: str) -> str | None:
    """Yahoo Finance ticker for an eToro instrument, or None if it can't be analyzed."""
    kind = type_description.lower()
    if "crypto" in kind:
        coin = symbol_full.upper()
        for quote in _CRYPTO_QUOTES:
            if coin.endswith(quote) and len(coin) > len(quote):
                coin = coin[: -len(quote)]
                break
        return f"{coin}-USD"
    if "stock" not in kind and "etf" not in kind:
        return None

    base, dot, tail = symbol_full.rpartition(".")
    if not dot:
        return symbol_full
    suffix = f".{tail}"
    if suffix in _SUFFIX_TO_YAHOO:
        return base + _SUFFIX_TO_YAHOO[suffix]
    if suffix in _SHARED_SUFFIXES:
        return symbol_full
    if len(tail) == 1 and tail.isupper():  # US share class: BRK.B -> BRK-B
        return f"{base}-{tail}"
    return None


def build_tickers(
    positions: Iterable,
    infos: Mapping[int, object],
    type_names: Mapping[int, str],
) -> tuple[list[str], list[str]]:
    """Sorted unique Yahoo tickers for ``positions``, plus a note per skipped instrument."""
    tickers: set[str] = set()
    skipped: dict[int, str] = {}
    for position in positions:
        instrument_id = position.instrument_id
        info = infos.get(instrument_id)
        if info is None:
            skipped[instrument_id] = f"instrument {instrument_id}: no metadata from eToro"
            continue
        type_name = type_names.get(info.instrument_type_id, f"type {info.instrument_type_id}")
        ticker = to_yahoo_symbol(info.symbol_full, type_name)
        if ticker is None:
            skipped[instrument_id] = f"{info.symbol_full} ({type_name}): not analyzable"
            continue
        tickers.add(ticker)
    return sorted(tickers), list(skipped.values())


def pending_orders_amount(portfolio) -> float:
    """Cash reserved by pending orders (eToro's available-cash formula)."""
    manual_opens = sum(
        order.amount for order in portfolio.orders_for_open if not (order.model_extra or {}).get("mirrorID")
    )
    return manual_opens + sum(order.amount for order in portfolio.orders)


def mirrors_invested(portfolio) -> float:
    """Money in copy-trading, per eToro's total-invested formula."""
    return sum(
        sum(p.amount for p in mirror.positions) + mirror.available_amount - mirror.closed_positions_net_profit
        for mirror in portfolio.mirrors
    )


def build_snapshot(portfolio, infos: Mapping[int, object], type_names: Mapping[int, str], mode: str) -> dict:
    positions = []
    for p in portfolio.positions:
        info = infos.get(p.instrument_id)
        type_name = type_names.get(info.instrument_type_id, "") if info else ""
        positions.append({
            "position_id": p.position_id,
            "instrument_id": p.instrument_id,
            "symbol_full": info.symbol_full if info else str(p.instrument_id),
            "ticker": to_yahoo_symbol(info.symbol_full, type_name) if info else None,
            "is_buy": p.is_buy,
            "leverage": p.leverage,
            "amount": p.amount,
            "units": p.units,
            "open_rate": p.open_rate,
            "open_date": p.open_date_time,
            "stop_loss_rate": p.stop_loss_rate,
            "take_profit_rate": p.take_profit_rate,
            "is_trailing_stop": p.is_tsl_enabled,
        })
    return {
        "synced_at": datetime.datetime.now().isoformat(timespec="minutes"),
        "mode": mode,
        "credit": portfolio.credit,
        "pending_orders_amount": pending_orders_amount(portfolio),
        "mirrors_invested": mirrors_invested(portfolio),
        "positions": positions,
    }


async def fetch_portfolio() -> tuple[object, dict[int, object], dict[int, str]]:
    async with EToroTrading() as etoro:
        portfolio = (await etoro.get_portfolio()).client_portfolio
        ids = sorted({p.instrument_id for p in portfolio.positions})
        infos = {i.instrument_id: i for i in await etoro.resolver.get_instrument_info_batch(ids)}
        types = await etoro.rest.market_data.get_instrument_types()
    type_names = {t.instrument_type_id: t.instrument_type_description for t in types.instrument_types}
    return portfolio, infos, type_names


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def portfolio_text(tickers: list[str]) -> str:
    header = (
        f"# Synced from eToro by scripts/sync_etoro_portfolio.py at "
        f"{datetime.datetime.now():%Y-%m-%d %H:%M}.\n"
        "# Manual edits are overwritten on the next sync.\n"
    )
    return header + "".join(f"{t}\n" for t in tickers)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Print tickers without writing any files")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    missing = [v for v in ("ETORO_API_KEY", "ETORO_USER_KEY") if not os.environ.get(v)]
    if missing:
        print(f"Missing {', '.join(missing)} in {REPO_ROOT / '.env'}; portfolio.txt left unchanged.")
        return 1

    try:
        portfolio, infos, type_names = asyncio.run(fetch_portfolio())
    except Exception as exc:  # noqa: BLE001 - any API failure keeps the last synced list
        print(f"eToro sync failed ({type(exc).__name__}: {exc}); portfolio.txt left unchanged.")
        return 1

    tickers, skipped = build_tickers(portfolio.positions, infos, type_names)
    print(f"{len(portfolio.positions)} open positions -> {len(tickers)} tickers: {', '.join(tickers) or '(none)'}")
    for note in skipped:
        print(f"  skipped {note}")
    copied = sum(len(m.positions) for m in portfolio.mirrors)
    if copied:
        print(f"  not included: {copied} copy-trading positions")

    if args.dry_run:
        return 0
    write_atomic(DEFAULT_PORTFOLIO, portfolio_text(tickers))
    snapshot = build_snapshot(portfolio, infos, type_names, os.environ.get("ETORO_MODE", "demo"))
    write_atomic(ACCOUNT_SNAPSHOT, json.dumps(snapshot, indent=2))
    print(f"Wrote {DEFAULT_PORTFOLIO} and {ACCOUNT_SNAPSHOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
