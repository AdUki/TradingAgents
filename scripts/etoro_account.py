#!/usr/bin/env python3
"""Value your eToro positions and check their stop loss / take profit.

Reads the account snapshot saved by scripts/sync_etoro_portfolio.py, prices
each position with Yahoo Finance and writes ~/.tradingagents/etoro/report.json:
free cash and invested totals, each position's value, P/L and share of the
account, and recommended stop-loss / take-profit levels.

Stop losses sit a multiple of the 14-day average true range (ATR) away from the
price, tighter as the latest TradingAgents rating worsens (Buy/Overweight 3x,
Hold 2.5x, Underweight 1.5x, Sell 1x). A stop loss is only ever tightened, never
loosened. A take profit is only proposed where none is set, at twice the
stop-loss distance. eToro's near-zero placeholder rates count as "not set".

Usage:
    scripts/etoro_account.py           # report and recommend only
    scripts/etoro_account.py --apply   # also send the changes to eToro

Applying (--apply, or ETORO_AUTO_APPLY_STOPS=true in .env) needs an eToro key
with Write permission and a real (not demo) account.
"""

from __future__ import annotations

import argparse
import datetime
import functools
import json
import os
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"

# Run directly, the shebang picks the system python, which lacks the project's
# dependencies; switch to the repo's venv.
if VENV_PYTHON.exists() and Path(sys.prefix).resolve() != VENV_PYTHON.parent.parent.resolve():
    os.execv(VENV_PYTHON, [str(VENV_PYTHON), *sys.argv])

import httpx  # noqa: E402
import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from tradingagents.agents.utils.memory import TradingMemoryLog  # noqa: E402
from tradingagents.default_config import DEFAULT_CONFIG  # noqa: E402

ETORO_DIR = Path.home() / ".tradingagents" / "etoro"
SNAPSHOT = ETORO_DIR / "account.json"
REPORT = ETORO_DIR / "report.json"
POSITIONS_API = "https://public-api.etoro.com/api/v2/trading/positions"

ATR_DAYS = 14
STOP_ATR_MULTIPLE = {"buy": 3.0, "overweight": 3.0, "hold": 2.5, "underweight": 1.5, "sell": 1.0}
DEFAULT_STOP_ATR_MULTIPLE = 2.5
TAKE_PROFIT_RISK_MULTIPLE = 2.0
# eToro stores "no stop loss" as a rate near zero (0.0001-0.01).
UNSET_RATE_FRACTION = 0.05
MIN_STOP_MOVE = 0.005
# A Yahoo price this many times away from the entry is in other units (e.g. pence).
UNIT_MISMATCH_RATIO = 5.0
CONCENTRATION_WARNING = 0.20
LOW_CASH_WARNING = 0.05


def average_true_range(history: pd.DataFrame, days: int = ATR_DAYS) -> float | None:
    previous_close = history["Close"].shift(1)
    true_range = pd.concat(
        [
            history["High"] - history["Low"],
            (history["High"] - previous_close).abs(),
            (history["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1).dropna()
    if len(true_range) < days:
        return None
    return float(true_range.tail(days).mean())


def effective_rate(rate: float | None, entry: float) -> float | None:
    """The rate, or None when it is eToro's near-zero "not set" placeholder."""
    return rate if rate and entry and rate > entry * UNSET_RATE_FRACTION else None


def recommend_levels(
    *,
    is_buy: bool,
    price: float,
    atr: float,
    rating: str | None,
    stop_loss: float | None,
    take_profit: float | None,
) -> tuple[float | None, float | None]:
    """(new stop loss, new take profit); None keeps the current level.

    ``stop_loss`` / ``take_profit`` are effective rates (None = not set).
    """
    direction = 1 if is_buy else -1
    multiple = STOP_ATR_MULTIPLE.get((rating or "").lower(), DEFAULT_STOP_ATR_MULTIPLE)
    candidate = price - direction * multiple * atr

    new_stop = None
    if candidate > 0 and (
        stop_loss is None or direction * (candidate - stop_loss) > stop_loss * MIN_STOP_MOVE
    ):
        new_stop = candidate

    new_take_profit = None
    stop = new_stop if new_stop is not None else stop_loss
    if take_profit is None and stop is not None:
        new_take_profit = price + direction * TAKE_PROFIT_RISK_MULTIPLE * abs(price - stop)
    return new_stop, new_take_profit


def round_rate(rate: float | None) -> float | None:
    if rate is None:
        return None
    return round(rate, 2 if rate >= 1 else 4)


def position_value(*, is_buy: bool, amount: float, units: float, entry: float, price: float, usd_per_unit: float) -> tuple[float, float]:
    """(current value, P/L) in USD."""
    pnl = (1 if is_buy else -1) * (price - entry) * units * usd_per_unit
    return amount + pnl, pnl


@functools.lru_cache
def usd_per_unit(currency: str) -> float:
    if currency == "USD":
        return 1.0
    base, divisor = ("GBP", 100.0) if currency == "GBp" else (currency, 1.0)
    return float(yf.Ticker(f"{base}USD=X").history(period="5d")["Close"].iloc[-1]) / divisor


def latest_ratings() -> dict[str, tuple[str, str]]:
    latest: dict[str, dict] = {}
    for entry in TradingMemoryLog(DEFAULT_CONFIG).load_entries():
        ticker = entry["ticker"].upper()
        if ticker not in latest or entry["date"] >= latest[ticker]["date"]:
            latest[ticker] = entry
    return {t: (e["rating"], e["date"]) for t, e in latest.items()}


def apply_change(position_id: int, stop_loss: float | None, take_profit: float | None) -> str:
    body: dict = {}
    if stop_loss is not None:
        body.update(stopLossRate=stop_loss, stopLossType="fixed")
    if take_profit is not None:
        body["takeProfitRate"] = take_profit
    response = httpx.patch(
        f"{POSITIONS_API}/{position_id}",
        json=body,
        timeout=30,
        headers={
            "x-api-key": os.environ["ETORO_API_KEY"],
            "x-user-key": os.environ["ETORO_USER_KEY"],
            "x-request-id": str(uuid.uuid4()),
        },
    )
    if response.status_code in (401, 403):
        raise PermissionError("the eToro key lacks Write permission")
    if response.status_code == 202:
        return "applied"
    return f"rejected by eToro ({response.status_code}: {response.text[:200]})"


def build_row(position: dict, ratings: dict[str, tuple[str, str]]) -> dict:
    entry = position["open_rate"]
    ticker = position.get("ticker")
    rating, rating_date = ratings.get((ticker or "").upper(), (None, None))
    row = {
        "ticker": ticker,
        "symbol": position["symbol_full"],
        "position_id": position["position_id"],
        "is_buy": position["is_buy"],
        "leverage": position["leverage"],
        "amount": position["amount"],
        "value": position["amount"],
        "pnl": None,
        "price": None,
        "currency": None,
        "rating": rating,
        "rating_date": rating_date,
        "atr": None,
        "stop_loss": effective_rate(position["stop_loss_rate"], entry),
        "take_profit": effective_rate(position["take_profit_rate"], entry),
        "recommended_stop_loss": None,
        "recommended_take_profit": None,
        "status": "ok",
    }
    if not ticker or not position["amount"]:
        row["status"] = "not checked (no Yahoo ticker)"
        return row

    yahoo = yf.Ticker(ticker)
    history = yahoo.history(period="3mo", auto_adjust=False)
    if history.empty:
        row["status"] = "not checked (no Yahoo prices)"
        return row
    price = float(history["Close"].iloc[-1])
    currency = yahoo.fast_info.get("currency") or "USD"
    row.update(price=round_rate(price), currency=currency)
    if entry and max(price, entry) / min(price, entry) > UNIT_MISMATCH_RATIO:
        row["status"] = "not checked (Yahoo and eToro prices use different units)"
        return row

    row["value"], row["pnl"] = position_value(
        is_buy=position["is_buy"], amount=position["amount"], units=position["units"],
        entry=entry, price=price, usd_per_unit=usd_per_unit(currency),
    )
    atr = average_true_range(history)
    if atr is None:
        row["status"] = "levels not checked (too little price history)"
        return row
    new_stop, new_take_profit = recommend_levels(
        is_buy=position["is_buy"], price=price, atr=atr, rating=rating,
        stop_loss=row["stop_loss"], take_profit=row["take_profit"],
    )
    row.update(
        atr=round_rate(atr),
        recommended_stop_loss=round_rate(new_stop),
        recommended_take_profit=round_rate(new_take_profit),
    )
    if new_stop is not None or new_take_profit is not None:
        row["status"] = "change recommended"
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="Send recommended stop loss / take profit changes to eToro")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    apply = args.apply or os.environ.get("ETORO_AUTO_APPLY_STOPS", "").lower() in ("1", "true", "yes", "on")
    if not SNAPSHOT.exists():
        print(f"No account snapshot at {SNAPSHOT}; run scripts/sync_etoro_portfolio.py first.")
        return 1
    snapshot = json.loads(SNAPSHOT.read_text())
    if apply and snapshot.get("mode") != "real":
        print("Applying changes is only supported for real accounts; reporting only.")
        apply = False

    ratings = latest_ratings()
    rows = [build_row(position, ratings) for position in snapshot["positions"]]

    if apply:
        for row in rows:
            if row["status"] != "change recommended":
                continue
            try:
                row["status"] = apply_change(row["position_id"], row["recommended_stop_loss"], row["recommended_take_profit"])
            except PermissionError as exc:
                for pending in rows:
                    if pending["status"] == "change recommended":
                        pending["status"] = f"change recommended, not applied: {exc}"
                break
            except httpx.HTTPError as exc:
                row["status"] = f"change recommended, not applied: {exc}"

    available_cash = snapshot["credit"] - snapshot["pending_orders_amount"]
    positions_value = sum(row["value"] for row in rows)
    equity = snapshot["credit"] + positions_value + snapshot["mirrors_invested"]
    for row in rows:
        row["allocation"] = row["value"] / equity if equity else None

    warnings = []
    if equity and available_cash / equity < LOW_CASH_WARNING:
        warnings.append(f"Only ${available_cash:,.2f} ({available_cash / equity:.1%}) is free to invest.")
    warnings += [
        f"{row['ticker']} is {row['allocation']:.0%} of the account."
        for row in rows
        if row["allocation"] and row["allocation"] > CONCENTRATION_WARNING
    ]
    unprotected = [row["ticker"] or row["symbol"] for row in rows if row["stop_loss"] is None and row["amount"]]
    if unprotected:
        warnings.append(f"No stop loss set on: {', '.join(unprotected)}.")

    report = {
        "generated_at": datetime.datetime.now().isoformat(timespec="minutes"),
        "synced_at": snapshot["synced_at"],
        "mode": snapshot.get("mode"),
        "applied": apply,
        "available_cash": available_cash,
        "pending_orders": snapshot["pending_orders_amount"],
        "invested": sum(row["amount"] for row in rows) + snapshot["mirrors_invested"],
        "positions_value": positions_value,
        "equity": equity,
        "cash_share": available_cash / equity if equity else None,
        "positions": rows,
        "warnings": warnings,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    tmp = REPORT.with_name(REPORT.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=2))
    tmp.replace(REPORT)

    print(
        f"Free cash ${available_cash:,.2f} | invested ${report['invested']:,.2f} | "
        f"positions worth ${positions_value:,.2f} | equity ${equity:,.2f}"
    )
    for row in rows:
        if row["status"] != "ok":
            print(
                f"  {row['ticker'] or row['symbol']:<9} {row['status']}"
                + (f": SL {row['stop_loss']} -> {row['recommended_stop_loss']}, TP {row['take_profit']} -> {row['recommended_take_profit']}" if row["atr"] else "")
            )
    for warning in warnings:
        print(f"  ! {warning}")
    print(f"Wrote {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
