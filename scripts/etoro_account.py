#!/usr/bin/env python3
"""Value your eToro positions and check their stop loss / take profit.

Reads the account snapshot saved by scripts/sync_etoro_portfolio.py, prices
each position with Yahoo Finance and writes ~/.tradingagents/etoro/report.json:
free cash and invested totals, each position's value, P/L and share of the
account, and recommended stop-loss / take-profit levels.

Stop losses sit a multiple of the 14-day average true range (ATR) below the
price, tighter as the latest TradingAgents rating worsens (Buy/Overweight 3x,
Hold 2.5x, Underweight 1.5x, Sell 1x), and never closer than 3%. A stop loss is
only ever tightened, never loosened. A take profit is only proposed where none
is set, at twice the stop-loss distance. eToro's near-zero placeholder rates
count as "not set".

Usage:
    scripts/etoro_account.py                 # report and recommend only
    scripts/etoro_account.py --apply         # send changes to eToro after you type "yes"
    scripts/etoro_account.py --apply --yes   # send without asking

Applying changes real money, so --apply:
- re-reads your live positions and eToro prices and plans from those, not
  from the snapshot or Yahoo (a stop you tightened since is never loosened)
- only touches plain 1x buy positions without a trailing stop or copy trading
- skips a position when eToro's and Yahoo's prices differ by more than 10%
- refuses stops closer than 3% or deeper than 50% below the live price
- asks for confirmation unless --yes or ETORO_AUTO_APPLY_STOPS=true (.env)
- logs every request to ~/.tradingagents/etoro/stop_changes.jsonl and confirms
  each change by re-reading the positions
- runs one at a time
Needs an eToro key with Write permission and a real (not demo) account.
"""

from __future__ import annotations

import argparse
import datetime
import fcntl
import functools
import json
import math
import os
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import IO

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"

# Run directly, the shebang picks the system python, which lacks the project's
# dependencies; switch to the repo's venv.
if VENV_PYTHON.exists() and Path(sys.prefix).resolve() != VENV_PYTHON.parent.parent.resolve():
    os.execv(VENV_PYTHON, [str(VENV_PYTHON), *sys.argv])

import etoro_api  # noqa: E402
import httpx  # noqa: E402
import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from tradingagents.agents.utils.memory import TradingMemoryLog  # noqa: E402
from tradingagents.default_config import DEFAULT_CONFIG  # noqa: E402

ETORO_DIR = Path.home() / ".tradingagents" / "etoro"
SNAPSHOT = ETORO_DIR / "account.json"
REPORT = ETORO_DIR / "report.json"
AUDIT_LOG = ETORO_DIR / "stop_changes.jsonl"
APPLY_LOCK = ETORO_DIR / "apply.lock"
POSITIONS_API = "https://public-api.etoro.com/api/v2/trading/positions"

ATR_DAYS = 14
STOP_ATR_MULTIPLE = {"buy": 3.0, "overweight": 3.0, "hold": 2.5, "underweight": 1.5, "sell": 1.0}
DEFAULT_STOP_ATR_MULTIPLE = 2.5
TAKE_PROFIT_RISK_MULTIPLE = 2.0
# A stop closer than this to the price gets hit by ordinary daily moves.
MIN_STOP_DISTANCE = 0.03
# A computed stop deeper than this means bad data, not a real level.
MAX_STOP_DISTANCE = 0.50
MAX_TAKE_PROFIT_MULTIPLE = 3.0
# eToro's and Yahoo's prices further apart than this are in different units or stale.
PRICE_MATCH_TOLERANCE = 0.10
# eToro stores "no stop loss" as a rate near zero (0.0001-0.01).
UNSET_RATE_FRACTION = 0.05
MIN_STOP_MOVE = 0.005
# A Yahoo price this many times away from the entry is in other units (e.g. pence).
UNIT_MISMATCH_RATIO = 5.0
CONCENTRATION_WARNING = 0.20
LOW_CASH_WARNING = 0.05
CONFIRM_ATTEMPTS = 4
CONFIRM_DELAY_SECONDS = 5


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
    distance = max(multiple * atr, price * MIN_STOP_DISTANCE)
    candidate = price - direction * distance

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


def rate_decimals(rate: float) -> int:
    return 2 if rate >= 1 else 4


def round_rate(rate: float | None) -> float | None:
    return None if rate is None else round(rate, rate_decimals(rate))


def floor_rate(rate: float) -> float:
    """Round a buy's stop loss down, so rounding never moves it toward the price."""
    factor = 10 ** rate_decimals(rate)
    return math.floor(rate * factor) / factor


def same_rate(actual: float | None, wanted: float) -> bool:
    return actual is not None and abs(actual - wanted) <= max(0.01, abs(wanted) * 0.001)


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


def build_row(position: dict, ratings: dict[str, tuple[str, str]]) -> dict:
    entry = position["open_rate"]
    ticker = position.get("ticker")
    rating, rating_date = ratings.get((ticker or "").upper(), (None, None))
    stop_loss = effective_rate(position["stop_loss_rate"], entry)
    take_profit = effective_rate(position["take_profit_rate"], entry)
    row = {
        "ticker": ticker,
        "symbol": position["symbol_full"],
        "position_id": position["position_id"],
        "instrument_id": position["instrument_id"],
        "is_buy": position["is_buy"],
        "leverage": position["leverage"],
        "open_rate": entry,
        "amount": position["amount"],
        "value": position["amount"],
        "pnl": None,
        "price": None,
        "currency": None,
        "rating": rating,
        "rating_date": rating_date,
        "atr": None,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "previous_stop_loss": stop_loss,
        "previous_take_profit": take_profit,
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
        stop_loss=stop_loss, take_profit=take_profit,
    )
    row.update(
        atr=atr,
        recommended_stop_loss=round_rate(new_stop),
        recommended_take_profit=round_rate(new_take_profit),
    )
    if new_stop is not None or new_take_profit is not None:
        row["status"] = "change recommended"
    return row


# --- applying changes -----------------------------------------------------------

def plan_change(row: dict, live, price: float | None) -> tuple[float | None, float | None, str | None]:
    """What to send for one position, from its live eToro state.

    ``live`` is the position as eToro reports it now (None if closed) and
    ``price`` its live bid, the price a buy closes at. Returns
    (stop loss, take profit, None), None meaning "keep", or (None, None, reason)
    when the position must be left alone.
    """
    if live is None:
        return None, None, "position no longer open"
    if live.mirror_id:
        return None, None, "copy-trading position"
    if not live.is_buy:
        return None, None, "short position (only buys are changed)"
    if live.leverage != 1:
        return None, None, f"{live.leverage}x leveraged position (only 1x positions are changed)"
    if live.is_tsl_enabled:
        return None, None, "trailing stop loss is on"
    if row.get("price") is None or row.get("atr") is None:
        return None, None, "no Yahoo price history to size the stop"
    if not price or price <= 0:
        return None, None, "no live eToro price"
    scale = price / row["price"]
    if abs(scale - 1) > PRICE_MATCH_TOLERANCE:
        return None, None, f"eToro price {price:g} and Yahoo price {row['price']:g} differ by {scale - 1:+.0%}"

    stop_loss = effective_rate(live.stop_loss_rate, live.open_rate)
    take_profit = effective_rate(live.take_profit_rate, live.open_rate)
    new_stop, new_take_profit = recommend_levels(
        is_buy=True, price=price, atr=row["atr"] * scale, rating=row.get("rating"),
        stop_loss=stop_loss, take_profit=take_profit,
    )
    if new_stop is not None:
        new_stop = floor_rate(new_stop)
        if not price * (1 - MAX_STOP_DISTANCE) <= new_stop <= price * (1 - MIN_STOP_DISTANCE):
            return None, None, f"computed stop loss {new_stop:g} failed the safety check at price {price:g}"
        if stop_loss is not None and new_stop <= stop_loss:
            new_stop = None
    if new_take_profit is not None:
        new_take_profit = round_rate(new_take_profit)
        if not price * (1 + MIN_STOP_DISTANCE) <= new_take_profit <= price * MAX_TAKE_PROFIT_MULTIPLE:
            new_take_profit = None
    if new_stop is None and new_take_profit is None:
        return None, None, "no change needed at the live price"
    return new_stop, new_take_profit, None


def parse_position(raw: dict) -> SimpleNamespace:
    """The fields plan_change needs from a raw eToro position.

    A missing safety flag takes the value that makes plan_change skip the
    position (unknown leverage, trailing stop on, copy-trading).
    """
    return SimpleNamespace(
        position_id=raw["positionID"],
        instrument_id=raw["instrumentID"],
        is_buy=raw["isBuy"],
        leverage=raw.get("leverage"),
        open_rate=raw["openRate"],
        stop_loss_rate=raw.get("stopLossRate"),
        take_profit_rate=raw.get("takeProfitRate"),
        is_tsl_enabled=raw.get("isTslEnabled", True),
        mirror_id=raw.get("mirrorID", -1),
    )


def fetch_live() -> tuple[dict[int, SimpleNamespace], dict[int, float]]:
    """Open positions by position ID, and the live bid by instrument ID."""
    with make_client() as client:
        raw_positions = etoro_api.get_portfolio(client).get("positions", [])
        positions = {p.position_id: p for p in map(parse_position, raw_positions)}
        bids = {}
        for instrument_id in sorted({p.instrument_id for p in positions.values()}):
            bid = etoro_api.get_bid(client, instrument_id)
            if bid is not None:
                bids[instrument_id] = bid
    return positions, bids


def live_levels() -> dict[int, tuple[float | None, float | None]]:
    """Current (stop loss, take profit) rate of every open position."""
    with make_client() as client:
        raw_positions = etoro_api.get_portfolio(client).get("positions", [])
    return {p["positionID"]: (p.get("stopLossRate"), p.get("takeProfitRate")) for p in raw_positions}


def make_client() -> httpx.Client:
    return httpx.Client(
        timeout=30,
        headers={"x-api-key": os.environ["ETORO_API_KEY"], "x-user-key": os.environ["ETORO_USER_KEY"]},
    )


def send_change(client: httpx.Client, row: dict) -> tuple[str, str | None, dict]:
    """PATCH one position: (new status, reason to stop sending the rest or None, audit details).

    A request with no response may still have reached eToro, so it counts as
    "sent" and the confirmation step decides.
    """
    body: dict = {}
    if row["recommended_stop_loss"] is not None:
        body.update(stopLossRate=row["recommended_stop_loss"], stopLossType="fixed")
    if row["recommended_take_profit"] is not None:
        body["takeProfitRate"] = row["recommended_take_profit"]
    details: dict = {"request_id": str(uuid.uuid4()), "body": body}
    try:
        response = client.patch(
            f"{POSITIONS_API}/{row['position_id']}", json=body, headers={"x-request-id": details["request_id"]}
        )
    except httpx.TransportError as exc:
        details["error"] = f"{type(exc).__name__}: {exc}"
        return "sent", None, details
    details.update(http_status=response.status_code, response=response.text[:500])
    code = response.status_code
    if code == 202:
        return "sent", None, details
    if code in (401, 403):
        reason = "the eToro key lacks Write permission"
        return f"not applied: {reason}", reason, details
    if code == 429:
        reason = "eToro rate limit reached, try again later"
        return f"not applied: {reason}", reason, details
    if code in (404, 409):
        return "not applied: position closed", None, details
    return f"rejected by eToro ({code}: {response.text[:200]})", None, details


def reconcile(rows: list[dict], live: dict[int, tuple[float, float]]) -> list[dict]:
    """Confirm sent edits against the levels eToro now reports.

    Rows whose levels match become "set on eToro" (and take the live levels);
    returns the sent rows eToro doesn't show yet.
    """
    waiting = []
    for row in rows:
        if row["status"] != "sent":
            continue
        if row["position_id"] not in live:
            row["status"] = "position no longer open"
            continue
        stop_loss, take_profit = (effective_rate(rate, row["open_rate"]) for rate in live[row["position_id"]])
        wanted_stop, wanted_take_profit = row["recommended_stop_loss"], row["recommended_take_profit"]
        if (wanted_stop is None or same_rate(stop_loss, wanted_stop)) and (
            wanted_take_profit is None or same_rate(take_profit, wanted_take_profit)
        ):
            row.update(stop_loss=stop_loss, take_profit=take_profit, status="set on eToro")
        else:
            waiting.append(row)
    return waiting


def audit(event: str, row: dict, **details) -> None:
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "event": event,
        "position_id": row["position_id"],
        "ticker": row["ticker"] or row["symbol"],
        "stop_loss": [row["previous_stop_loss"], row["recommended_stop_loss"]],
        "take_profit": [row["previous_take_profit"], row["recommended_take_profit"]],
        "status": row["status"],
        **details,
    }
    with AUDIT_LOG.open("a") as log:
        log.write(json.dumps(entry) + "\n")


def acquire_lock(path: Path) -> IO[str] | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a")  # noqa: SIM115 - must stay open to hold the lock
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def stdin_is_terminal() -> bool:
    return sys.stdin.isatty()


def row_name(row: dict) -> str:
    return row["ticker"] or row["symbol"]


def describe_change(row: dict) -> str:
    changes = []
    if row["recommended_stop_loss"] is not None:
        changes.append(f"stop loss {row['previous_stop_loss'] or 'none'} -> {row['recommended_stop_loss']}")
    if row["recommended_take_profit"] is not None:
        changes.append(f"take profit {row['previous_take_profit'] or 'none'} -> {row['recommended_take_profit']}")
    return ", ".join(changes)


def mark(rows: list[dict], status: str) -> None:
    for row in rows:
        row["status"] = status


def apply_changes(rows: list[dict], snapshot: dict, *, assume_yes: bool) -> None:
    """Plan from live eToro data, confirm, send, and verify stop loss / take profit edits."""
    lock = acquire_lock(APPLY_LOCK)
    if lock is None:
        print("Another --apply run is in progress; nothing changed.")
        mark([r for r in rows if r["status"] == "change recommended"], "not applied: another --apply run in progress")
        return
    try:
        _apply_locked(rows, snapshot, assume_yes=assume_yes)
    finally:
        lock.close()


def _apply_locked(rows: list[dict], snapshot: dict, *, assume_yes: bool) -> None:
    try:
        positions, bids = fetch_live()
    except Exception as exc:  # noqa: BLE001 - never trade without live data
        print(f"Could not read live positions and prices from eToro ({type(exc).__name__}: {exc}); nothing changed.")
        mark([r for r in rows if r["status"] == "change recommended"], "not applied: live eToro data unavailable")
        return

    planned = []
    for row in rows:
        live = positions.get(row["position_id"])
        if live is not None:
            # The snapshot may be hours old: report and plan from what eToro has now.
            stop_loss = effective_rate(live.stop_loss_rate, live.open_rate)
            take_profit = effective_rate(live.take_profit_rate, live.open_rate)
            row.update(stop_loss=stop_loss, take_profit=take_profit, previous_stop_loss=stop_loss, previous_take_profit=take_profit)
        if row["atr"] is None:
            continue
        stop, take_profit, reason = plan_change(row, live, bids.get(row["instrument_id"]) if live else None)
        if reason == "no change needed at the live price":
            row.update(status="ok", recommended_stop_loss=None, recommended_take_profit=None)
            continue
        if reason:
            if row["status"] == "change recommended":
                row["status"] = f"not applied: {reason}"
            continue
        row.update(
            recommended_stop_loss=stop,
            recommended_take_profit=take_profit,
            live_price=bids[row["instrument_id"]],
            status="planned",
        )
        planned.append(row)

    for position in snapshot["positions"]:
        live = positions.get(position["position_id"])
        if live is not None:
            position["stop_loss_rate"], position["take_profit_rate"] = live.stop_loss_rate, live.take_profit_rate
    write_json(SNAPSHOT, snapshot)

    if not planned:
        print("Nothing to change on eToro.")
        return
    print(f"Planned changes on your REAL eToro account ({len(planned)}):")
    for row in planned:
        print(f"  {row_name(row):<9} {describe_change(row)} (eToro price {row['live_price']:g})")
    if not assume_yes:
        if not stdin_is_terminal():
            print("Not sending: confirmation needs a terminal (or pass --yes).")
            mark(planned, "not applied: needs confirmation (run in a terminal, or pass --yes)")
            return
        if input("Send these changes to eToro? Type 'yes' to confirm: ").strip().lower() != "yes":
            print("Nothing sent.")
            mark(planned, "not applied: not confirmed")
            return

    with make_client() as client:
        for index, row in enumerate(planned):
            row["status"], stop_reason, details = send_change(client, row)
            audit("send", row, **details)
            if stop_reason:
                mark(planned[index + 1:], f"not applied: {stop_reason}")
                break

    waiting = [row for row in planned if row["status"] == "sent"]
    for _ in range(CONFIRM_ATTEMPTS):
        if not waiting:
            break
        time.sleep(CONFIRM_DELAY_SECONDS)
        try:
            waiting = reconcile(planned, live_levels())
        except Exception as exc:  # noqa: BLE001 - leave the edits marked unconfirmed
            print(f"Could not re-read positions from eToro ({type(exc).__name__}: {exc}).")
            break
    mark(waiting, "sent, but eToro doesn't show it yet: check in the eToro app")
    for row in planned:
        audit("result", row)

    confirmed = {row["position_id"]: row for row in planned if row["status"] == "set on eToro"}
    for position in snapshot["positions"]:
        row = confirmed.get(position["position_id"])
        if row:
            position["stop_loss_rate"] = row["stop_loss"] or position["stop_loss_rate"]
            position["take_profit_rate"] = row["take_profit"] or position["take_profit_rate"]
    if confirmed:
        write_json(SNAPSHOT, snapshot)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="Send recommended stop loss / take profit changes to eToro")
    parser.add_argument("--yes", action="store_true", help="With --apply: don't ask for confirmation")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    auto_apply = os.environ.get("ETORO_AUTO_APPLY_STOPS", "").lower() in ("1", "true", "yes", "on")
    apply = args.apply or auto_apply
    if not SNAPSHOT.exists():
        print(f"No account snapshot at {SNAPSHOT}; run scripts/sync_etoro_portfolio.py first.")
        return 1
    snapshot = json.loads(SNAPSHOT.read_text())
    if apply and (snapshot.get("mode") != "real" or os.environ.get("ETORO_MODE") != "real"):
        print("Applying changes is only supported for real accounts (ETORO_MODE=real); reporting only.")
        apply = False
    if apply and not (os.environ.get("ETORO_API_KEY") and os.environ.get("ETORO_USER_KEY")):
        print("ETORO_API_KEY / ETORO_USER_KEY missing in .env; reporting only.")
        apply = False

    ratings = latest_ratings()
    rows = [build_row(position, ratings) for position in snapshot["positions"]]
    if apply:
        apply_changes(rows, snapshot, assume_yes=args.yes or auto_apply)

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
    unprotected = [row_name(row) for row in rows if row["stop_loss"] is None and row["amount"]]
    if unprotected:
        warnings.append(f"No stop loss set on: {', '.join(unprotected)}.")
    tight = [
        f"{row_name(row)} ({row['stop_loss']:g}, {1 - row['stop_loss'] / row['price']:.1%} below)"
        for row in rows
        if row["atr"] is not None and row["is_buy"] and row["stop_loss"] and 1 - row["stop_loss"] / row["price"] < MIN_STOP_DISTANCE
    ]
    if tight:
        warnings.append(f"Stop loss within {MIN_STOP_DISTANCE:.0%} of the price, a normal move could close it: {', '.join(tight)}.")

    for row in rows:
        row["atr"] = round_rate(row["atr"])
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
    write_json(REPORT, report)

    print(
        f"Free cash ${available_cash:,.2f} | invested ${report['invested']:,.2f} | "
        f"positions worth ${positions_value:,.2f} | equity ${equity:,.2f}"
    )
    for row in rows:
        if row["status"] != "ok":
            change = describe_change(row)
            print(f"  {row_name(row):<9} {row['status']}" + (f": {change}" if change else ""))
    for warning in warnings:
        print(f"  ! {warning}")
    print(f"Wrote {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
