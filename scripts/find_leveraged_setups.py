#!/usr/bin/env python3
"""Short-term leveraged trade ideas with a small, fixed maximum loss.

Leverage multiplies losses exactly as much as gains, so no leveraged trade is
low risk. These ideas only keep the risk small and known before you enter:

- only stocks TradingAgents rated Buy or Overweight in the last 7 days
  (your holdings and the nightly buy candidates)
- in an uptrend (price above its 50-day average) after a pullback (RSI 40-60)
- calm enough: 14-day average true range (ATR) at most 3% of the price
- plan: long at 2x leverage, stop loss 1.5 ATR below, take profit 3 ATR above
- amount sized so hitting the stop loses at most 1% of your equity, and never
  more than your free cash

Writes ~/.tradingagents/etoro/leveraged.json for show_stocks.py. Nothing is
opened on eToro.

Usage:
    scripts/find_leveraged_setups.py
"""

from __future__ import annotations

import datetime
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"

# Run directly, the shebang picks the system python, which lacks the project's
# dependencies; switch to the repo's venv.
if VENV_PYTHON.exists() and Path(sys.prefix).resolve() != VENV_PYTHON.parent.parent.resolve():
    os.execv(VENV_PYTHON, [str(VENV_PYTHON), *sys.argv])

import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402

from tradingagents.agents.utils.memory import TradingMemoryLog  # noqa: E402
from tradingagents.default_config import DEFAULT_CONFIG  # noqa: E402

ETORO_DIR = Path.home() / ".tradingagents" / "etoro"
ACCOUNT_REPORT = ETORO_DIR / "report.json"
OUTPUT = ETORO_DIR / "leveraged.json"

RATED_WITHIN_DAYS = 7
BULLISH_RATINGS = {"buy", "overweight"}
LEVERAGE = 2
STOP_ATR = 1.5
TARGET_ATR = 3.0
RSI_LOW, RSI_HIGH = 40.0, 60.0
MAX_ATR_SHARE = 0.03
RISK_PER_TRADE = 0.01
TREND_DAYS = 50
PERIOD_DAYS = 14


def average_true_range(history: pd.DataFrame, days: int = PERIOD_DAYS) -> float | None:
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


def relative_strength_index(close: pd.Series, days: int = PERIOD_DAYS) -> float | None:
    if len(close) <= days:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / days, adjust=False).mean().iloc[-1]
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / days, adjust=False).mean().iloc[-1]
    return 100.0 if loss == 0 else float(100 - 100 / (1 + gain / loss))


def rejection_reason(price: float, trend: float, rsi: float | None, atr: float) -> str | None:
    """Why a stock is not a setup, or None when it is one."""
    if price <= trend:
        return f"below its {TREND_DAYS}-day average"
    if rsi is None:
        return "too little price history for RSI"
    if not RSI_LOW <= rsi <= RSI_HIGH:
        return f"RSI {rsi:.0f} outside {RSI_LOW:.0f}-{RSI_HIGH:.0f}"
    if atr / price > MAX_ATR_SHARE:
        return f"too volatile (daily range {atr / price:.1%} of price)"
    return None


def plan_trade(price: float, atr: float, equity: float | None, available_cash: float | None) -> dict:
    stop = price - STOP_ATR * atr
    target = price + TARGET_ATR * atr
    stop_move = (price - stop) / price
    plan = {
        "leverage": LEVERAGE,
        "stop_loss": round(stop, 2),
        "take_profit": round(target, 2),
        "amount": None,
        "max_loss": None,
        "potential_gain": None,
    }
    if equity:
        amount = equity * RISK_PER_TRADE / (LEVERAGE * stop_move)
        if available_cash is not None:
            amount = min(amount, max(available_cash, 0.0))
        plan.update(
            amount=round(amount, 2),
            max_loss=round(amount * LEVERAGE * stop_move, 2),
            potential_gain=round(amount * LEVERAGE * (target - price) / price, 2),
        )
    return plan


def bullish_tickers(days: int) -> dict[str, tuple[str, str]]:
    cutoff = (datetime.date.today() - datetime.timedelta(days=days)).isoformat()
    latest: dict[str, dict] = {}
    for entry in TradingMemoryLog(DEFAULT_CONFIG).load_entries():
        ticker = entry["ticker"].upper()
        if ticker not in latest or entry["date"] >= latest[ticker]["date"]:
            latest[ticker] = entry
    return {
        ticker: (e["rating"], e["date"])
        for ticker, e in latest.items()
        if e["date"] >= cutoff and e["rating"].lower() in BULLISH_RATINGS and not ticker.endswith("-USD")
    }


def main() -> int:
    account = json.loads(ACCOUNT_REPORT.read_text()) if ACCOUNT_REPORT.exists() else {}
    equity, available_cash = account.get("equity"), account.get("available_cash")

    setups, rejected = [], []
    for ticker, (rating, rating_date) in sorted(bullish_tickers(RATED_WITHIN_DAYS).items()):
        history = yf.Ticker(ticker).history(period="6mo", auto_adjust=False)
        if len(history) < TREND_DAYS:
            rejected.append({"ticker": ticker, "reason": "too little price history"})
            continue
        close = history["Close"]
        price = float(close.iloc[-1])
        trend = float(close.tail(TREND_DAYS).mean())
        rsi = relative_strength_index(close)
        atr = average_true_range(history)
        reason = "too little price history" if atr is None else rejection_reason(price, trend, rsi, atr)
        if reason:
            rejected.append({"ticker": ticker, "reason": reason})
            continue
        setups.append({
            "ticker": ticker,
            "rating": rating,
            "rating_date": rating_date,
            "price": round(price, 2),
            "rsi": round(rsi, 1),
            "atr_share": round(atr / price, 4),
            **plan_trade(price, atr, equity, available_cash),
        })
    setups.sort(key=lambda s: s["atr_share"])

    ETORO_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT.with_name(OUTPUT.name + ".tmp")
    tmp.write_text(json.dumps({
        "generated_at": datetime.datetime.now().isoformat(timespec="minutes"),
        "equity": equity,
        "available_cash": available_cash,
        "setups": setups,
        "rejected": rejected,
    }, indent=2))
    tmp.replace(OUTPUT)

    print(f"{len(setups)} leveraged setups, {len(rejected)} rejected")
    for s in setups:
        sizing = f"${s['amount']:,.2f}, max loss ${s['max_loss']:,.2f}" if s["amount"] is not None else "no equity data for sizing"
        print(f"  {s['ticker']:<8} {s['leverage']}x at {s['price']}: SL {s['stop_loss']}, TP {s['take_profit']} ({sizing})")
    for r in rejected:
        print(f"  - {r['ticker']}: {r['reason']}")
    print(f"Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
