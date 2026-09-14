#!/usr/bin/env python3
"""Run TradingAgents analysis over a list of tickers you hold and print a
summary table of the decisions.

TradingAgents has no concept of "your portfolio" — each ticker is analyzed
independently, with no knowledge of your position size or entry price. This
script just loops ``TradingAgentsGraph.propagate()`` over a ticker list so
you can check your whole eToro (or any) portfolio in one pass instead of
running the CLI once per stock.

Usage:
    # tickers from portfolio.txt in the repo root (the default)
    scripts/portfolio_review.py

    # tickers as arguments
    scripts/portfolio_review.py AAPL TSLA BTC-USD

    # or from files, one ticker per line (# comments allowed); repeatable,
    # analyzed in order
    scripts/portfolio_review.py --file portfolio.txt --file candidates.txt

    # pin the analysis date (defaults to today)
    scripts/portfolio_review.py AAPL TSLA --date 2026-09-01

    # only analyze inside a local-time window (for scheduled runs)
    scripts/portfolio_review.py --window 02:00-06:00

Requires the API key for whichever provider is configured (see .env /
TRADINGAGENTS_LLM_PROVIDER) to be set. Analyzing many tickers makes that
many full agent runs, which costs real LLM tokens and takes a few minutes
per ticker — start with a short list.

Only one review runs at a time: a second launch exits while another holds
<results_dir>/portfolio_review.lock.

Progress is printed as it happens (one line per model call with the
subscription CLI providers, plus a heartbeat every minute) and kept in
<results_dir>/progress.json while the run lasts. The summary is written to
<results_dir>/portfolio_reviews/<date>.txt.
"""

from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import logging
import os
import sys
import threading
import traceback
from pathlib import Path
from typing import IO

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
DEFAULT_PORTFOLIO = REPO_ROOT / "portfolio.txt"

# Run directly, the shebang picks the system python, which lacks the project's
# dependencies; switch to the repo's venv.
if VENV_PYTHON.exists() and Path(sys.prefix).resolve() != VENV_PYTHON.parent.parent.resolve():
    os.execv(VENV_PYTHON, [str(VENV_PYTHON), *sys.argv])

from tradingagents.default_config import DEFAULT_CONFIG  # noqa: E402

HEARTBEAT_SECONDS = 60
STEP_LOGGER = "tradingagents.llm_clients.subscription_client"


def acquire_run_lock(path: Path) -> tuple[IO[str] | None, str]:
    """Take the single-run lock: (open lock file, "") or (None, current holder).

    The OS drops the lock when the holding process exits, even on a crash or
    kill, so a stale lock can't block the next run. Keep the returned file open
    for as long as the run lasts.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+")  # noqa: SIM115 - must stay open to hold the lock
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.seek(0)
        holder = handle.read().strip() or "unknown process"
        handle.close()
        return None, holder
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid {os.getpid()}, started {datetime.datetime.now():%Y-%m-%d %H:%M}\n")
    handle.flush()
    return handle, ""


def load_tickers(args: argparse.Namespace) -> list[str]:
    tickers = list(args.tickers)
    for path in args.file or []:
        if not Path(path).exists():
            # A missing optional list (e.g. candidates.txt before its first
            # sync) must not cancel the review of the others.
            print(f"Skipping missing ticker file {path}", flush=True)
            continue
        with open(path) as f:
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


def parse_window(text: str) -> tuple[datetime.time, datetime.time]:
    try:
        start_s, end_s = text.split("-")
        start = datetime.time.fromisoformat(start_s)
        end = datetime.time.fromisoformat(end_s)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected HH:MM-HH:MM, got {text!r}") from exc
    if start >= end:
        raise argparse.ArgumentTypeError("window must start before it ends, within one day")
    return start, end


def window_block_reason(
    window: tuple[datetime.time, datetime.time],
    now: datetime.datetime,
    longest_run: datetime.timedelta,
) -> str | None:
    """Why a ticker must not start now, or None if it fits in the window.

    A ticker is only started when the longest ticker so far would still finish
    before the window closes, so runs don't spill past the end.
    """
    start, end = window
    if not start <= now.time() < end:
        return f"{now:%H:%M} is outside {start:%H:%M}-{end:%H:%M}"
    window_end = now.replace(hour=end.hour, minute=end.minute, second=0, microsecond=0)
    if now + longest_run > window_end:
        return f"not enough time left before {end:%H:%M} (a ticker takes up to {longest_run})"
    return None


def minutes_since(stamp: str) -> int:
    return int((datetime.datetime.now() - datetime.datetime.fromisoformat(stamp)).total_seconds() // 60)


class Progress(logging.Handler):
    """Tracks the ticker and step being analyzed, prints a heartbeat, and
    mirrors the state to a JSON file that show_stocks.py reads."""

    def __init__(self, path: Path, total: int):
        super().__init__(level=logging.INFO)
        self.path = path
        self.state_lock = threading.Lock()
        now = datetime.datetime.now().isoformat(timespec="seconds")
        self.state = {
            "run_started": now, "total": total, "done": 0, "index": 0,
            "ticker": None, "ticker_started": None, "step": None, "reply": None, "updated": now,
        }
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)

    def start(self) -> None:
        self.update()
        self.thread.start()

    def emit(self, record: logging.LogRecord) -> None:
        step = getattr(record, "step", None)
        if step:
            self.update(step=step, reply=getattr(record, "reply", None))

    def update(self, **changes) -> None:
        with self.state_lock:
            self.state.update(changes, updated=datetime.datetime.now().isoformat(timespec="seconds"))
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps(self.state, indent=2))
            tmp.replace(self.path)

    def _heartbeat(self) -> None:
        while not self.stopped.wait(HEARTBEAT_SECONDS):
            with self.state_lock:
                state = dict(self.state)
            if state["ticker"]:
                step = f", last step: {state['step']}" if state["step"] else ""
                print(
                    f"  ... still analyzing {state['ticker']} ({state['index']}/{state['total']}, "
                    f"{minutes_since(state['ticker_started'])}m){step}",
                    flush=True,
                )
            self.update()

    def finish(self) -> None:
        self.stopped.set()
        self.thread.join()
        self.path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tickers", nargs="*", help="Tickers to analyze, e.g. AAPL TSLA BTC-USD")
    parser.add_argument("--file", action="append", help=f"File with one ticker per line; repeatable (default: {DEFAULT_PORTFOLIO} when no tickers are given)")
    parser.add_argument("--date", default=datetime.date.today().isoformat(), help="Analysis date, YYYY-MM-DD (default: today)")
    parser.add_argument("--window", type=parse_window, help="Only start tickers inside this local-time window, e.g. 02:00-06:00")
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    results_dir = Path(config["results_dir"])
    lock, holder = acquire_run_lock(results_dir / "portfolio_review.lock")
    if lock is None:
        print(f"Another portfolio review is already running ({holder}); not starting a second one.", flush=True)
        return 1

    if not args.tickers and not args.file and DEFAULT_PORTFOLIO.exists():
        args.file = [str(DEFAULT_PORTFOLIO)]
    tickers = load_tickers(args)
    if not tickers:
        parser.error(f"no tickers given — pass them as arguments, via --file, or list them in {DEFAULT_PORTFOLIO}")

    # Imported after the lock: loading the graph takes a while on a Pi.
    from cli.utils import detect_asset_type
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    ta = TradingAgentsGraph(debug=False, config=config)

    progress = Progress(results_dir / "progress.json", len(tickers))
    step_log = logging.getLogger(STEP_LOGGER)
    step_log.setLevel(logging.INFO)
    step_log.propagate = False
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("  %(asctime)s %(message)s", "%H:%M:%S"))
    step_log.addHandler(console)
    step_log.addHandler(progress)
    progress.start()

    results: list[tuple[str, str]] = []
    longest_run = datetime.timedelta(0)
    failed = False
    try:
        for i, ticker in enumerate(tickers):
            if args.window:
                reason = window_block_reason(args.window, datetime.datetime.now(), longest_run)
                if reason:
                    print(f"\nStopping: {reason}. Skipped: {', '.join(tickers[i:])}", flush=True)
                    results.extend((t, "SKIPPED (outside window)") for t in tickers[i:])
                    break
            asset_type = detect_asset_type(ticker).value
            started = datetime.datetime.now()
            print(f"\n=== {ticker} ({i + 1}/{len(tickers)}, {args.date}, {asset_type}) started {started:%H:%M} ===", flush=True)
            progress.update(ticker=ticker, index=i + 1, ticker_started=started.isoformat(timespec="seconds"), step=None, reply=None)
            try:
                _, decision = ta.propagate(ticker, args.date, asset_type=asset_type)
            except Exception as exc:  # noqa: BLE001 - keep going, report every ticker
                print(f"  ERROR: {exc}", flush=True)
                traceback.print_exc()
                results.append((ticker, f"ERROR: {exc}"))
                failed = True
                continue
            finally:
                took = datetime.datetime.now() - started
                longest_run = max(longest_run, took)
                progress.update(done=i + 1)
            print(f"  Decision: {decision} (took {int(took.total_seconds() // 60)}m)", flush=True)
            results.append((ticker, decision))
    finally:
        progress.finish()

    width = max(len(t) for t, _ in results)
    summary = "\n".join(
        ["=" * 40, f"Portfolio review — {args.date}", "=" * 40]
        + [f"  {ticker:<{width}}  {decision}" for ticker, decision in results]
    )
    print("\n" + summary)

    summary_path = results_dir / "portfolio_reviews" / f"{args.date}.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary + "\n")
    print(f"\nSummary saved to {summary_path}")

    lock.close()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
