"""Only one portfolio review may run at a time."""

from __future__ import annotations

import datetime
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "portfolio_review.py"
_spec = importlib.util.spec_from_file_location("portfolio_review", _SCRIPT)
review = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(review)


@pytest.mark.unit
def test_second_lock_is_refused_until_the_first_is_released(tmp_path):
    lock_path = tmp_path / "portfolio_review.lock"

    first, _ = review.acquire_run_lock(lock_path)
    second, holder = review.acquire_run_lock(lock_path)

    assert first is not None
    assert second is None
    assert f"pid {os.getpid()}" in holder

    first.close()
    third, _ = review.acquire_run_lock(lock_path)
    assert third is not None
    third.close()


@pytest.mark.unit
def test_script_exits_without_analyzing_while_another_review_runs(tmp_path):
    held, _ = review.acquire_run_lock(tmp_path / "portfolio_review.lock")
    try:
        result = subprocess.run(
            [sys.executable, str(_SCRIPT), "AAPL"],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "TRADINGAGENTS_RESULTS_DIR": str(tmp_path)},
        )
    finally:
        held.close()

    assert result.returncode == 1
    assert "already running" in result.stdout
    assert "=== AAPL" not in result.stdout
    assert not (tmp_path / "progress.json").exists()


_WINDOW = review.parse_window("02:00-06:00")


@pytest.mark.unit
def test_manual_run_stops_on_usage_limit_instead_of_waiting():
    now = datetime.datetime(2026, 9, 15, 3, 0)

    assert review.usage_limit_retry_at(None, now, datetime.timedelta(0)) is None


@pytest.mark.unit
def test_scheduled_run_retries_while_a_ticker_still_fits_the_window():
    now = datetime.datetime(2026, 9, 15, 3, 0)

    assert review.usage_limit_retry_at(_WINDOW, now, datetime.timedelta(minutes=7)) == datetime.datetime(2026, 9, 15, 3, 15)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("now", "longest_run"),
    [
        (datetime.datetime(2026, 9, 15, 5, 40), datetime.timedelta(0)),  # retry would start after 05:50
        (datetime.datetime(2026, 9, 15, 4, 30), datetime.timedelta(hours=1, minutes=30)),  # no time to finish
        (datetime.datetime(2026, 9, 15, 5, 50), datetime.timedelta(0)),  # retry lands past the window
    ],
)
def test_scheduled_run_stops_when_no_ticker_could_finish_after_the_wait(now, longest_run):
    assert review.usage_limit_retry_at(_WINDOW, now, longest_run) is None
