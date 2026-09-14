"""Only one portfolio review may run at a time."""

from __future__ import annotations

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
