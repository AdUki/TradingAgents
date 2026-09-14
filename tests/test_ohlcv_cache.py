"""The OHLCV price cache must survive empty files (parallel tool calls read it)."""

from __future__ import annotations

import pandas as pd
import pytest

from tradingagents.dataflows import stockstats_utils as su
from tradingagents.dataflows.config import set_config
from tradingagents.default_config import DEFAULT_CONFIG


def _prices() -> pd.DataFrame:
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize() - pd.Timedelta(days=3), periods=30)
    return pd.DataFrame(
        {"Open": 10.0, "High": 11.0, "Low": 9.0, "Close": 10.5, "Volume": 1000},
        index=pd.DatetimeIndex(dates, name="Date"),
    )


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    set_config({**DEFAULT_CONFIG, "data_cache_dir": str(tmp_path)})
    monkeypatch.setattr(su.yf, "download", lambda *a, **k: _prices())
    yield tmp_path
    set_config(DEFAULT_CONFIG.copy())


def _cache_file(cache_dir, symbol):
    today = pd.Timestamp.today()
    start = (today - pd.DateOffset(years=5)).strftime("%Y-%m-%d")
    end = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return cache_dir / f"{symbol}-YFin-data-{start}-{end}.csv"


@pytest.mark.unit
def test_empty_cache_file_is_refetched_not_fatal(cache_dir):
    cache = _cache_file(cache_dir, "TEST")
    cache.write_text("")
    curr_date = _prices().index[-1].strftime("%Y-%m-%d")

    data = su.load_ohlcv("TEST", curr_date)

    assert not data.empty
    assert cache.stat().st_size > 0


@pytest.mark.unit
def test_cache_write_leaves_no_temp_files(cache_dir):
    curr_date = _prices().index[-1].strftime("%Y-%m-%d")

    su.load_ohlcv("TEST", curr_date)

    assert _cache_file(cache_dir, "TEST").exists()
    assert list(cache_dir.glob("*.tmp")) == []
