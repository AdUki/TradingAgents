"""Calculations in the eToro helper scripts (no network)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip("etoropy")

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


account = _load("etoro_account")
candidates = _load("find_buy_candidates")
leveraged = _load("find_leveraged_setups")
sync = _load("sync_etoro_portfolio")


# --- stop loss / take profit -------------------------------------------------

@pytest.mark.unit
def test_placeholder_stop_loss_counts_as_unset():
    assert account.effective_rate(0.0001, 238.0) is None
    assert account.effective_rate(0.01, 82.72) is None
    assert account.effective_rate(200.0, 238.0) == 200.0


@pytest.mark.unit
def test_missing_stop_is_set_below_price_by_rating_multiple():
    stop, take_profit = account.recommend_levels(
        is_buy=True, price=100.0, atr=2.0, rating="Hold", stop_loss=None, take_profit=150.0
    )
    assert stop == pytest.approx(95.0)  # 2.5 ATR for Hold
    assert take_profit is None  # an existing take profit is kept


@pytest.mark.unit
def test_worse_rating_tightens_the_stop():
    buy_stop, _ = account.recommend_levels(is_buy=True, price=100.0, atr=2.0, rating="Buy", stop_loss=None, take_profit=150.0)
    sell_stop, _ = account.recommend_levels(is_buy=True, price=100.0, atr=2.0, rating="Sell", stop_loss=None, take_profit=150.0)
    assert buy_stop == pytest.approx(94.0)
    assert sell_stop == pytest.approx(98.0)


@pytest.mark.unit
def test_stop_is_never_loosened():
    stop, _ = account.recommend_levels(
        is_buy=True, price=100.0, atr=2.0, rating="Buy", stop_loss=96.0, take_profit=150.0
    )
    assert stop is None


@pytest.mark.unit
def test_tiny_stop_moves_are_ignored():
    stop, _ = account.recommend_levels(
        is_buy=True, price=100.0, atr=2.0, rating="Hold", stop_loss=94.9, take_profit=150.0
    )
    assert stop is None  # 95.0 is under a 0.5% move


@pytest.mark.unit
def test_missing_take_profit_is_twice_the_stop_distance():
    stop, take_profit = account.recommend_levels(
        is_buy=True, price=100.0, atr=2.0, rating="Hold", stop_loss=None, take_profit=None
    )
    assert take_profit == pytest.approx(100.0 + 2 * (100.0 - stop))


@pytest.mark.unit
def test_short_positions_mirror_the_rules():
    stop, take_profit = account.recommend_levels(
        is_buy=False, price=100.0, atr=2.0, rating="Hold", stop_loss=None, take_profit=None
    )
    assert stop == pytest.approx(105.0)
    assert take_profit == pytest.approx(90.0)


@pytest.mark.unit
def test_position_value_uses_units_and_currency():
    value, pnl = account.position_value(is_buy=True, amount=140.0, units=2.0, entry=50.0, price=55.0, usd_per_unit=1.2)
    assert pnl == pytest.approx(12.0)
    assert value == pytest.approx(152.0)


@pytest.mark.unit
def test_average_true_range_includes_gaps():
    history = pd.DataFrame({
        "High": [10.0, 12.0, 11.0],
        "Low": [9.0, 11.0, 10.0],
        "Close": [9.5, 11.5, 10.5],
    })
    # true ranges: 1, max(1, |12-9.5|, |11-9.5|)=2.5, max(1, |11-11.5|, |10-11.5|)=1.5
    assert account.average_true_range(history, days=3) == pytest.approx((1 + 2.5 + 1.5) / 3)
    assert account.average_true_range(history, days=4) is None


# --- buy candidates ------------------------------------------------------------

def _quote(symbol, **overrides):
    quote = {
        "symbol": symbol, "quoteType": "EQUITY", "currency": "USD", "shortName": symbol,
        "averageAnalystRating": "1.8 - Buy", "regularMarketPrice": 50.0, "marketCap": 5e10, "screen": "s",
    }
    quote.update(overrides)
    return quote


@pytest.mark.unit
def test_analyst_score_parses_yahoo_rating():
    assert candidates.analyst_score("1.6 - Buy") == 1.6
    assert candidates.analyst_score(None) is None


@pytest.mark.unit
def test_screen_filters_and_orders_candidates():
    quotes = [
        _quote("HELD"),
        _quote("HOLDRATED", averageAnalystRating="2.8 - Hold"),
        _quote("PENNY", regularMarketPrice=3.0),
        _quote("TINY", marketCap=1e8),
        _quote("FOREIGN", currency="EUR"),
        _quote("ETF", quoteType="ETF"),
        _quote("GOOD", averageAnalystRating="1.9 - Buy"),
        _quote("BEST", averageAnalystRating="1.5 - Strong Buy"),
        _quote("BEST"),
    ]
    picked = candidates.screen_candidates(quotes, exclude={"HELD"})
    assert [c["symbol"] for c in picked] == ["BEST", "GOOD"]


# --- leveraged setups --------------------------------------------------------------

@pytest.mark.unit
def test_setup_rules():
    assert leveraged.rejection_reason(price=100, trend=90, rsi=50, atr=2) is None
    assert "average" in leveraged.rejection_reason(price=100, trend=110, rsi=50, atr=2)
    assert "RSI" in leveraged.rejection_reason(price=100, trend=90, rsi=75, atr=2)
    assert "volatile" in leveraged.rejection_reason(price=100, trend=90, rsi=50, atr=5)


@pytest.mark.unit
def test_trade_is_sized_to_lose_one_percent_at_the_stop():
    plan = leveraged.plan_trade(price=100.0, atr=2.0, equity=10_000.0, available_cash=5_000.0)
    assert plan["stop_loss"] == 97.0 and plan["take_profit"] == 106.0
    assert plan["max_loss"] == pytest.approx(100.0)
    assert plan["potential_gain"] == pytest.approx(200.0)


@pytest.mark.unit
def test_trade_amount_is_capped_by_free_cash():
    plan = leveraged.plan_trade(price=100.0, atr=2.0, equity=10_000.0, available_cash=50.0)
    assert plan["amount"] == 50.0
    assert plan["max_loss"] == pytest.approx(3.0)


@pytest.mark.unit
def test_rsi_bounds():
    rising = pd.Series(range(1, 40), dtype=float)
    assert leveraged.relative_strength_index(rising) == 100.0
    assert leveraged.relative_strength_index(pd.Series([1.0, 2.0])) is None


# --- account snapshot ---------------------------------------------------------------

@pytest.mark.unit
def test_pending_orders_follow_etoros_available_cash_formula():
    portfolio = SimpleNamespace(
        orders_for_open=[
            SimpleNamespace(amount=200.0, model_extra={"mirrorID": 0}),
            SimpleNamespace(amount=200.0, model_extra={}),
            SimpleNamespace(amount=100.0, model_extra={"mirrorID": 77}),
        ],
        orders=[SimpleNamespace(amount=150.0)],
    )
    assert sync.pending_orders_amount(portfolio) == 550.0
