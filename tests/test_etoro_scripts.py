"""Calculations and safety rules in the eToro helper scripts (no network)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pandas as pd
import pytest

pytest.importorskip("etoropy")

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))  # the scripts import etoro_api


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


account = _load("etoro_account")
candidates = _load("find_buy_candidates")
leveraged = _load("find_leveraged_setups")
sync = _load("sync_etoro_portfolio")


# --- stop loss / take profit rules ------------------------------------------------

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
def test_worse_rating_tightens_the_stop_but_never_within_min_distance():
    buy_stop, _ = account.recommend_levels(is_buy=True, price=100.0, atr=2.0, rating="Buy", stop_loss=None, take_profit=150.0)
    underweight_stop, _ = account.recommend_levels(is_buy=True, price=100.0, atr=2.0, rating="Underweight", stop_loss=None, take_profit=150.0)
    sell_stop, _ = account.recommend_levels(is_buy=True, price=100.0, atr=2.0, rating="Sell", stop_loss=None, take_profit=150.0)
    assert buy_stop == pytest.approx(94.0)
    assert underweight_stop == pytest.approx(97.0)  # 1.5 ATR = 3%, the minimum
    assert sell_stop == pytest.approx(97.0)  # 1 ATR = 2% is closer than 3%


@pytest.mark.unit
def test_pinned_price_stock_gets_min_distance_stop():
    # A stock under a takeover offer barely moves; its ATR would put the stop a cent away.
    stop, _ = account.recommend_levels(is_buy=True, price=13.41, atr=0.05, rating=None, stop_loss=None, take_profit=67.0)
    assert stop == pytest.approx(13.41 * 0.97)


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
def test_floor_rate_rounds_toward_a_deeper_stop():
    assert account.floor_rate(95.678) == 95.67
    assert account.floor_rate(0.123456) == 0.1234


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


# --- planning an edit from live eToro data -----------------------------------------

def _live(**overrides):
    live = {
        "position_id": 1, "instrument_id": 1001, "is_buy": True, "leverage": 1, "open_rate": 80.0,
        "stop_loss_rate": 0.0001, "take_profit_rate": 150.0, "is_tsl_enabled": False, "mirror_id": 0,
    }
    live.update(overrides)
    return SimpleNamespace(**live)


def _row(**overrides):
    row = {
        "ticker": "TEST", "symbol": "TEST", "position_id": 1, "instrument_id": 1001, "is_buy": True,
        "open_rate": 80.0, "price": 100.0, "atr": 2.0, "rating": "Hold", "status": "change recommended",
        "stop_loss": None, "take_profit": 150.0, "previous_stop_loss": None, "previous_take_profit": 150.0,
        "recommended_stop_loss": 95.0, "recommended_take_profit": None,
    }
    row.update(overrides)
    return row


@pytest.mark.unit
@pytest.mark.parametrize(
    ("live", "reason"),
    [
        (None, "no longer open"),
        (_live(mirror_id=5), "copy-trading"),
        (_live(is_buy=False), "short position"),
        (_live(leverage=2), "2x leveraged"),
        (_live(is_tsl_enabled=True), "trailing stop"),
    ],
)
def test_plan_leaves_non_plain_positions_alone(live, reason):
    stop, take_profit, why = account.plan_change(_row(), live, 100.0)
    assert (stop, take_profit) == (None, None)
    assert reason in why


@pytest.mark.unit
def test_plan_refuses_when_live_and_yahoo_prices_disagree():
    _, _, why = account.plan_change(_row(price=100.0), _live(), 1.0)  # e.g. pence vs pounds
    assert "differ" in why
    _, _, why = account.plan_change(_row(price=100.0), _live(), 0.0)
    assert "no live eToro price" in why


@pytest.mark.unit
def test_plan_uses_the_live_price_and_scales_atr():
    stop, take_profit, why = account.plan_change(_row(price=100.0, atr=2.0), _live(), 92.0)
    assert why is None
    assert stop == account.floor_rate(92.0 - 2.5 * 2.0 * 0.92)
    assert stop < 92.0 * (1 - account.MIN_STOP_DISTANCE)
    assert take_profit is None


@pytest.mark.unit
def test_plan_never_loosens_a_stop_tightened_since_the_snapshot():
    # The snapshot said "no stop", but the live position already has a tighter one.
    stop, take_profit, why = account.plan_change(_row(), _live(stop_loss_rate=97.0), 100.0)
    assert (stop, take_profit) == (None, None)
    assert "no change needed" in why


@pytest.mark.unit
def test_plan_rejects_implausibly_deep_stops():
    _, _, why = account.plan_change(_row(atr=30.0), _live(), 100.0)
    assert "safety check" in why


@pytest.mark.unit
def test_plan_adds_take_profit_only_when_none_is_set():
    stop, take_profit, why = account.plan_change(_row(), _live(take_profit_rate=0.0), 100.0)
    assert why is None
    assert take_profit == pytest.approx(100.0 + 2 * (100.0 - stop))


# --- sending ------------------------------------------------------------------

def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status_code", "expected_status", "stops_the_rest"),
    [
        (202, "sent", False),
        (401, "not applied: the eToro key lacks Write permission", True),
        (403, "not applied: the eToro key lacks Write permission", True),
        (429, "not applied: eToro rate limit reached, try again later", True),
        (404, "not applied: position closed", False),
        (409, "not applied: position closed", False),
        (400, "rejected by eToro (400: bad rate)", False),
    ],
)
def test_send_change_maps_responses(status_code, expected_status, stops_the_rest):
    client = _client(lambda request: httpx.Response(status_code, text="bad rate" if status_code == 400 else "{}"))
    status, stop_reason, details = account.send_change(client, _row())
    assert status == expected_status
    assert bool(stop_reason) is stops_the_rest
    assert details["http_status"] == status_code


@pytest.mark.unit
def test_send_change_body_and_path():
    seen = {}

    def handler(request):
        seen.update(method=request.method, url=str(request.url), body=json.loads(request.content), request_id=request.headers.get("x-request-id"))
        return httpx.Response(202, json={"operationId": "op"})

    account.send_change(_client(handler), _row(recommended_stop_loss=95.0, recommended_take_profit=110.0))
    assert seen["method"] == "PATCH"
    assert seen["url"].endswith("/api/v2/trading/positions/1")
    assert seen["body"] == {"stopLossRate": 95.0, "stopLossType": "fixed", "takeProfitRate": 110.0}
    assert seen["request_id"]


@pytest.mark.unit
def test_send_change_without_response_is_left_for_confirmation():
    def handler(request):
        raise httpx.ReadTimeout("timed out", request=request)

    status, stop_reason, details = account.send_change(_client(handler), _row())
    assert (status, stop_reason) == ("sent", None)
    assert "ReadTimeout" in details["error"]


# --- the whole --apply flow -----------------------------------------------------

@pytest.fixture
def apply_env(tmp_path, monkeypatch):
    monkeypatch.setattr(account, "AUDIT_LOG", tmp_path / "audit.jsonl")
    monkeypatch.setattr(account, "APPLY_LOCK", tmp_path / "apply.lock")
    monkeypatch.setattr(account, "SNAPSHOT", tmp_path / "account.json")
    monkeypatch.setattr(account.time, "sleep", lambda seconds: None)
    state = {"positions": {1: _live(), 2: _live(position_id=2, instrument_id=2002)}, "bids": {1001: 100.0, 2002: 100.0}, "requests": []}

    def fake_fetch_live():
        return state["positions"], state["bids"]

    def fake_live_levels():
        return {pid: (p.stop_loss_rate, p.take_profit_rate) for pid, p in state["positions"].items()}

    def handler(request):
        body = json.loads(request.content)
        state["requests"].append((request.url.path, body))
        code = state.get("status_code", 202)
        if code == 202 and state.get("eToro_applies", True):
            position = state["positions"][int(request.url.path.rsplit("/", 1)[1])]
            position.stop_loss_rate = body.get("stopLossRate", position.stop_loss_rate)
            position.take_profit_rate = body.get("takeProfitRate", position.take_profit_rate)
        return httpx.Response(code, json={})

    monkeypatch.setattr(account, "fetch_live", fake_fetch_live)
    monkeypatch.setattr(account, "live_levels", fake_live_levels)
    monkeypatch.setattr(account, "make_client", lambda: _client(handler))
    monkeypatch.setattr(account, "stdin_is_terminal", lambda: False)
    return state


def _snapshot():
    return {"positions": [
        {"position_id": 1, "stop_loss_rate": 0.0001, "take_profit_rate": 150.0},
        {"position_id": 2, "stop_loss_rate": 0.0001, "take_profit_rate": 150.0},
    ]}


def _rows():
    return [_row(), _row(ticker="OTHER", position_id=2, instrument_id=2002)]


@pytest.mark.unit
def test_apply_with_yes_sends_confirms_audits_and_updates_snapshot(apply_env):
    rows, snapshot = _rows(), _snapshot()

    account.apply_changes(rows, snapshot, assume_yes=True)

    assert [r["status"] for r in rows] == ["set on eToro", "set on eToro"]
    assert [path for path, _ in apply_env["requests"]] == ["/api/v2/trading/positions/1", "/api/v2/trading/positions/2"]
    assert all(body["stopLossRate"] == 95.0 for _, body in apply_env["requests"])
    events = [json.loads(line) for line in account.AUDIT_LOG.read_text().splitlines()]
    assert [e["event"] for e in events] == ["send", "send", "result", "result"]
    assert events[0]["stop_loss"] == [None, 95.0] and events[0]["http_status"] == 202
    assert json.loads(account.SNAPSHOT.read_text())["positions"][0]["stop_loss_rate"] == 95.0


@pytest.mark.unit
def test_apply_without_terminal_or_yes_sends_nothing(apply_env):
    rows = _rows()

    account.apply_changes(rows, _snapshot(), assume_yes=False)

    assert apply_env["requests"] == []
    assert all("needs confirmation" in r["status"] for r in rows)


@pytest.mark.unit
@pytest.mark.parametrize(("answer", "sent"), [("y", False), ("no", False), ("", False), ("yes", True), ("YES", True)])
def test_apply_requires_typing_yes(apply_env, monkeypatch, answer, sent):
    monkeypatch.setattr(account, "stdin_is_terminal", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: answer)

    account.apply_changes(_rows(), _snapshot(), assume_yes=False)

    assert bool(apply_env["requests"]) is sent


@pytest.mark.unit
def test_apply_stops_sending_after_permission_error(apply_env):
    apply_env["status_code"] = 403
    rows = _rows()

    account.apply_changes(rows, _snapshot(), assume_yes=True)

    assert len(apply_env["requests"]) == 1
    assert all("lacks Write permission" in r["status"] for r in rows)


@pytest.mark.unit
def test_apply_reports_edits_etoro_never_shows(apply_env):
    apply_env["eToro_applies"] = False
    rows, snapshot = _rows(), _snapshot()

    account.apply_changes(rows, snapshot, assume_yes=True)

    assert all("doesn't show it yet" in r["status"] for r in rows)
    assert json.loads(account.SNAPSHOT.read_text())["positions"][0]["stop_loss_rate"] == 0.0001


@pytest.mark.unit
def test_apply_without_live_data_sends_nothing(apply_env, monkeypatch):
    def broken():
        raise RuntimeError("eToro down")

    monkeypatch.setattr(account, "fetch_live", broken)
    rows = _rows()

    account.apply_changes(rows, _snapshot(), assume_yes=True)

    assert apply_env["requests"] == []
    assert all(r["status"] == "not applied: live eToro data unavailable" for r in rows)


@pytest.mark.unit
def test_apply_skips_positions_changed_since_the_snapshot(apply_env):
    apply_env["positions"][1].stop_loss_rate = 97.0  # tightened in the eToro app
    del apply_env["positions"][2]  # closed
    rows = _rows()

    account.apply_changes(rows, _snapshot(), assume_yes=True)

    assert apply_env["requests"] == []
    assert rows[0]["status"] == "ok"
    assert rows[0]["stop_loss"] == 97.0  # reported from live data, not the stale snapshot
    assert rows[0]["recommended_stop_loss"] is None
    assert rows[1]["status"] == "not applied: position no longer open"
    assert json.loads(account.SNAPSHOT.read_text())["positions"][0]["stop_loss_rate"] == 97.0


@pytest.mark.unit
def test_apply_refuses_while_another_apply_runs(apply_env):
    held = account.acquire_lock(account.APPLY_LOCK)
    try:
        rows = _rows()
        account.apply_changes(rows, _snapshot(), assume_yes=True)
    finally:
        held.close()

    assert apply_env["requests"] == []
    assert all("another --apply run" in r["status"] for r in rows)


# --- confirmation -----------------------------------------------------------------

def _sent_row(**overrides):
    row = {
        "position_id": 1, "open_rate": 100.0, "status": "sent", "stop_loss": None, "take_profit": 150.0,
        "recommended_stop_loss": 90.0, "recommended_take_profit": None,
    }
    row.update(overrides)
    return row


@pytest.mark.unit
def test_reconcile_confirms_levels_etoro_now_reports():
    rows = [_sent_row()]
    assert account.reconcile(rows, {1: (90.0, 150.0)}) == []
    assert rows[0]["status"] == "set on eToro"
    assert rows[0]["stop_loss"] == 90.0


@pytest.mark.unit
def test_reconcile_keeps_waiting_while_old_level_shows():
    rows = [_sent_row()]
    assert account.reconcile(rows, {1: (0.0001, 150.0)}) == rows
    assert rows[0]["status"] == "sent"


@pytest.mark.unit
def test_reconcile_ignores_unsent_rows_and_flags_closed_positions():
    rows = [_sent_row(), _sent_row(position_id=2, status="ok")]
    assert account.reconcile(rows, {}) == []
    assert [r["status"] for r in rows] == ["position no longer open", "ok"]


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
    portfolio = {
        "ordersForOpen": [
            {"amount": 200.0, "mirrorID": 0},
            {"amount": 200.0},
            {"amount": 100.0, "mirrorID": 77},
        ],
        "orders": [{"amount": 150.0}],
    }
    assert sync.pending_orders_amount(portfolio) == 550.0


# --- live reads tolerate what eToro actually returns ---------------------------------

# A pending order's ordersForOpen entry has no rate/units; etoropy's models rejected
# the whole portfolio over it.
_PORTFOLIO_WITH_PENDING_ORDER = {
    "clientPortfolio": {
        "positions": [{
            "positionID": 1, "instrumentID": 1001, "isBuy": True, "leverage": 1, "openRate": 80.0,
            "stopLossRate": 0.0001, "takeProfitRate": 150.0, "isTslEnabled": False, "mirrorID": 0,
            "amount": 100.0, "units": 1.25, "openDateTime": "2026-01-02T15:00:00Z",
        }],
        "ordersForOpen": [{"instrumentID": 14350, "amount": 50.0, "mirrorID": 0, "openDateTime": "2026-09-14T11:33:20.587Z"}],
        "orders": [],
        "mirrors": [],
        "credit": 1196.68,
    }
}


@pytest.fixture
def etoro_http(monkeypatch):
    seen = []

    def handler(request):
        seen.append(request)
        if request.url.path == "/api/v1/trading/info/portfolio":
            return httpx.Response(200, json=_PORTFOLIO_WITH_PENDING_ORDER)
        if request.url.path == "/api/v1/market-data/instruments/rates":
            instrument_id = int(request.url.params["instrumentIds"])
            return httpx.Response(200, json={"rates": [{"instrumentID": instrument_id, "bid": 99.5, "ask": 99.7}]})
        return httpx.Response(404)

    monkeypatch.setenv("ETORO_MODE", "real")
    monkeypatch.setenv("ETORO_API_KEY", "api")
    monkeypatch.setenv("ETORO_USER_KEY", "user")
    monkeypatch.setattr(account, "make_client", lambda: httpx.Client(transport=httpx.MockTransport(handler)))
    return seen


@pytest.mark.unit
def test_fetch_live_reads_a_portfolio_with_a_pending_order(etoro_http):
    positions, bids = account.fetch_live()

    assert positions[1].stop_loss_rate == 0.0001
    assert positions[1].leverage == 1 and positions[1].mirror_id == 0 and positions[1].is_tsl_enabled is False
    assert bids == {1001: 99.5}
    assert account.live_levels() == {1: (0.0001, 150.0)}
    assert all(r.headers["x-user-key"] == "user" and r.headers["x-request-id"] for r in etoro_http)


@pytest.mark.unit
def test_positions_missing_safety_fields_are_skipped():
    position = account.parse_position({"positionID": 1, "instrumentID": 1001, "isBuy": True, "openRate": 80.0})
    stop, take_profit, why = account.plan_change(_row(), position, 100.0)
    assert (stop, take_profit) == (None, None)
    assert why


@pytest.mark.unit
def test_sync_counts_a_pending_order_as_reserved_cash():
    portfolio = _PORTFOLIO_WITH_PENDING_ORDER["clientPortfolio"]
    assert sync.pending_orders_amount(portfolio) == 50.0
    snapshot = sync.build_snapshot(portfolio, {}, {}, "real")
    assert snapshot["credit"] == 1196.68 and snapshot["pending_orders_amount"] == 50.0
    assert snapshot["positions"][0]["stop_loss_rate"] == 0.0001
