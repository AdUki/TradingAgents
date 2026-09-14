"""eToro -> Yahoo ticker mapping for scripts/sync_etoro_portfolio.py (no network)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("etoropy")

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "sync_etoro_portfolio.py"
_spec = importlib.util.spec_from_file_location("sync_etoro_portfolio", _SCRIPT)
sync = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sync)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("symbol", "type_name", "expected"),
    [
        ("AAPL", "Stocks", "AAPL"),
        ("SPY", "ETF", "SPY"),
        ("BRK.B", "Stocks", "BRK-B"),
        ("PBR.A", "Stocks", "PBR-A"),
        ("NESN.ZU", "Stocks", "NESN.SW"),
        ("ADYEN.NV", "Stocks", "ADYEN.AS"),
        ("BHP.ASX", "Stocks", "BHP.AX"),
        ("JD.US", "Stocks", "JD"),
        ("TSLA.EXT", "Stocks", "TSLA"),
        ("META.RTH", "Stocks", "META"),
        ("BARC.L", "Stocks", "BARC.L"),
        ("SIE.DE", "Stocks", "SIE.DE"),
        ("0939.HK", "Stocks", "0939.HK"),
        ("NOVO-B.CO", "Stocks", "NOVO-B.CO"),
        ("BTC", "Crypto", "BTC-USD"),
        ("BTCEUR", "Crypto", "BTC-USD"),
        ("ETH", "Cryptocurrencies", "ETH-USD"),
    ],
)
def test_maps_analyzable_instruments(symbol, type_name, expected):
    assert sync.to_yahoo_symbol(symbol, type_name) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("symbol", "type_name"),
    [
        ("EURUSD", "Currencies"),
        ("GOLD", "Commodities"),
        ("SPX500", "Indices"),
        ("IPO56.l", "Stocks"),
        ("XYZ.QQ", "Stocks"),
        ("ZYNE.CVR", "Stocks"),
    ],
)
def test_skips_what_yahoo_would_misread(symbol, type_name):
    assert sync.to_yahoo_symbol(symbol, type_name) is None


@pytest.mark.unit
def test_build_tickers_dedupes_and_reports_skips():
    positions = [SimpleNamespace(instrument_id=i) for i in (1001, 1001, 18, 999)]
    infos = {
        1001: SimpleNamespace(symbol_full="AAPL", instrument_type_id=5),
        18: SimpleNamespace(symbol_full="GOLD", instrument_type_id=2),
    }
    type_names = {5: "Stocks", 2: "Commodities"}

    tickers, skipped = sync.build_tickers(positions, infos, type_names)

    assert tickers == ["AAPL"]
    assert skipped == [
        "GOLD (Commodities): not analyzable",
        "instrument 999: no metadata from eToro",
    ]
