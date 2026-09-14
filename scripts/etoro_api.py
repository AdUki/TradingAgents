"""Minimal read access to the eToro Public API for the eToro helper scripts.

Portfolios and prices are read as plain JSON rather than through etoropy's
models: those reject a whole portfolio when a pending order lacks fields
(ordersForOpen entries without rate/units), which would stop the portfolio
sync and the stop-loss checks whenever you have an order open.

Needs ETORO_API_KEY, ETORO_USER_KEY and ETORO_MODE in the environment.
"""

from __future__ import annotations

import os
import uuid

import httpx

BASE_URL = "https://public-api.etoro.com/api/v1"


def request_headers() -> dict[str, str]:
    return {
        "x-api-key": os.environ["ETORO_API_KEY"],
        "x-user-key": os.environ["ETORO_USER_KEY"],
        "x-request-id": str(uuid.uuid4()),
    }


def get_json(client: httpx.Client, path: str, **params) -> dict:
    response = client.get(f"{BASE_URL}{path}", params=params or None, headers=request_headers(), timeout=30)
    response.raise_for_status()
    return response.json()


def get_portfolio(client: httpx.Client) -> dict:
    """The raw ``clientPortfolio`` object for the account in ETORO_MODE."""
    path = "/trading/info/portfolio" if os.environ.get("ETORO_MODE") == "real" else "/trading/info/demo/portfolio"
    return get_json(client, path)["clientPortfolio"]


def get_bid(client: httpx.Client, instrument_id: int) -> float | None:
    """Live bid for one instrument (the API rejects several IDs in one request)."""
    rates = get_json(client, "/market-data/instruments/rates", instrumentIds=instrument_id).get("rates") or []
    return next((rate["bid"] for rate in rates if rate.get("instrumentID") == instrument_id), None)
