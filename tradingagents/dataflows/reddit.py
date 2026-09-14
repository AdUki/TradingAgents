"""Reddit search fetcher for ticker-specific discussion posts.

Default path is Reddit's public Atom/RSS search feed. All subreddits are
searched with one combined request (``reddit.com/r/a+b+c/search.rss``) and the
entries are grouped back per subreddit by their Atom ``<category>``: the feed
allows roughly one request per minute per IP, so a request per subreddit
guaranteed a 429 (and a ~60s back-off) on every subreddit after the first. The
richer JSON search endpoint (``/search.json``) is reliably WAF-blocked
(``HTTP 403``) for public clients (issue #862), so it is kept
(``_fetch_subreddit_json``) but not used by default. On a 429 we back off once
(honouring ``Retry-After``). RSS lacks score / comment counts, so those posts are
marked and the formatter omits the metrics rather than printing fake zeros.

A fetch that fails is reported as ``<unavailable>``, never as "no posts found":
the two are different claims, and passing a rate-limited fetch off as silence
hands the sentiment analyst a signal that was never observed (#1295).

No API key required. Returns formatted plaintext blocks ready for prompt
injection and degrades gracefully — returns a placeholder string rather than
raising, so callers never special-case missing data.
"""

from __future__ import annotations

import html
import http.client
import json
import logging
import random
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from datetime import datetime, timezone
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .date_window import in_window
from .symbol_utils import crypto_base

logger = logging.getLogger(__name__)


def _within_window(posts, start_date, end_date):
    """Keep only posts published in [start_date, end_date] (look-ahead safe).

    No window (both None) leaves the list untouched for live callers. A post with
    no ``created_utc`` epoch is dropped in a historical window (#1220).
    """
    if not (start_date and end_date):
        return posts
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    kept = []
    for p in posts:
        ts = p.get("created_utc")
        created = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None
        if in_window(created, start_dt, end_dt):
            kept.append(p)
    return kept

_API = "https://www.reddit.com/r/{sub}/search.json?{qs}"
_RSS = "https://www.reddit.com/r/{sub}/search.rss?{qs}"
# A descriptive, identified User-Agent (per Reddit's API etiquette). Reddit
# blocks generic/anonymous tokens like bare "Mozilla/5.0" or "curl/…" but
# serves this one on both endpoints; the RSS feed accepts it even when the
# JSON search endpoint 403s, so no browser-spoofing is needed.
_UA = "tradingagents/0.2 (+https://github.com/TauricResearch/TradingAgents)"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

# Default subreddits ordered roughly by signal density for ticker-specific
# discussion. wallstreetbets has the most volume but most noise; stocks /
# investing trend more measured. Caller can override.
DEFAULT_SUBREDDITS = ("wallstreetbets", "stocks", "investing")

# Reddit's maximum page size; one combined search page is split across subreddits.
_COMBINED_SEARCH_LIMIT = 100


def _search_qs(ticker: str, limit: int) -> str:
    return urlencode({
        "q": ticker,
        "restrict_sr": "on",
        "sort": "new",
        "t": "week",  # last 7 days
        "limit": limit,
    })


def _iso_to_timestamp(iso_str: str | None) -> float | None:
    """Parse an Atom ``published`` timestamp to a UTC epoch, or None."""
    if not iso_str:
        return None
    try:
        normalized = iso_str[:-1] + "+00:00" if iso_str.endswith("Z") else iso_str
        return datetime.fromisoformat(normalized).timestamp()
    except (ValueError, TypeError):
        return None


def _strip_html(content: str) -> str:
    """Reduce the HTML body Reddit embeds in an Atom entry to plain text."""
    if not content:
        return ""
    # Reddit wraps the real selftext between SC_OFF / SC_ON markers.
    if "<!-- SC_OFF -->" in content and "<!-- SC_ON -->" in content:
        content = content.split("<!-- SC_OFF -->")[1].split("<!-- SC_ON -->")[0]
    text = re.sub(r"<[^>]+>", " ", content)
    return " ".join(html.unescape(text).split())


# Headerless-429 backoff when Reddit gives no Retry-After. Measured against
# /r/{sub}/search.rss, a retry still 429s at 8s, 10s and 30s of spacing and
# succeeds at 60s, so a shorter wait spends the one retry on a request that
# cannot succeed (#1295). Jittered upward only, so several analyses sharing an
# IP don't retry in lockstep and no retry fires before the measured 60s.
_RETRY_FALLBACK_SECONDS = 60.0


def _jitter(seconds: float, frac: float = 0.2) -> float:
    """Return ``seconds`` plus up to ``frac`` random jitter (never less), to
    desynchronize concurrent runs without undercutting a measured minimum."""
    return seconds * (1.0 + random.uniform(0.0, frac))


def _retry_after_seconds(exc: HTTPError) -> float | None:
    """Seconds to wait from a 429's ``Retry-After`` header, capped at 60s.

    The cap matches ``_RETRY_FALLBACK_SECONDS``: honouring less than we would
    wait on our own would spend the one retry on a request we already know is
    too early.

    Returns ``None`` only when the header is absent or unparseable; a valid
    ``Retry-After: 0`` returns ``0.0`` (retry at once), not ``None``.
    """
    try:
        val = exc.headers.get("Retry-After") if getattr(exc, "headers", None) else None
        return min(float(val), 60.0) if val is not None else None
    except (ValueError, TypeError, AttributeError):
        return None


# Reddit search feeds are small (a page of results); cap the read so a
# compromised or misbehaving endpoint can't stream an unbounded body into
# memory before we parse it. Overflow raises http.client.HTTPException, which
# both fetch paths already treat as a failed fetch (degrade to empty / RSS).
_MAX_FEED_BYTES = 5 * 1024 * 1024


def _read_capped(resp) -> bytes:
    """Read a response body bounded to ``_MAX_FEED_BYTES``, raising on overflow."""
    data = resp.read(_MAX_FEED_BYTES + 1)
    if len(data) > _MAX_FEED_BYTES:
        raise http.client.HTTPException(
            f"Reddit feed exceeded {_MAX_FEED_BYTES} bytes; refusing to parse"
        )
    return data


def _fetch_subreddit_rss(
    ticker: str,
    sub: str,
    limit: int,
    timeout: float,
    _retry: bool = True,
) -> list[dict] | None:
    """Default path: parse the public Atom search feed for a subreddit.

    ``sub`` may be a ``+``-joined multireddit; each post is tagged with the
    subreddit named in its Atom ``<category>``, falling back to ``sub``.

    Carries no score / comment counts, so those fields are left None and the
    post is tagged ``source="rss"`` for honest display. On a 429 (Reddit's
    per-IP rate limit) we back off once — honouring ``Retry-After`` when
    present — before giving up, so a transient burst doesn't blank the feed.

    Returns ``[]`` when the search ran and matched nothing, and ``None`` when
    the fetch itself failed. The caller must keep these apart: rendering a
    failed fetch as "no posts found" hands the sentiment analyst an absence of
    discussion that was never observed (#1295).
    """
    url = _RSS.format(sub=sub, qs=_search_qs(ticker, limit))
    req = Request(url, headers={"User-Agent": _UA})
    try:
        with urlopen(req, timeout=timeout) as resp:
            root = ET.fromstring(_read_capped(resp))
    except HTTPError as exc:
        if exc.code == 429 and _retry:
            # Honour a server-supplied Retry-After exactly (including 0); jitter
            # only our own fallback so concurrent runs don't retry in lockstep.
            retry_after = _retry_after_seconds(exc)
            wait = retry_after if retry_after is not None else _jitter(_RETRY_FALLBACK_SECONDS)
            logger.warning(
                "Reddit RSS 429 for r/%s · %s — backing off %.1fs then retrying once",
                sub, ticker, wait,
            )
            time.sleep(wait)
            return _fetch_subreddit_rss(ticker, sub, limit, timeout, _retry=False)
        logger.warning("Reddit RSS fetch failed for r/%s · %s: %s", sub, ticker, exc)
        return None
    except (OSError, http.client.HTTPException, ET.ParseError) as exc:
        # OSError covers URLError/TimeoutError/connection resets; HTTPException
        # covers chunked-transfer errors (IncompleteRead/BadStatusLine, #1024).
        logger.warning("Reddit RSS fetch failed for r/%s · %s: %s", sub, ticker, exc)
        return None

    posts = []
    for entry in root.findall("atom:entry", _ATOM_NS)[:limit]:
        title_el = entry.find("atom:title", _ATOM_NS)
        published_el = entry.find("atom:published", _ATOM_NS)
        content_el = entry.find("atom:content", _ATOM_NS)
        category_el = entry.find("atom:category", _ATOM_NS)
        posts.append({
            "title": (title_el.text if title_el is not None else "") or "",
            "score": None,
            "num_comments": None,
            "created_utc": _iso_to_timestamp(
                published_el.text if published_el is not None else None
            ),
            "selftext": _strip_html(content_el.text if content_el is not None else ""),
            "source": "rss",
            "subreddit": (category_el.get("term") if category_el is not None else None) or sub,
        })
    return posts


def _fetch_subreddit_json(
    ticker: str,
    sub: str,
    limit: int,
    timeout: float,
) -> list[dict]:
    """Richer JSON search path (carries score / comment counts).

    Reddit's WAF currently returns ``403 Blocked`` on this endpoint for
    non-OAuth clients (issue #862), so it is NOT used by default — calling it on
    every request only doubled our volume against the per-IP rate limit and
    triggered 429s on the RSS fallback. Kept for the day the WAF relaxes or an
    OAuth token is wired in; degrades to RSS on failure.
    """
    url = _API.format(sub=sub, qs=_search_qs(ticker, limit))
    req = Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            payload = json.loads(_read_capped(resp))
        children = (payload.get("data") or {}).get("children") or []
        return [c.get("data", {}) for c in children if isinstance(c, dict)]
    except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
        logger.warning(
            "Reddit JSON fetch failed for r/%s · %s: %s — falling back to RSS feed.",
            sub, ticker, exc,
        )
        return _fetch_subreddit_rss(ticker, sub, limit, timeout)


def _fetch_subreddit(
    ticker: str,
    sub: str,
    limit: int,
    timeout: float,
) -> list[dict] | None:
    """Fetch a subreddit (or ``+``-joined multireddit), RSS-first. ``None``
    means the fetch failed.

    The JSON search endpoint is reliably WAF-blocked (403) for public clients,
    so we go straight to the RSS feed, which serves our identified User-Agent.
    """
    return _fetch_subreddit_rss(ticker, sub, limit, timeout)


def fetch_reddit_posts(
    ticker: str,
    subreddits: Iterable[str] = DEFAULT_SUBREDDITS,
    limit_per_sub: int = 5,
    timeout: float = 10.0,
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    """Fetch recent Reddit posts mentioning ``ticker`` across finance
    subreddits and return them as a formatted plaintext block.

    All subreddits are searched with one combined request, since the RSS feed
    allows about one request per minute per IP. The shared page of results is
    split back per subreddit and capped at ``limit_per_sub`` each, so a very
    high-volume subreddit can crowd quieter ones out of that page.

    When ``start_date``/``end_date`` (yyyy-mm-dd) are given, posts are trimmed to
    that window so a historical run does not leak current discussion into a
    backtest (#1220).
    """
    # Crypto reaches us as a Yahoo pair (BTC-USD); search Reddit for the base
    # ("BTC") so the query actually matches discussion instead of near-nothing.
    ticker = crypto_base(ticker) or ticker
    subreddits = list(subreddits)
    fetched = _fetch_subreddit(ticker, "+".join(subreddits), _COMBINED_SEARCH_LIMIT, timeout)
    if fetched is None:
        # A failed fetch is not an absence of discussion, so it must not be
        # rendered as "no posts found" (#1295).
        return (
            f"<Reddit unavailable: fetch failed for "
            f"{', '.join(f'r/{s}' for s in subreddits)}; this is not an "
            f"absence of discussion>"
        )

    # Window before capping, so posts newer than a historical window don't use
    # up a subreddit's slots and push out the in-window ones.
    windowed = _within_window(fetched, start_date, end_date)
    blocks = []
    total_posts = 0
    for sub in subreddits:
        posts = [p for p in windowed if (p.get("subreddit") or "").lower() == sub.lower()]
        posts = posts[:limit_per_sub]
        total_posts += len(posts)
        if not posts:
            blocks.append(f"r/{sub}: <no posts found mentioning {ticker.upper()} in the past 7 days>")
            continue

        via_rss = any(p.get("source") == "rss" for p in posts)
        header = f"r/{sub} — {len(posts)} recent posts mentioning {ticker.upper()}"
        header += " (via RSS feed; scores/comments unavailable):" if via_rss else ":"
        lines = [header]
        for p in posts:
            title = (p.get("title") or "").replace("\n", " ").strip()
            score = p.get("score")
            comments = p.get("num_comments")
            created = p.get("created_utc")
            created_str = (
                time.strftime("%Y-%m-%d", time.gmtime(created)) if created else "?"
            )
            # Score / comment counts are absent on the RSS fallback path —
            # show them only when present rather than printing fake zeros.
            meta = created_str
            if score is not None and comments is not None:
                meta += f" · {score:>4}↑ · {comments:>3}c"
            selftext = (p.get("selftext") or "").replace("\n", " ").strip()
            if len(selftext) > 240:
                selftext = selftext[:240] + "…"
            lines.append(
                f"  [{meta}] {title}"
                + (f"\n    body excerpt: {selftext}" if selftext else "")
            )
        blocks.append("\n".join(lines))

    if total_posts == 0:
        return (
            f"<no Reddit posts found mentioning {ticker.upper()} across "
            f"{', '.join(f'r/{s}' for s in subreddits)} in the past 7 days>"
        )
    return "\n\n".join(blocks)
