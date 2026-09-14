"""Tests for the RSS-first Reddit fetcher, its single combined search request,
its 429 backoff, the opt-in JSON path's degradation (#862), and chunked-transfer
error handling (#1024)."""

from __future__ import annotations

import http.client
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from tradingagents.dataflows import reddit

_SAMPLE_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <category term="stocks" label="r/stocks"/>
    <title>NVDA earnings beat, stock pops</title>
    <published>2026-05-20T14:30:00+00:00</published>
    <content type="html">&lt;!-- SC_OFF --&gt;&lt;div class="md"&gt;&lt;p&gt;Great &lt;b&gt;quarter&lt;/b&gt; for NVDA&amp;#39;s datacenter unit.&lt;/p&gt;&lt;/div&gt;&lt;!-- SC_ON --&gt;</content>
  </entry>
  <entry>
    <title>Is NVDA overvalued?</title>
    <published>2026-05-19T09:00:00Z</published>
    <content type="html">&lt;p&gt;Forward P/E discussion&lt;/p&gt;</content>
  </entry>
</feed>
"""


def _resp(read_fn):
    """A minimal context-manager response whose read() runs ``read_fn``."""
    class _Resp:
        def __enter__(self_inner):
            return self_inner

        def __exit__(self_inner, *a):
            return False

        def read(self_inner, size=-1):
            data = read_fn()
            return data if size is None or size < 0 else data[:size]
    return _Resp()


def _atom_resp():
    return _resp(lambda: _SAMPLE_ATOM.encode("utf-8"))


def _raise(exc):
    def _r():
        raise exc
    return _resp(_r)


def _post(sub, title="NVDA pops", created="2026-05-20T14:30:00Z", **overrides):
    post = {
        "title": title, "score": None, "num_comments": None,
        "created_utc": reddit._iso_to_timestamp(created),
        "selftext": "", "source": "rss", "subreddit": sub,
    }
    post.update(overrides)
    return post


@pytest.mark.unit
class TestIsoToTimestamp:
    def test_parses_offset_and_z(self):
        assert reddit._iso_to_timestamp("2026-05-20T14:30:00+00:00") > 0
        assert reddit._iso_to_timestamp("2026-05-19T09:00:00Z") > 0

    def test_none_and_garbage_return_none(self):
        assert reddit._iso_to_timestamp(None) is None
        assert reddit._iso_to_timestamp("not-a-date") is None


@pytest.mark.unit
class TestStripHtml:
    def test_extracts_between_sc_markers_and_unescapes(self):
        raw = "<!-- SC_OFF --><div class=\"md\"><p>Great <b>quarter</b> &amp; more</p></div><!-- SC_ON -->"
        assert reddit._strip_html(raw) == "Great quarter & more"

    def test_empty(self):
        assert reddit._strip_html("") == ""


@pytest.mark.unit
class TestRssParsing:
    def test_parses_atom_entries(self):
        with patch.object(reddit, "urlopen", return_value=_atom_resp()):
            posts = reddit._fetch_subreddit_rss("NVDA", "stocks", limit=5, timeout=5.0)
        assert len(posts) == 2
        assert posts[0]["title"] == "NVDA earnings beat, stock pops"
        assert posts[0]["source"] == "rss"
        assert posts[0]["score"] is None
        assert posts[0]["num_comments"] is None
        assert posts[0]["created_utc"] > 0
        assert "datacenter unit" in posts[0]["selftext"]

    def test_tags_posts_with_category_subreddit_or_requested_sub(self):
        with patch.object(reddit, "urlopen", return_value=_atom_resp()):
            posts = reddit._fetch_subreddit_rss("NVDA", "wallstreetbets+stocks", 5, 5.0)
        assert posts[0]["subreddit"] == "stocks"
        assert posts[1]["subreddit"] == "wallstreetbets+stocks"

    def test_malformed_xml_reports_unavailable(self):
        with patch.object(reddit, "urlopen", return_value=_resp(lambda: b"<<not xml>>")):
            assert reddit._fetch_subreddit_rss("NVDA", "stocks", 5, 5.0) is None


@pytest.mark.unit
class TestFetchSubredditIsRssFirst:
    """The default per-subreddit fetch goes straight to RSS — it must not hit
    the WAF-blocked JSON endpoint, which only burned rate-limit budget."""

    def test_delegates_to_rss_without_touching_json(self):
        sentinel = [_post("stocks", title="x")]
        with patch.object(reddit, "_fetch_subreddit_rss", return_value=sentinel) as rss, \
             patch.object(reddit, "urlopen",
                          side_effect=AssertionError("JSON endpoint must not be called")):
            out = reddit._fetch_subreddit("NVDA", "stocks", 5, 5.0)
        rss.assert_called_once()
        assert out is sentinel


@pytest.mark.unit
class TestJsonPathFallsBackToRss:
    """The opt-in JSON path still degrades to RSS on a 403 (kept for #862)."""

    def test_403_triggers_rss(self):
        err = HTTPError("url", 403, "Blocked", {}, None)
        rss_posts = [_post("stocks", title="x")]
        with patch.object(reddit, "urlopen", side_effect=err), \
             patch.object(reddit, "_fetch_subreddit_rss", return_value=rss_posts) as rss:
            out = reddit._fetch_subreddit_json("NVDA", "stocks", 5, 5.0)
        rss.assert_called_once()
        assert out and out[0]["source"] == "rss"


@pytest.mark.unit
class TestRss429Backoff:
    def test_429_then_success_retries_once(self):
        err = HTTPError("url", 429, "Too Many Requests", {}, None)
        with patch.object(reddit, "urlopen", side_effect=[err, _atom_resp()]) as op, \
             patch.object(reddit.time, "sleep") as slept:
            posts = reddit._fetch_subreddit_rss("NVDA", "stocks", 5, 5.0)
        assert op.call_count == 2          # original + exactly one retry
        slept.assert_called_once()         # backed off before retrying
        assert len(posts) == 2

    def test_429_twice_gives_up_after_one_retry(self):
        err = HTTPError("url", 429, "Too Many Requests", {}, None)
        with patch.object(reddit, "urlopen", side_effect=[err, err]) as op, \
             patch.object(reddit.time, "sleep"):
            posts = reddit._fetch_subreddit_rss("NVDA", "stocks", 5, 5.0)
        assert op.call_count == 2          # one retry, then gives up cleanly
        assert posts is None

    def test_retry_after_header_is_honoured(self):
        err = HTTPError("url", 429, "Too Many Requests", {"Retry-After": "12"}, None)
        with patch.object(reddit, "urlopen", side_effect=[err, _atom_resp()]), \
             patch.object(reddit.time, "sleep") as slept:
            reddit._fetch_subreddit_rss("NVDA", "stocks", 5, 5.0)
        slept.assert_called_once_with(12.0)

    def test_retry_after_zero_is_honoured_not_treated_as_absent(self):
        # A valid "Retry-After: 0" means retry at once; it must not fall through
        # to the fallback wait (the earlier `or 5.0` bug turned 0 into 5s).
        err = HTTPError("url", 429, "Too Many Requests", {"Retry-After": "0"}, None)
        with patch.object(reddit, "urlopen", side_effect=[err, _atom_resp()]), \
             patch.object(reddit.time, "sleep") as slept:
            reddit._fetch_subreddit_rss("NVDA", "stocks", 5, 5.0)
        slept.assert_called_once_with(0.0)

    def test_headerless_429_fallback_is_jittered_never_below_60s(self):
        # No Retry-After -> our own 60s fallback, jittered upward only: a retry
        # sooner than 60s is measured to still 429, wasting the one retry.
        err = HTTPError("url", 429, "Too Many Requests", {}, None)
        for _ in range(50):
            with patch.object(reddit, "urlopen", side_effect=[err, _atom_resp()]), \
                 patch.object(reddit.time, "sleep") as slept:
                reddit._fetch_subreddit_rss("NVDA", "stocks", 5, 5.0)
            slept.assert_called_once()
            (wait,), _ = slept.call_args
            assert 60.0 <= wait <= 72.0


@pytest.mark.unit
class TestChunkedTransferErrorsHandled:
    """IncompleteRead/RemoteDisconnected come from http.client and are NOT
    OSErrors, so they were previously uncaught and crashed the pipeline (#1024)."""

    def test_rss_incomplete_read_reports_unavailable(self):
        with patch.object(reddit, "urlopen", return_value=_raise(http.client.IncompleteRead(b""))):
            assert reddit._fetch_subreddit_rss("NVDA", "stocks", 5, 5.0) is None

    def test_json_incomplete_read_falls_back_to_rss(self):
        with patch.object(reddit, "urlopen", return_value=_raise(http.client.IncompleteRead(b""))), \
             patch.object(reddit, "_fetch_subreddit_rss", return_value=[]) as rss:
            reddit._fetch_subreddit_json("NVDA", "stocks", 5, 5.0)
        rss.assert_called_once()

    def test_oversized_rss_feed_is_refused_not_parsed(self):
        # A hostile/misbehaving endpoint streaming an unbounded body must not be
        # read into memory before parsing; overflow degrades to an empty feed.
        big = _resp(lambda: b"x" * 100)
        with patch.object(reddit, "_MAX_FEED_BYTES", 10), \
             patch.object(reddit, "urlopen", return_value=big):
            assert reddit._fetch_subreddit_rss("NVDA", "stocks", 5, 5.0) is None


@pytest.mark.unit
class TestCombinedSearch:
    """All subreddits share one request: the RSS feed allows ~1 request/minute
    per IP, so a request per subreddit 429'd every subreddit after the first."""

    def test_one_request_covers_every_subreddit(self):
        calls = []

        def record(t, sub, limit, timeout):
            calls.append((sub, limit))
            return []

        with patch.object(reddit, "_fetch_subreddit", side_effect=record):
            reddit.fetch_reddit_posts("NVDA", subreddits=("a", "b", "c"))
        assert calls == [("a+b+c", reddit._COMBINED_SEARCH_LIMIT)]

    def test_posts_grouped_and_capped_per_subreddit(self):
        fetched = (
            [_post("stocks", title=f"stocks post {i}") for i in range(7)]
            + [_post("Investing", title="investing post")]
            + [_post("other", title="unrequested post")]
        )
        with patch.object(reddit, "_fetch_subreddit", return_value=fetched):
            out = reddit.fetch_reddit_posts(
                "NVDA", subreddits=("stocks", "investing"), limit_per_sub=5
            )
        assert "r/stocks — 5 recent posts" in out
        assert "stocks post 4" in out and "stocks post 5" not in out
        assert "r/investing — 1 recent posts" in out
        assert "unrequested post" not in out

    def test_window_applied_before_per_subreddit_cap(self):
        fetched = [
            _post("stocks", title="too new 1", created="2026-05-25T12:00:00Z"),
            _post("stocks", title="too new 2", created="2026-05-25T11:00:00Z"),
            _post("stocks", title="in window 1", created="2026-05-16T12:00:00Z"),
            _post("stocks", title="in window 2", created="2026-05-15T12:00:00Z"),
        ]
        with patch.object(reddit, "_fetch_subreddit", return_value=fetched):
            out = reddit.fetch_reddit_posts(
                "NVDA", subreddits=("stocks",), limit_per_sub=2,
                start_date="2026-05-13", end_date="2026-05-20",
            )
        assert "in window 1" in out and "in window 2" in out
        assert "too new" not in out


@pytest.mark.unit
class TestFormatterHandlesRssPosts:
    def test_rss_posts_omit_fake_counts_and_note_source(self):
        rss_posts = [_post("stocks", selftext="great quarter")]
        with patch.object(reddit, "_fetch_subreddit", return_value=rss_posts):
            out = reddit.fetch_reddit_posts("NVDA", subreddits=("stocks",))
        assert "via RSS feed" in out
        assert "↑" not in out  # no fake score arrow
        assert "NVDA pops" in out
        assert "great quarter" in out

    def test_json_posts_still_show_counts(self):
        json_posts = [_post("stocks", score=1234, num_comments=56, source=None)]
        with patch.object(reddit, "_fetch_subreddit", return_value=json_posts):
            out = reddit.fetch_reddit_posts("NVDA", subreddits=("stocks",))
        assert "1234↑" in out
        assert "56c" in out
        assert "via RSS" not in out


@pytest.mark.unit
class TestCryptoSearchTerm:
    """A crypto pair (BTC-USD) barely matches Reddit text; search the base (#1113)."""

    def _captured_ticker(self, ticker):
        seen = {}

        def fake_fetch(t, sub, limit, timeout, **kwargs):
            seen["ticker"] = t
            return []

        with patch.object(reddit, "_fetch_subreddit", side_effect=fake_fetch):
            reddit.fetch_reddit_posts(ticker, subreddits=("stocks",))
        return seen["ticker"]

    def test_crypto_pair_searches_base(self):
        assert self._captured_ticker("BTC-USD") == "BTC"

    def test_equity_passes_through(self):
        assert self._captured_ticker("NVDA") == "NVDA"


@pytest.mark.unit
class TestFailedFetchIsNotSilence:
    """A throttled fetch must not be rendered as "no posts found" (#1295).

    Returning [] for both a failed request and a genuinely empty search made the
    sentiment analyst read rate limiting as real silence ("r/stocks and
    r/investing are silent"), which is a signal that was never observed.
    """

    def _run(self, fetched):
        with patch.object(reddit, "_fetch_subreddit", return_value=fetched):
            return reddit.fetch_reddit_posts("NVDA", subreddits=("s0", "s1"))

    def test_failed_fetch_does_not_claim_no_posts(self):
        out = self._run(None)
        assert "Reddit unavailable" in out
        assert "r/s0" in out and "r/s1" in out
        assert "no posts found" not in out
        assert "no Reddit posts found" not in out

    def test_genuine_empty_still_reports_no_posts(self):
        out = self._run([])
        assert "no Reddit posts found" in out
        assert "unavailable" not in out

    def test_quiet_subreddit_reported_empty_beside_active_one(self):
        out = self._run([_post("s0")])
        assert "NVDA pops" in out
        assert "r/s1: <no posts found" in out
        assert "Reddit unavailable" not in out
