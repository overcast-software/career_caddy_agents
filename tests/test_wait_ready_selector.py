"""Tests for WaitReadySelector iteration.

Covers the post-2026-05-04 fix: instead of feeding a comma-joined
selector string into `page.wait_for_selector`, the node iterates over
a list of candidates and tries each via
`page.locator(s).first.wait_for(state="visible")`. The iteration is
also tolerant of mixed input types (str legacy, list new).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from scrape_graph.nodes_scrape import (
    ScrollToLoad,
    SettleWait,
    WaitReadySelector,
)
from scrape_graph.state import ScrapeGraphState


def _run(node, state: ScrapeGraphState):
    ctx = SimpleNamespace(state=state)
    return asyncio.run(node.run(ctx))


class _FakeLocator:
    def __init__(self, *, matches: bool, raises: Exception | None = None) -> None:
        self._matches = matches
        self._raises = raises

    @property
    def first(self) -> "_FakeLocator":
        return self

    async def wait_for(self, *, state: str = "visible", timeout: float = 1500) -> None:
        if self._raises is not None:
            raise self._raises
        if self._matches:
            return None
        # Match Playwright's TimeoutError shape — class name carries the
        # signal, message ignored.
        raise _FakeTimeout(f"Timeout {timeout}ms exceeded.")


class _FakeTimeout(Exception):
    """Mimics playwright._impl._errors.TimeoutError so the node's
    branch on class-name picks `error: timeout` over `parse:...`."""
    pass


_FakeTimeout.__name__ = "TimeoutError"


class _FakePage:
    """Minimal page that returns a per-selector locator stub."""

    def __init__(self, mapping: dict[str, _FakeLocator]) -> None:
        self._mapping = mapping
        self.calls: list[str] = []

    def locator(self, selector: str) -> _FakeLocator:
        self.calls.append(selector)
        loc = self._mapping.get(selector)
        if loc is None:
            return _FakeLocator(matches=False)
        return loc


def _state_with(page: _FakePage | None, ready_selector) -> ScrapeGraphState:
    state = ScrapeGraphState(scrape_id=1, submitted_url="https://x.com/job/1")
    state.profile = {"ready_selector": ready_selector} if ready_selector else {}
    if page is not None:
        state._browser_page = page  # type: ignore[attr-defined]
    return state


def test_wait_ready_returns_first_match_and_stops():
    page = _FakePage({
        # First two entries miss; third matches.
        ".jobs-description__container": _FakeLocator(matches=False),
        "h2:has-text(\"About this job\")": _FakeLocator(matches=False),
        "h2:has-text(\"About the job\")": _FakeLocator(matches=True),
    })
    selectors = [
        ".jobs-description__container",
        "h2:has-text(\"About this job\")",
        "h2:has-text(\"About the job\")",
        ".never-tried",  # must NOT be tried — loop exits on first hit
    ]
    state = _state_with(page, selectors)
    next_node = _run(WaitReadySelector(), state)
    assert isinstance(next_node, ScrollToLoad)
    payload = state.node_trace[-1].payload
    assert payload["matched_selector"] == "h2:has-text(\"About the job\")"
    assert payload["matched_index"] == 2
    assert payload["timed_out"] is False
    assert payload["selector_count"] == 4
    # Three attempts, no fourth — early exit on match.
    assert len(payload["attempts"]) == 3
    assert page.calls == [
        ".jobs-description__container",
        "h2:has-text(\"About this job\")",
        "h2:has-text(\"About the job\")",
    ]


def test_wait_ready_all_miss_routes_to_settle_wait():
    page = _FakePage({})  # no entry matches
    selectors = ["a", "b", "c"]
    state = _state_with(page, selectors)
    next_node = _run(WaitReadySelector(), state)
    assert isinstance(next_node, SettleWait)
    payload = state.node_trace[-1].payload
    assert payload["matched_selector"] is None
    assert payload["timed_out"] is True
    assert len(payload["attempts"]) == 3
    assert all(a["matched"] is False for a in payload["attempts"])
    assert all(a["error"] == "timeout" for a in payload["attempts"])


def test_wait_ready_normalizes_legacy_string_input():
    """Profiles authored before this PR stored ready_selector as a
    comma-joined string. The node must split-and-iterate, not feed the
    whole string into Playwright (the original bug)."""
    legacy = "h2:has-text(\"About the job\"), .jobs-description__container"
    page = _FakePage({
        "h2:has-text(\"About the job\")": _FakeLocator(matches=True),
    })
    state = _state_with(page, legacy)
    next_node = _run(WaitReadySelector(), state)
    assert isinstance(next_node, ScrollToLoad)
    payload = state.node_trace[-1].payload
    assert payload["selector_count"] == 2
    assert payload["matched_index"] == 0
    # Comma inside a `:has-text("About, the job")` would NOT split — the
    # quote-aware splitter keeps it intact. Cover that here too.


def test_wait_ready_split_respects_parens_and_quotes():
    legacy = 'h2:has-text("About, the job"), .container'
    page = _FakePage({
        '.container': _FakeLocator(matches=True),
    })
    state = _state_with(page, legacy)
    _run(WaitReadySelector(), state)
    payload = state.node_trace[-1].payload
    # The comma inside the quoted has-text must NOT split.
    assert payload["selector_count"] == 2
    # First entry tried is the has-text one; it misses, then the
    # container matches.
    attempts = payload["attempts"]
    assert attempts[0]["selector"] == 'h2:has-text("About, the job")'
    assert attempts[1]["selector"] == ".container"


def test_wait_ready_no_page_no_selectors_routes_to_settle_wait():
    state = _state_with(None, None)
    next_node = _run(WaitReadySelector(), state)
    assert isinstance(next_node, SettleWait)
    payload = state.node_trace[-1].payload
    assert payload["selector_count"] == 0
    assert payload["timed_out"] is False
    assert payload["attempts"] == []


def test_wait_ready_records_parse_error_distinctly_from_timeout():
    """A bad selector raises a non-TimeoutError exception. The trace
    must surface that as `parse:` so we can distinguish flake from
    user error."""
    bad = "::::malformed"
    page = _FakePage({
        bad: _FakeLocator(
            matches=False,
            raises=ValueError("Unknown engine \"::::\" while parsing selector"),
        ),
        ".good": _FakeLocator(matches=True),
    })
    state = _state_with(page, [bad, ".good"])
    _run(WaitReadySelector(), state)
    payload = state.node_trace[-1].payload
    assert payload["matched_selector"] == ".good"
    assert payload["attempts"][0]["error"].startswith("parse:")
