"""Tests for the ScrollToLoad node.

Covers the lazy-hydration fix: incrementally scroll until either the
profile's ready_selector matches OR the page stops growing. Drives a
fake browser page (no real Playwright) — the node only depends on
.evaluate() and .query_selector(), which we mock.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from scrape_graph.nodes_scrape import ExpandTruncations, ScrollToLoad
from scrape_graph.state import ScrapeGraphState


def _run(node, state: ScrapeGraphState):
    ctx = SimpleNamespace(state=state)
    return asyncio.run(node.run(ctx))


class _FakePage:
    """Minimal Playwright-page stand-in. Hydrates the ready_selector
    after `hydrate_after_ticks` scroll ticks, simulating
    IntersectionObserver-driven lazy load.
    """

    def __init__(self, *, ready_selector: str | None,
                 hydrate_after_ticks: int = 0,
                 final_height: int = 4000) -> None:
        self.ready_selector = ready_selector
        self.hydrate_after_ticks = hydrate_after_ticks
        self.final_height = final_height
        self.scroll_y = 0
        self.scrolls = 0
        self.height = 1000  # initial pre-hydration height

    async def evaluate(self, expr: str):
        if expr.startswith("window.scrollBy"):
            self.scrolls += 1
            self.scroll_y += 800
            # Page grows while content lazy-loads, then plateaus.
            self.height = min(self.final_height, self.height + 600)
            return None
        if expr == "document.body.scrollHeight":
            return self.height
        if expr == "window.scrollY":
            return self.scroll_y
        return None

    async def query_selector(self, selector: str):
        if (
            self.ready_selector
            and selector == self.ready_selector
            and self.scrolls >= self.hydrate_after_ticks
        ):
            return object()  # truthy handle
        return None


def _state_with_page(page: _FakePage, ready_selector: str | None = None) -> ScrapeGraphState:
    state = ScrapeGraphState(scrape_id=1, submitted_url="https://x.com/job/1")
    state.profile = {"ready_selector": ready_selector} if ready_selector else {}
    state._browser_page = page  # type: ignore[attr-defined]
    return state


def test_scrolltoload_stops_when_ready_selector_matches():
    page = _FakePage(
        ready_selector="[aria-label*='About the job' i]",
        hydrate_after_ticks=3,
    )
    state = _state_with_page(page, ready_selector="[aria-label*='About the job' i]")
    next_node = _run(ScrollToLoad(), state)
    assert isinstance(next_node, ExpandTruncations)
    # Loop exited as soon as the selector hydrated, not after the full budget.
    assert page.scrolls == 3
    last_payload = state.node_trace[-1].payload
    assert last_payload["matched_selector"] == "[aria-label*='About the job' i]"
    assert last_payload["ticks"] == 3


def test_scrolltoload_stops_on_scroll_height_stall():
    """No ready_selector to find — loop must exit on scrollHeight stall."""
    page = _FakePage(ready_selector=None, final_height=2000)
    state = _state_with_page(page)
    next_node = _run(ScrollToLoad(), state)
    assert isinstance(next_node, ExpandTruncations)
    # Height grows from 1000 → 1600 → 2000 → 2000 (stall) → 2000 (stall, exit).
    assert page.scrolls < 20  # well short of the budget
    last_payload = state.node_trace[-1].payload
    assert last_payload["matched_selector"] is None


def test_scrolltoload_no_page_is_noop():
    state = ScrapeGraphState(scrape_id=1, submitted_url="https://x.com/job/1")
    next_node = _run(ScrollToLoad(), state)
    assert isinstance(next_node, ExpandTruncations)
    last_payload = state.node_trace[-1].payload
    assert last_payload["ticks"] == 0
    assert last_payload["matched_selector"] is None
