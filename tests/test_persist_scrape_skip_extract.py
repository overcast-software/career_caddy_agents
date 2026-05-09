"""Tests for the PersistScrape branch on `state.skip_extract`.

When a Scrape is created with `skip_extract=True` (e.g. via the
staff "Resolve & dedupe" action on jp.edit), the scrape-graph runs:

    Navigate → DetectObstacle → ResolveFinalUrl → CheckLinkDedup →
    WaitReadySelector → SettleWait → ScrollToLoad → ExpandTruncations →
    Capture → PersistScrape → ResolveApplyUrl → End

…skipping the StartExtract → Tier* → PersistJobPost →
ReviewCompleteness → UpdateProfile chain. The originating JobPost's
fields are intentionally left untouched — the goal is redirect
resolution + apply-URL capture + duplicate detection (via
CheckLinkDedup → DuplicateShortCircuit), not field overwrite.

The branch is in PersistScrape.run: when `state.skip_extract` is
True, route to ResolveApplyUrl instead of StartExtract.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from scrape_graph.nodes_extract import ResolveApplyUrl, StartExtract
from scrape_graph.nodes_scrape import PersistScrape
from scrape_graph.state import ScrapeGraphState


def _run(node, state: ScrapeGraphState):
    ctx = SimpleNamespace(state=state)
    return asyncio.run(node.run(ctx))


def _state(skip_extract: bool) -> ScrapeGraphState:
    state = ScrapeGraphState(scrape_id=1, submitted_url="https://example.com/jobs/1")
    state.job_content = "(content)"
    state.html = "<html></html>"
    state.skip_extract = skip_extract
    return state


def test_skip_extract_routes_to_resolve_apply_url():
    """skip_extract=True → PersistScrape returns ResolveApplyUrl, not
    StartExtract. The extraction chain is bypassed entirely."""
    state = _state(skip_extract=True)

    # PATCH /scrapes/1/ is fire-and-forget here — stub it out so we can
    # exercise just the routing decision.
    with patch("scrape_graph.nodes_scrape.httpx.patch") as mock_patch:
        next_node = _run(PersistScrape(), state)

    assert isinstance(next_node, ResolveApplyUrl), (
        "skip_extract=True must route past extraction directly to apply-URL capture"
    )
    # PATCH still fires — the scrape row needs status='extracting' so
    # the graph-transition log is consistent. Only the routing changes.
    assert mock_patch.called, "scrape PATCH should still happen for status update"


def test_default_routes_to_start_extract():
    """skip_extract=False (the default) → PersistScrape returns
    StartExtract. Existing behavior unchanged."""
    state = _state(skip_extract=False)

    with patch("scrape_graph.nodes_scrape.httpx.patch"):
        next_node = _run(PersistScrape(), state)

    assert isinstance(next_node, StartExtract), (
        "default skip_extract=False must keep the existing extract chain"
    )


def test_state_default_is_false():
    """ScrapeGraphState.skip_extract defaults to False — backwards-
    compatible with every existing caller that doesn't set it."""
    state = ScrapeGraphState(scrape_id=1, submitted_url="https://example.com/jobs/1")
    assert state.skip_extract is False
