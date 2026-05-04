"""Tests for ResolveFinalUrl's wall-clock budget.

Repro: scrape 309 (ZipRecruiter /km/ tracker URL with auth_token +
expires) sat in `status='running'` for hours — trace ended at
DetectObstacle routed_to=ResolveFinalUrl (29ms) and the next node
never emitted. Without a budget the parent scrape never reaches a
terminal status and the poller keeps re-dispatching it.

The fix: wrap ResolveFinalUrl's sync body in
`asyncio.wait_for(asyncio.to_thread(...))` with a configurable
budget (default 15s). On timeout, best-effort PATCH the parent to
`failed` so the row escapes `running` and proceed to
CheckLinkDedup with whatever state was already set.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from scrape_graph.nodes_scrape import (
    CheckLinkDedup,
    ResolveFinalUrl,
)
from scrape_graph.state import ScrapeGraphState


def _run(node, state: ScrapeGraphState):
    ctx = SimpleNamespace(state=state)
    return asyncio.run(node.run(ctx))


def test_timeout_closes_parent_and_routes_to_check_link_dedup(monkeypatch):
    """When the sync body exceeds budget, ResolveFinalUrl PATCHes the
    parent to `failed` and still routes to CheckLinkDedup so the graph
    reaches a terminal node instead of wedging in `running`."""
    # Tiny budget so the test runs fast.
    monkeypatch.setattr(
        "scrape_graph.nodes_scrape._RESOLVE_FINAL_URL_BUDGET_S", 0.05,
    )

    state = ScrapeGraphState(
        scrape_id=309,
        submitted_url=(
            "https://www.ziprecruiter.com/km/AAGKsosUq...?"
            "auth_token=foo&expires=1778180646"
        ),
    )
    state.final_url = "https://www.ziprecruiter.com/jobs/r3O3dqevMC2SsDr-F6D80g"

    patches: list[dict] = []

    def fake_patch_status(scrape_id, status, note=None):
        patches.append({"scrape_id": scrape_id, "status": status, "note": note})

    def slow_body(state):
        # Simulate a wedged httpx.post — sleep past the budget. We can't
        # mock asyncio.to_thread away cleanly, so just block the worker
        # thread; wait_for will fire its timer.
        import time as _time
        _time.sleep(0.5)

    with patch(
        "scrape_graph.nodes_scrape._resolve_final_url_body",
        side_effect=slow_body,
    ), patch(
        "scrape_graph.nodes_scrape._patch_scrape_status",
        side_effect=fake_patch_status,
    ):
        nxt = _run(ResolveFinalUrl(), state)

    assert isinstance(nxt, CheckLinkDedup)
    assert len(patches) == 1, "parent must be terminal-closed exactly once"
    assert patches[0]["scrape_id"] == 309
    assert patches[0]["status"] == "failed"
    assert "timeout" in (patches[0]["note"] or "").lower()
    # canonical_url falls back to the submitted URL so CheckLinkDedup
    # has a non-empty key to filter on.
    assert state.canonical_url == state.submitted_url


def test_fast_body_does_not_close_parent(monkeypatch):
    """Happy path: when the body finishes well within budget, no
    timeout PATCH fires."""
    monkeypatch.setattr(
        "scrape_graph.nodes_scrape._RESOLVE_FINAL_URL_BUDGET_S", 5.0,
    )

    same = "https://example.com/jobs/42"
    state = ScrapeGraphState(scrape_id=400, submitted_url=same)
    state.final_url = same

    patches: list[dict] = []

    def fake_patch_status(scrape_id, status, note=None):
        patches.append({"scrape_id": scrape_id, "status": status, "note": note})

    with patch(
        "scrape_graph.nodes_scrape._patch_scrape_status",
        side_effect=fake_patch_status,
    ):
        nxt = _run(ResolveFinalUrl(), state)

    assert isinstance(nxt, CheckLinkDedup)
    assert patches == [], "no terminal PATCH on the happy path"
    assert state.did_redirect is False
