"""Tests for ResolveFinalUrl's parent-scrape terminal-close on redirect.

Repro: scrape 302 (LinkedIn /uas/login redirect URL) ran twice, both
producing JobPost 1647 successfully via DuplicateShortCircuit on the
child scrape — but scrape 302 itself stayed at status='running'
forever because once ResolveFinalUrl swapped state.scrape_id to the
child, no terminal PATCH ever landed on the parent. The poller kept
re-dispatching it.

The fix: when ResolveFinalUrl creates a child scrape and swaps to it,
it must first PATCH the parent to a terminal status with a note
pointing at the child.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from scrape_graph.nodes_scrape import ResolveFinalUrl, CheckLinkDedup
from scrape_graph.state import ScrapeGraphState


def _run(node, state: ScrapeGraphState):
    ctx = SimpleNamespace(state=state)
    return asyncio.run(node.run(ctx))


def _state_for_redirect(submitted: str, landed: str) -> ScrapeGraphState:
    state = ScrapeGraphState(scrape_id=302, submitted_url=submitted)
    state.final_url = landed
    return state


class _FakeResp:
    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = ""

    def json(self):
        return self._payload


def test_redirect_terminal_closes_parent_then_swaps_id():
    """When the browser landed at a different URL, ResolveFinalUrl
    must terminal-close the parent scrape with note pointing at the
    new child id, then swap state.scrape_id."""
    state = _state_for_redirect(
        "https://www.linkedin.com/uas/login?session_redirect=%2Fjobs%2Fview%2F4385027587%2F",
        "https://www.linkedin.com/jobs/view/4385027587/",
    )

    patches: list[dict] = []

    def fake_post(url, json=None, headers=None, timeout=None):
        return _FakeResp(201, {"data": {"id": "303"}})

    def fake_patch_status(scrape_id, status, note=None):
        patches.append({"scrape_id": scrape_id, "status": status, "note": note})

    with patch("scrape_graph.nodes_scrape.httpx.post", side_effect=fake_post), \
         patch("scrape_graph.nodes_scrape._patch_scrape_status", side_effect=fake_patch_status):
        next_node = _run(ResolveFinalUrl(), state)

    assert isinstance(next_node, CheckLinkDedup)
    # Parent (302) was terminal-closed exactly once before the id swap.
    assert len(patches) == 1
    closed = patches[0]
    assert closed["scrape_id"] == 302, "parent scrape must be closed before swap"
    assert closed["status"] == "completed"
    assert "303" in (closed["note"] or ""), "note must reference the child scrape id"
    # state.scrape_id was swapped to the child.
    assert state.scrape_id == 303
    assert state.did_redirect is True


def test_no_redirect_no_parent_close():
    """When submitted == landed (no redirect), no child is created and
    no terminal PATCH on the parent should fire — graph keeps going
    against the original scrape."""
    same = "https://example.com/jobs/42"
    state = _state_for_redirect(same, same)

    patches: list[dict] = []

    def fake_patch_status(scrape_id, status, note=None):
        patches.append({"scrape_id": scrape_id, "status": status, "note": note})

    with patch("scrape_graph.nodes_scrape._patch_scrape_status", side_effect=fake_patch_status):
        next_node = _run(ResolveFinalUrl(), state)

    assert isinstance(next_node, CheckLinkDedup)
    assert patches == [], "no terminal PATCH when submitted URL == landed URL"
    assert state.scrape_id == 302
    assert state.did_redirect is False


def test_child_create_failure_does_not_close_parent():
    """If the child-scrape POST returns non-2xx (e.g. 5xx), don't
    terminal-close the parent — there's no child to redirect to, the
    parent should stay live so the poller can retry."""
    state = _state_for_redirect(
        "https://x.com/login?redirect=%2Fjob%2F1",
        "https://x.com/job/1",
    )

    patches: list[dict] = []

    def fake_post(url, json=None, headers=None, timeout=None):
        return _FakeResp(503)

    def fake_patch_status(scrape_id, status, note=None):
        patches.append({"scrape_id": scrape_id, "status": status, "note": note})

    with patch("scrape_graph.nodes_scrape.httpx.post", side_effect=fake_post), \
         patch("scrape_graph.nodes_scrape._patch_scrape_status", side_effect=fake_patch_status):
        _run(ResolveFinalUrl(), state)

    assert patches == [], "no parent close when child POST failed"
    assert state.scrape_id == 302
