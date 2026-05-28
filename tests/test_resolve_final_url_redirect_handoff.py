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


def test_redirect_propagates_canonical_link_to_parent_jp():
    """When the parent scrape is attached to a JobPost, ResolveFinalUrl
    must PATCH that JobPost's canonical_link to the resolved
    post-redirect URL.

    Motivating incident: cc_auto created jp 3036 with link = SendGrid
    tracker wrapper. canonicalize_link couldn't decode the opaque
    `upn=` token, so jp 3036.canonical_link stayed pinned to the
    wrapper. A later scrape of the resolved destination
    `hiring.cafe/job/<id>` directly created a fresh jp 3040 — same
    opening, unmerged duplicate, because canonical_link comparison
    couldn't match wrapper vs destination.

    The fix propagates the resolved URL back to the parent JP. Future
    canonical_link lookups against the destination URL find the
    pre-existing JP and dedupe correctly.
    """
    submitted = "https://u52508838.ct.sendgrid.net/ls/click?upn=opaque"
    landed = "https://hiring.cafe/job/0qbs9smus2t06ecu"
    state = _state_for_redirect(submitted, landed)

    posts: list[dict] = []
    patches_status: list[dict] = []
    patches_jp: list[dict] = []
    gets: list[str] = []

    def fake_post(url, json=None, headers=None, timeout=None):
        posts.append({"url": url, "json": json})
        return _FakeResp(201, {"data": {"id": "478"}})

    def fake_patch_status(scrape_id, status, note=None):
        patches_status.append(
            {"scrape_id": scrape_id, "status": status, "note": note}
        )

    def fake_get(url, headers=None, timeout=None):
        gets.append(url)
        if "scrapes/302" in url:
            # Parent scrape is attached to jp 3036.
            return _FakeResp(200, {
                "data": {
                    "id": "302",
                    "type": "scrape",
                    "attributes": {},
                    "relationships": {
                        "job-post": {"data": {"type": "job-post", "id": "3036"}}
                    },
                }
            })
        if "job-posts/3036" in url:
            # JP 3036's current canonical_link is the SendGrid wrapper.
            return _FakeResp(200, {
                "data": {
                    "id": "3036",
                    "type": "job-post",
                    "attributes": {"canonical_link": submitted},
                }
            })
        return _FakeResp(404)

    def fake_patch(url, json=None, headers=None, timeout=None):
        patches_jp.append({"url": url, "json": json})
        return _FakeResp(200, {})

    with patch("scrape_graph.nodes_scrape.httpx.post", side_effect=fake_post), \
         patch("scrape_graph.nodes_scrape.httpx.get", side_effect=fake_get), \
         patch("scrape_graph.nodes_scrape.httpx.patch", side_effect=fake_patch), \
         patch(
             "scrape_graph.nodes_scrape._patch_scrape_status",
             side_effect=fake_patch_status,
         ):
        _run(ResolveFinalUrl(), state)

    # Child scrape was created on the resolved URL — handoff covered by
    # the other tests; just confirm the state.scrape_id swap landed.
    # (httpx.post is also invoked by trace_node for graph-transition
    # logging, so a raw `len(posts) == 1` here would be racy.)
    assert state.scrape_id == 478
    assert any(
        p["url"].endswith("/api/v1/scrapes/")
        and (p.get("json") or {}).get("data", {}).get("attributes", {}).get("source") == "redirect"
        for p in posts
    ), "child scrape create POST must fire exactly once"

    # The JobPost PATCH lands exactly once, on jp 3036, with the
    # resolved canonical URL — NOT the wrapper that submitted was.
    assert len(patches_jp) == 1, (
        "JobPost canonical_link PATCH must fire exactly once"
    )
    patched = patches_jp[0]
    assert "job-posts/3036" in patched["url"]
    assert patched["json"]["data"]["attributes"]["canonical_link"] == landed


def test_redirect_skips_jp_patch_when_canonical_already_resolved():
    """Idempotency: if a previous resolution already wrote the canonical
    URL onto the parent JP, the helper must not PATCH again. Prevents
    write-amplification when the same wrapped URL is scraped repeatedly
    (e.g. a poller retry, or a JobPost the user re-scrapes manually)."""
    landed = "https://hiring.cafe/job/abc"
    state = _state_for_redirect(
        "https://tracker.example.test/wrap?dest=abc", landed,
    )

    patches_jp: list[dict] = []

    def fake_post(url, json=None, headers=None, timeout=None):
        return _FakeResp(201, {"data": {"id": "999"}})

    def fake_get(url, headers=None, timeout=None):
        if "scrapes/302" in url:
            return _FakeResp(200, {
                "data": {
                    "id": "302", "type": "scrape", "attributes": {},
                    "relationships": {
                        "job-post": {"data": {"type": "job-post", "id": "42"}}
                    },
                }
            })
        if "job-posts/42" in url:
            # Already at the resolved URL — no PATCH needed.
            return _FakeResp(200, {
                "data": {
                    "id": "42", "type": "job-post",
                    "attributes": {"canonical_link": landed},
                }
            })
        return _FakeResp(404)

    def fake_patch(url, json=None, headers=None, timeout=None):
        patches_jp.append({"url": url, "json": json})
        return _FakeResp(200, {})

    with patch("scrape_graph.nodes_scrape.httpx.post", side_effect=fake_post), \
         patch("scrape_graph.nodes_scrape.httpx.get", side_effect=fake_get), \
         patch("scrape_graph.nodes_scrape.httpx.patch", side_effect=fake_patch), \
         patch("scrape_graph.nodes_scrape._patch_scrape_status"):
        _run(ResolveFinalUrl(), state)

    assert patches_jp == [], (
        "PATCH must not fire when canonical_link is already current"
    )


def test_redirect_no_jp_attached_skips_propagation():
    """When the parent scrape isn't attached to any JobPost (e.g. an
    ad-hoc URL scrape with no `job_post` relationship), the propagation
    helper has nowhere to write — must no-op cleanly."""
    state = _state_for_redirect(
        "https://tracker.example.test/wrap", "https://dest.example.test/job/1",
    )

    patches_jp: list[dict] = []

    def fake_post(url, json=None, headers=None, timeout=None):
        return _FakeResp(201, {"data": {"id": "999"}})

    def fake_get(url, headers=None, timeout=None):
        if "scrapes/302" in url:
            return _FakeResp(200, {
                "data": {
                    "id": "302", "type": "scrape", "attributes": {},
                    "relationships": {"job-post": {"data": None}},
                }
            })
        return _FakeResp(404)

    def fake_patch(url, json=None, headers=None, timeout=None):
        patches_jp.append({"url": url, "json": json})
        return _FakeResp(200, {})

    with patch("scrape_graph.nodes_scrape.httpx.post", side_effect=fake_post), \
         patch("scrape_graph.nodes_scrape.httpx.get", side_effect=fake_get), \
         patch("scrape_graph.nodes_scrape.httpx.patch", side_effect=fake_patch), \
         patch("scrape_graph.nodes_scrape._patch_scrape_status"):
        _run(ResolveFinalUrl(), state)

    assert patches_jp == [], (
        "PATCH must not fire when parent scrape has no job_post relationship"
    )
