"""CC-101 regression — CheckLinkDedup must read JobPost ids as NanoID strings.

CheckLinkDedup short-circuits to DuplicateShortCircuit when the canonical
URL already maps to a non-stub JobPost (description >= 60 words). It used to
``non_stub_id = int(row["id"])``; under CC-77 JobPost ids are 10-char NanoID
strings, so int() raised ValueError, the bare ``except`` swallowed it,
``non_stub_id`` stayed None, the short-circuit never fired, and a DUPLICATE
JobPost got created — a direct dedupe-first violation.

This test feeds a NanoID id + a non-stub description and asserts the
short-circuit fires with the string id carried onto state. It routes to
WaitReadySelector (no dedupe) against the pre-fix int() cast.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from scrape_graph.nodes_scrape import CheckLinkDedup, DuplicateShortCircuit, WaitReadySelector
from scrape_graph.state import ScrapeGraphState

NANOID_JP = "V1StGXR8_Z"
_NON_STUB_DESC = " ".join(f"word{i}" for i in range(80))  # >= 60 words


class _FakeResp:
    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = ""

    def json(self):
        return self._payload


def _run(node, state):
    return asyncio.run(node.run(SimpleNamespace(state=state)))


def test_dedup_short_circuit_carries_nanoid_string():
    state = ScrapeGraphState(scrape_id="a1fFQQe1xV", submitted_url="https://x.test/job/1")
    state.canonical_url = "https://x.test/job/1"

    def fake_get(url, params=None, headers=None, timeout=None):
        return _FakeResp(200, {
            "data": [
                {"id": NANOID_JP, "type": "job-post",
                 "attributes": {"description": _NON_STUB_DESC}},
            ]
        })

    with patch("scrape_graph.nodes_scrape.httpx.get", side_effect=fake_get), \
         patch("scrape_graph.nodes_scrape.trace_node"):
        next_node = _run(CheckLinkDedup(), state)

    assert isinstance(next_node, DuplicateShortCircuit)
    assert not isinstance(next_node, WaitReadySelector)
    # The NanoID string is carried onto state — no int() coercion.
    assert state.job_post_id == NANOID_JP
    assert isinstance(state.job_post_id, str)
    assert state.was_duplicate is True
