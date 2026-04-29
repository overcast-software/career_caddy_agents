"""Auto-trace every node transition as a ScrapeStatus row.

pydantic-graph needs concrete return type hints on each node's `run()`
for edge inference, so we can't do tracing via a mixin override of
run(). Instead nodes call `trace_node(...)` at the end of run() to
record the transition.

All posts are best-effort — tracing must never break a scrape.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any

import httpx

from .state import NodeTraceEntry, ScrapeGraphState

logger = logging.getLogger(__name__)


def digest(value: Any) -> str:
    """Short sha256 digest for clustering identical node invocations."""
    try:
        payload = json.dumps(value, sort_keys=True, default=str)
    except Exception:
        payload = repr(value)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def trace_node(
    state: ScrapeGraphState,
    node: str,
    routed_to: str,
    started_at: float,
    payload: dict | None = None,
) -> None:
    """Record transition locally + best-effort POST to api.

    Call at the end of a node's run() before returning the next node.
    `routed_to` is the class name of the next node (or "End" / a
    terminal tag). `started_at` is `time.time()` captured at run()
    entry.
    """
    entry = NodeTraceEntry(
        node=node,
        t_start=started_at,
        t_end=time.time(),
        routed_to=routed_to,
        payload=payload or {},
    )
    state.node_trace.append(entry)
    _post_transition(state.scrape_id, node, routed_to, entry, payload or {})


def _post_transition(
    scrape_id: int,
    node: str,
    routed_to: str,
    entry: NodeTraceEntry,
    payload: dict,
) -> None:
    if not scrape_id:
        return
    base = os.environ.get("CC_API_BASE_URL", "").rstrip("/")
    token = os.environ.get("CC_API_TOKEN")
    if not base or not token:
        logger.debug("graph-transition POST skipped — missing base/token")
        return
    body = {
        "graph_node": node,
        "graph_payload": {
            "routed_to": routed_to,
            "duration_ms": int((entry.t_end - entry.t_start) * 1000),
            **payload,
        },
    }
    try:
        resp = httpx.post(
            f"{base}/api/v1/scrapes/{scrape_id}/graph-transition/",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=5.0,
        )
    except Exception as exc:
        # Surface network/timeout failures — with shadow mode shipping,
        # a silent drop of graph transitions makes the per-scrape trace
        # page empty for no visible reason. Keep best-effort semantics
        # (don't re-raise), but log at warning so the poller's stdout
        # shows the problem.
        logger.warning(
            "graph-transition POST failed scrape_id=%s node=%s: %s",
            scrape_id, node, exc,
        )
        return
    if resp.status_code >= 400:
        body_preview = resp.text[:200] if resp.text else ""
        logger.warning(
            "graph-transition POST %s for scrape_id=%s node=%s: %s",
            resp.status_code, scrape_id, node, body_preview,
        )
