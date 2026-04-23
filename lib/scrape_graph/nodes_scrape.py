"""Scrape-side nodes — Playwright navigation, ready-selector waiting,
truncation expansion, capture, persist.

Each node's `run()` has a concrete `Union[...]` return type so
pydantic-graph can infer edges. Tracing is called inline at the end
of each run() via `trace_node(state, ...)`.
"""
# ruff: noqa: F811
# Forward-declare stubs then redefine — see nodes_extract for rationale.
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Union

import httpx
from pydantic_graph import BaseNode, End, GraphRunContext

from .state import ScrapeGraphState
from .tracing import trace_node
from .url_canonicalize import canonicalize_url, urls_differ

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Forward-reference stubs so Union annotations resolve. Real nodes below
# redefine them. pydantic-graph uses get_type_hints to read the annotation,
# so the names must be in module scope at class-definition time.
# ---------------------------------------------------------------------------

class LoadProfile(BaseNode[ScrapeGraphState, None, dict]):  # re-declared below
    pass


class Navigate(BaseNode[ScrapeGraphState, None, dict]):
    pass


class ResolveFinalUrl(BaseNode[ScrapeGraphState, None, dict]):
    pass


class CheckLinkDedup(BaseNode[ScrapeGraphState, None, dict]):
    pass


class DuplicateShortCircuit(BaseNode[ScrapeGraphState, None, dict]):
    pass


class WaitReadySelector(BaseNode[ScrapeGraphState, None, dict]):
    pass


class SettleWait(BaseNode[ScrapeGraphState, None, dict]):
    pass


class ExpandTruncations(BaseNode[ScrapeGraphState, None, dict]):
    pass


class Capture(BaseNode[ScrapeGraphState, None, dict]):
    pass


class PersistScrape(BaseNode[ScrapeGraphState, None, dict]):
    pass


# Obstacle-side forward refs live in nodes_obstacle; import at run time.


def _api_base() -> str:
    return os.environ.get("CC_API_BASE_URL", "").rstrip("/")


def _api_headers() -> dict[str, str]:
    token = os.environ.get("CC_API_TOKEN", "")
    return {"Authorization": f"Bearer {token}"} if token else {}


# ---------------------------------------------------------------------------
# Real node implementations. Each class shadows the forward-ref stub above.
# ---------------------------------------------------------------------------

@dataclass
class StartScrape(BaseNode[ScrapeGraphState, None, dict]):
    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> LoadProfile:
        started = time.time()
        state = ctx.state
        if not state.original_scrape_id:
            state.original_scrape_id = state.scrape_id
        trace_node(state, "StartScrape", "LoadProfile", started)
        return LoadProfile()


@dataclass
class LoadProfile(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> Navigate:
        from urllib.parse import urlparse
        started = time.time()
        state = ctx.state
        host = (urlparse(state.submitted_url or "").hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        try:
            resp = httpx.get(
                f"{_api_base()}/api/v1/scrape-profiles/",
                params={"filter[hostname]": host},
                headers=_api_headers(),
                timeout=10.0,
            )
            payload = resp.json() if resp.status_code == 200 else {}
            rows = payload.get("data") or []
            if rows:
                state.profile = (rows[0] or {}).get("attributes")
        except Exception:
            logger.debug("LoadProfile: profile fetch failed", exc_info=True)
        trace_node(state, "LoadProfile", "Navigate", started)
        return Navigate()


@dataclass
class Navigate(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> ResolveFinalUrl:
        started = time.time()
        state = ctx.state
        page = getattr(state, "_browser_page", None)
        if page is not None:
            try:
                await page.goto(state.submitted_url, wait_until="load", timeout=60_000)
                state.final_url = page.url
            except Exception as exc:
                state.failure_reason = f"navigate_failed: {exc}"
        trace_node(state, "Navigate", "ResolveFinalUrl", started)
        return ResolveFinalUrl()


@dataclass
class ResolveFinalUrl(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> CheckLinkDedup:
        started = time.time()
        state = ctx.state
        landed = state.final_url or state.submitted_url
        state.canonical_url = canonicalize_url(landed)
        if urls_differ(state.submitted_url, landed):
            state.did_redirect = True
            # Chain a child scrape via source_scrape FK so provenance
            # of the tracker → destination step is queryable via
            # Scrape.child_scrapes later.
            try:
                resp = httpx.post(
                    f"{_api_base()}/api/v1/scrapes/",
                    json={
                        "data": {
                            "attributes": {
                                "url": state.canonical_url,
                                "source": "redirect",
                            },
                            "relationships": {
                                "source-scrape": {
                                    "data": {
                                        "type": "scrape",
                                        "id": str(state.scrape_id),
                                    }
                                }
                            },
                        }
                    },
                    headers={**_api_headers(), "Content-Type": "application/json"},
                    timeout=10.0,
                )
                if resp.status_code in (200, 201):
                    new_id = (resp.json() or {}).get("data", {}).get("id")
                    if new_id:
                        state.scrape_id = int(new_id)
            except Exception:
                logger.warning("ResolveFinalUrl: child-scrape create failed", exc_info=True)
        trace_node(
            state,
            "ResolveFinalUrl",
            "CheckLinkDedup",
            started,
            {"did_redirect": state.did_redirect, "canonical_url": state.canonical_url},
        )
        return CheckLinkDedup()


@dataclass
class CheckLinkDedup(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> Union[DuplicateShortCircuit, WaitReadySelector]:
        started = time.time()
        state = ctx.state
        canonical = state.canonical_url or state.submitted_url
        non_stub_id: int | None = None
        try:
            resp = httpx.get(
                f"{_api_base()}/api/v1/job-posts/",
                params={"filter[link]": canonical},
                headers=_api_headers(),
                timeout=10.0,
            )
            rows = (resp.json() or {}).get("data", []) if resp.status_code == 200 else []
            for row in rows:
                desc = (row.get("attributes") or {}).get("description") or ""
                if len(desc.split()) >= 60:
                    non_stub_id = int(row["id"])
                    break
        except Exception:
            pass
        if non_stub_id:
            state.job_post_id = non_stub_id
            state.was_duplicate = True
            trace_node(state, "CheckLinkDedup", "DuplicateShortCircuit", started)
            return DuplicateShortCircuit()
        trace_node(state, "CheckLinkDedup", "WaitReadySelector", started)
        return WaitReadySelector()


@dataclass
class DuplicateShortCircuit(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> End[dict]:
        started = time.time()
        state = ctx.state
        state.outcome = "duplicate"
        trace_node(state, "DuplicateShortCircuit", "End", started)
        return End({
            "outcome": "duplicate",
            "job_post_id": state.job_post_id,
            "scrape_id": state.scrape_id,
        })


@dataclass
class WaitReadySelector(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> Union[ExpandTruncations, SettleWait]:
        started = time.time()
        state = ctx.state
        page = getattr(state, "_browser_page", None)
        selector = (state.profile or {}).get("ready_selector")
        if page and selector:
            try:
                await page.wait_for_selector(selector, timeout=5_000)
                trace_node(state, "WaitReadySelector", "ExpandTruncations", started)
                return ExpandTruncations()
            except Exception:
                pass
        trace_node(state, "WaitReadySelector", "SettleWait", started)
        return SettleWait()


@dataclass
class SettleWait(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> ExpandTruncations:
        import asyncio
        started = time.time()
        page = getattr(ctx.state, "_browser_page", None)
        if page:
            try:
                await asyncio.sleep(2.0)
            except Exception:
                pass
        trace_node(ctx.state, "SettleWait", "ExpandTruncations", started)
        return ExpandTruncations()


@dataclass
class ExpandTruncations(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> "DetectObstacle":  # noqa: F821  — forward ref, resolved via local import below
        from . import nodes_obstacle
        started = time.time()
        page = getattr(ctx.state, "_browser_page", None)
        if page:
            try:
                from mcp_servers.browser_server import _try_expand_truncations
                await _try_expand_truncations(page)
            except Exception:
                logger.debug("ExpandTruncations failed", exc_info=True)
        trace_node(ctx.state, "ExpandTruncations", "DetectObstacle", started)
        return nodes_obstacle.DetectObstacle()


@dataclass
class Capture(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> PersistScrape:
        started = time.time()
        state = ctx.state
        page = getattr(state, "_browser_page", None)
        if page:
            try:
                state.job_content = await page.inner_text("body")
                state.html = await page.content()
            except Exception as exc:
                state.failure_reason = f"capture_failed: {exc}"
        trace_node(state, "Capture", "PersistScrape", started)
        return PersistScrape()


@dataclass
class PersistScrape(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ):  # return annotation lazy-set from nodes_extract
        from . import nodes_extract
        started = time.time()
        state = ctx.state
        try:
            httpx.patch(
                f"{_api_base()}/api/v1/scrapes/{state.scrape_id}/",
                json={
                    "data": {
                        "type": "scrape",
                        "id": str(state.scrape_id),
                        "attributes": {
                            "job_content": state.job_content,
                            "html": state.html,
                            "status": "extracting",
                        },
                    }
                },
                headers={**_api_headers(), "Content-Type": "application/json"},
                timeout=30.0,
            )
        except Exception:
            logger.warning("PersistScrape: patch failed", exc_info=True)
        trace_node(state, "PersistScrape", "StartExtract", started)
        return nodes_extract.StartExtract()


__all__ = [
    "StartScrape",
    "LoadProfile",
    "Navigate",
    "ResolveFinalUrl",
    "CheckLinkDedup",
    "DuplicateShortCircuit",
    "WaitReadySelector",
    "SettleWait",
    "ExpandTruncations",
    "Capture",
    "PersistScrape",
]
