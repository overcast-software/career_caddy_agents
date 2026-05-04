"""Scrape-side nodes — Playwright navigation, ready-selector waiting,
truncation expansion, capture, persist.

Each node's `run()` has a concrete `Union[...]` return type so
pydantic-graph can infer edges. Tracing is called inline at the end
of each run() via `trace_node(state, ...)`.
"""
# ruff: noqa: F811
# Forward-declare stubs then redefine — see nodes_extract for rationale.
from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass
from typing import Union

import httpx
from pydantic_graph import BaseNode, End, GraphRunContext

from .state import ScrapeGraphState
from .tracing import trace_node
from .url_canonicalize import apply_url_rewrites, canonicalize_url, urls_differ

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


class ScrollToLoad(BaseNode[ScrapeGraphState, None, dict]):
    pass


class ExpandTruncations(BaseNode[ScrapeGraphState, None, dict]):
    pass


class Capture(BaseNode[ScrapeGraphState, None, dict]):
    pass


class PersistScrape(BaseNode[ScrapeGraphState, None, dict]):
    pass


# Obstacle-side forward refs live in nodes_obstacle; import at run time.


def _split_top_level_commas(value: str) -> list[str]:
    """Split a CSS-selector list on top-level commas.

    Commas inside parens or quoted strings stay attached to their host
    selector, so `h2:has-text("About, the job")` is one entry, not two.
    """
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    in_str: str | None = None
    for ch in value:
        if in_str:
            buf.append(ch)
            if ch == in_str:
                in_str = None
            continue
        if ch in ("'", '"'):
            in_str = ch
            buf.append(ch)
            continue
        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch == "," and depth == 0:
            chunk = "".join(buf).strip()
            if chunk:
                parts.append(chunk)
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _normalize_ready_selectors(value) -> list[str]:
    """Profile may store ready_selector as list[str] (post-0067) or str
    (pre-0067 / hand-edited). Normalize to list[str] either way.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [s for s in (str(x).strip() for x in value) if s]
    if isinstance(value, str):
        return _split_top_level_commas(value)
    return []


def _selector_hash(selectors: list[str]) -> str:
    """Short stable hash of the resolved selector list — emit in
    LoadProfile telemetry so we can correlate "WaitReadySelector kept
    timing out" with "the profile was actually loaded with this list".
    """
    if not selectors:
        return ""
    blob = "\n".join(selectors).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


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


def _flatten_profile_attrs(attrs: dict) -> dict:
    """Lift `css_selectors` keys to the top of the profile dict.

    The api ScrapeProfile stores per-host knobs (rememberme_candidates,
    interaction_hints, ready_selector, ...) inside the `css_selectors`
    JSONB blob, but graph nodes read them directly off `state.profile`
    (e.g. `state.profile.get("rememberme_candidates")`). Flatten on
    load so every reader sees one dict; same-named top-level fields
    win on collision (none today).
    """
    if not isinstance(attrs, dict):
        return {}
    css = attrs.pop("css_selectors", None) or {}
    if isinstance(css, dict):
        for k, v in css.items():
            attrs.setdefault(k, v)
    return attrs


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
                attrs = (rows[0] or {}).get("attributes") or {}
                state.profile = _flatten_profile_attrs(attrs)
        except Exception:
            logger.debug("LoadProfile: profile fetch failed", exc_info=True)
        selectors = _normalize_ready_selectors(
            (state.profile or {}).get("ready_selector")
        )
        trace_node(
            state, "LoadProfile", "Navigate", started,
            {
                "profile_hostname": host,
                "profile_loaded": bool(state.profile),
                "ready_selector_count": len(selectors),
                "ready_selector_hash": _selector_hash(selectors),
            },
        )
        return Navigate()


@dataclass
class Navigate(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> "DetectObstacle":  # noqa: F821 — forward ref, resolved via local import below
        started = time.time()
        state = ctx.state
        page = getattr(state, "_browser_page", None)
        # Host-specific rewrites happen BEFORE navigation so we land on
        # the job page directly. Profile was just loaded by LoadProfile.
        # state.profile may be the full profile (dict) or just the
        # css_selectors (legacy callers); accept either shape.
        target_url = state.submitted_url
        rewrites = None
        if isinstance(state.profile, dict):
            rewrites = state.profile.get("url_rewrites")
        target_url = apply_url_rewrites(target_url, rewrites)
        if target_url != state.submitted_url:
            state.rewritten_url = target_url
        if page is not None:
            # `domcontentloaded` lands as soon as the HTML is parsed —
            # we don't need every tracker iframe to settle, and waiting
            # on `load` can deadlock on auth interstitials (e.g.
            # LinkedIn /comm/ → account-chooser) that hold sub-resources
            # open. Downstream obstacle handling does its own waits on
            # the actual elements it cares about.
            try:
                await page.goto(target_url, wait_until="domcontentloaded", timeout=30_000)
                state.final_url = page.url
            except Exception as exc:
                state.failure_reason = f"navigate_failed: {exc}"
        from . import nodes_obstacle
        trace_node(state, "Navigate", "DetectObstacle", started)
        return nodes_obstacle.DetectObstacle()


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
        _patch_scrape_status(
            state.scrape_id, "completed",
            note=f"duplicate: job_post {state.job_post_id}",
        )
        trace_node(state, "DuplicateShortCircuit", "End", started)
        return End({
            "outcome": "duplicate",
            "job_post_id": state.job_post_id,
            "scrape_id": state.scrape_id,
        })


# Per-selector wait budget. With ~20 candidates per profile that's a
# 30s upper bound, but we exit on the first hit so the typical cost is
# one miss + one match (~1.5–3s).
_READY_SELECTOR_PER_TIMEOUT_MS = 1500


@dataclass
class WaitReadySelector(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    """Iterate the profile's ready_selector list until one matches.

    Why iterate (vs one `wait_for_selector` on a comma-joined string):
    Playwright's CSS parser rejects comma lists that mix standard CSS
    with text-engine selectors (`:has-text(...)`), and a single bad
    fragment can silently neutralize the whole list. Iterating with
    `locator(s).first.wait_for(state="visible")` also gives us
    visibility semantics (the LinkedIn description container exists
    in DOM before it paints) and per-attempt telemetry so a failing
    profile is debuggable from the trace alone.
    """

    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> SettleWait:
        started = time.time()
        state = ctx.state
        page = getattr(state, "_browser_page", None)
        selectors = _normalize_ready_selectors(
            (state.profile or {}).get("ready_selector")
        )
        attempts: list[dict] = []
        matched_selector: str | None = None
        matched_index: int | None = None
        if page and selectors:
            for idx, sel in enumerate(selectors):
                attempt_started = time.time()
                ok = False
                error: str | None = None
                try:
                    locator = page.locator(sel).first
                    await locator.wait_for(
                        state="visible",
                        timeout=_READY_SELECTOR_PER_TIMEOUT_MS,
                    )
                    ok = True
                except Exception as exc:
                    # Distinguish parser errors (selector is malformed)
                    # from timeouts so the trace tells us which is which.
                    msg = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
                    if "TimeoutError" in exc.__class__.__name__ or "timeout" in msg.lower():
                        error = "timeout"
                    elif "Unknown engine" in msg or "selector" in msg.lower():
                        error = f"parse:{msg[:80]}"
                    else:
                        error = f"{exc.__class__.__name__}:{msg[:80]}"
                attempts.append({
                    "selector": sel,
                    "index": idx,
                    "duration_ms": int((time.time() - attempt_started) * 1000),
                    "matched": ok,
                    "error": error,
                })
                if ok:
                    matched_selector = sel
                    matched_index = idx
                    break

        payload: dict = {
            "matched_selector": matched_selector,
            "matched_index": matched_index,
            "timed_out": matched_selector is None and bool(selectors),
            "selector_count": len(selectors),
            "attempts": attempts,
        }
        # Always go through SettleWait — the heading selector matching
        # only proves the section header is in DOM, not that the
        # description body has finished streaming. A fixed post-match
        # sleep lets the rest of the sections fill in before Capture
        # reads the DOM. Cheaper and more general than maintaining a
        # tail-anchor selector.
        trace_node(state, "WaitReadySelector", "SettleWait", started, payload)
        return SettleWait()


@dataclass
class SettleWait(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> ScrollToLoad:
        import asyncio
        started = time.time()
        page = getattr(ctx.state, "_browser_page", None)
        if page:
            try:
                await asyncio.sleep(5.0)
            except Exception:
                pass
        trace_node(ctx.state, "SettleWait", "ScrollToLoad", started)
        return ScrollToLoad()


# Cap on the scroll-to-load loop. ~5s total budget, ~250ms per tick =
# ~20 ticks. LinkedIn's IntersectionObserver-driven hydration of the
# "About the job" card finishes inside this window in practice.
_SCROLL_STEP_PX = 800
_SCROLL_TICK_MS = 250
_SCROLL_MAX_TICKS = 20
_SCROLL_POST_SETTLE_MS = 500


@dataclass
class ScrollToLoad(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    """Scroll the page in increments to trigger IntersectionObserver-
    driven lazy hydration (e.g. LinkedIn's "About the job" card).

    Stops as soon as one of the profile's ready_selector candidates
    matches, or when scrollHeight stops advancing for two consecutive
    ticks, or when the per-loop budget is hit. A short post-settle
    lets the just-fetched description XHR paint before
    ExpandTruncations looks for "See more".
    """

    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> ExpandTruncations:
        import asyncio
        started = time.time()
        state = ctx.state
        page = getattr(state, "_browser_page", None)
        selectors = _normalize_ready_selectors(
            (state.profile or {}).get("ready_selector")
        )
        ticks = 0
        matched: str | None = None
        last_height = -1
        stalled = 0
        final_y = 0
        if page:
            try:
                for ticks in range(1, _SCROLL_MAX_TICKS + 1):
                    try:
                        await page.evaluate(
                            f"window.scrollBy(0, {_SCROLL_STEP_PX})"
                        )
                    except Exception:
                        break
                    await asyncio.sleep(_SCROLL_TICK_MS / 1000.0)
                    for sel in selectors:
                        try:
                            handle = await page.query_selector(sel)
                            if handle is not None:
                                matched = sel
                                break
                        except Exception:
                            pass
                    if matched:
                        break
                    try:
                        height = await page.evaluate(
                            "document.body.scrollHeight"
                        )
                    except Exception:
                        height = last_height
                    if height == last_height:
                        stalled += 1
                        if stalled >= 2:
                            break
                    else:
                        stalled = 0
                        last_height = height
                try:
                    final_y = await page.evaluate("window.scrollY") or 0
                except Exception:
                    final_y = 0
                await asyncio.sleep(_SCROLL_POST_SETTLE_MS / 1000.0)
            except Exception:
                logger.debug("ScrollToLoad failed", exc_info=True)
        trace_node(
            state, "ScrollToLoad", "ExpandTruncations", started,
            {
                "ticks": ticks,
                "matched_selector": matched,
                "final_scroll_y": int(final_y) if final_y else 0,
            },
        )
        return ExpandTruncations()


@dataclass
class ExpandTruncations(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> "Capture":  # noqa: F821  — forward ref, resolved at module scope
        started = time.time()
        page = getattr(ctx.state, "_browser_page", None)
        if page:
            try:
                from mcp_servers.browser_server import _try_expand_truncations
                await _try_expand_truncations(page)
            except Exception:
                logger.debug("ExpandTruncations failed", exc_info=True)
        trace_node(ctx.state, "ExpandTruncations", "Capture", started)
        return Capture()


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

            await _screenshot_and_upload(page, state)
            await _discover_selectors(page, state)
        trace_node(state, "Capture", "PersistScrape", started)
        return PersistScrape()


async def _screenshot_and_upload(page, state: ScrapeGraphState) -> None:
    """Take a full-page screenshot, upload to the api's screenshots
    endpoint, and record the filename on state. Matches the legacy
    poller's upload path so the /scrapes/:id/screenshots/ viewer keeps
    working after the primary cutover."""
    from datetime import datetime
    from urllib.parse import urlparse
    try:
        from mcp_servers.browser_server import SCREENSHOT_DIR
    except Exception:
        logger.debug("Capture: SCREENSHOT_DIR unavailable, skipping screenshot", exc_info=True)
        return
    host = (urlparse(state.canonical_url or state.submitted_url or "").hostname or "unknown").lower()
    if host.startswith("www."):
        host = host[4:]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{host}_{ts}.png"
    path = SCREENSHOT_DIR / name
    try:
        await page.screenshot(path=str(path), full_page=True)
    except Exception as exc:
        logger.warning("Capture: screenshot failed: %s", exc)
        return
    try:
        with open(path, "rb") as f:
            resp = httpx.post(
                f"{_api_base()}/api/v1/scrapes/{state.scrape_id}/screenshots/",
                files={"file": (name, f, "image/png")},
                headers=_api_headers(),
                timeout=30.0,
            )
        if resp.status_code < 400:
            state.screenshot_name = name
        else:
            logger.warning(
                "Capture: screenshot upload %s: %s", resp.status_code, resp.text[:200]
            )
    except Exception as exc:
        logger.warning("Capture: screenshot upload exception: %s", exc)
    finally:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


async def _discover_selectors(page, state: ScrapeGraphState) -> None:
    """Run the legacy browser-server selector discovery so profile
    probation gates (UpdateProfile node) have candidates to graduate
    on repeat hits."""
    profile_selectors = (state.profile or {}) if isinstance(state.profile, dict) else {}
    try:
        if not profile_selectors.get("job_data"):
            from mcp_servers.browser_server import _discover_job_selectors
            discovered = await _discover_job_selectors(page)
            if discovered:
                state.discovered_selectors = discovered
    except Exception:
        logger.debug("Capture: selector discovery failed", exc_info=True)
    try:
        if not profile_selectors.get("ready_selector"):
            from mcp_servers.browser_server import _JOB_SELECTOR_CANDIDATES
            for sel in _JOB_SELECTOR_CANDIDATES.get("title", []):
                if not any(c in sel for c in (".", "#", "[", ":")):
                    continue
                try:
                    el = await page.query_selector(sel)
                    if el and (await el.inner_text()).strip():
                        state.candidate_ready_selector = sel
                        break
                except Exception:
                    continue
    except Exception:
        logger.debug("Capture: ready-selector probing failed", exc_info=True)


@dataclass
class PersistScrape(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> "StartExtract":  # noqa: F821 — resolved via graph.py namespace
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
                            "note": (
                                f"Content delivered ({len(state.job_content or '')} chars)"
                            ),
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


def _patch_scrape_status(scrape_id: int, status: str, note: str | None = None) -> None:
    """Helper used by terminal nodes to close out the scrape row with
    the frontend-visible status the poller-polling UI watches for
    ({completed, failed})."""
    if not scrape_id:
        return
    attributes = {"status": status}
    if note is not None:
        attributes["note"] = note
    try:
        httpx.patch(
            f"{_api_base()}/api/v1/scrapes/{scrape_id}/",
            json={
                "data": {
                    "type": "scrape",
                    "id": str(scrape_id),
                    "attributes": attributes,
                }
            },
            headers={**_api_headers(), "Content-Type": "application/json"},
            timeout=15.0,
        )
    except Exception:
        logger.warning("terminal scrape PATCH failed scrape_id=%s", scrape_id, exc_info=True)


__all__ = [
    "StartScrape",
    "LoadProfile",
    "Navigate",
    "ResolveFinalUrl",
    "CheckLinkDedup",
    "DuplicateShortCircuit",
    "WaitReadySelector",
    "SettleWait",
    "ScrollToLoad",
    "ExpandTruncations",
    "Capture",
    "PersistScrape",
]
