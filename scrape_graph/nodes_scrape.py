"""Scrape-side nodes — Playwright navigation, ready-selector waiting,
truncation expansion, capture, persist.

Each node's `run()` has a concrete `Union[...]` return type so
pydantic-graph can infer edges. Tracing is called inline at the end
of each run() via `trace_node(state, ...)`.
"""
# ruff: noqa: F811
# Forward-declare stubs then redefine — see nodes_extract for rationale.
from __future__ import annotations

import asyncio
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

# Wall-clock budget for ResolveFinalUrl — bounds the redirect-handoff
# work (canonicalize + child-scrape POST + parent terminal-close) so a
# wedged httpx call or an unresponsive api can't park the parent scrape
# in `running` indefinitely. See TODO #309 (ZipRecruiter /km/ tracker
# URLs hung for hours).
_RESOLVE_FINAL_URL_BUDGET_S = float(
    os.environ.get("SCRAPE_GRAPH_RESOLVE_FINAL_URL_BUDGET_S", "15")
)


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


class DetectClosedState(BaseNode[ScrapeGraphState, None, dict]):
    pass


class PersistScrape(BaseNode[ScrapeGraphState, None, dict]):
    pass


class SkipBrowserTier(BaseNode[ScrapeGraphState, None, dict]):
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

_EXTENSION_DIRECT_REQUIRED_FIELDS = ("title", "company", "description")


def _captured_payload_is_complete(payload) -> bool:
    """Mirror of api-side ScrapeSerializer's extension-direct gate
    (validate_scrape_source_mode_payload). Returns True when the payload
    is a dict whose required fields (title / company / description) are
    each non-empty strings.

    Defense-in-depth: Phase A's serializer rejects an incomplete payload
    at write time, so a claimed source_mode='extension-direct' Scrape
    should always satisfy this check. The guard exists for the case
    where the api gate regresses or a hand-rolled curl sneaks past it —
    we route to ExtractFail rather than silently degrading to the
    browser tier (which would defeat the whole point of source_mode).
    """
    if not isinstance(payload, dict):
        return False
    for field_name in _EXTENSION_DIRECT_REQUIRED_FIELDS:
        value = payload.get(field_name)
        if not isinstance(value, str) or not value.strip():
            return False
    return True


@dataclass
class StartScrape(BaseNode[ScrapeGraphState, None, dict]):
    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> Union[LoadProfile, "SkipBrowserTier", "ExtractFail"]:  # noqa: F821 — ExtractFail resolved via graph.py namespace
        started = time.time()
        state = ctx.state
        if not state.original_scrape_id:
            state.original_scrape_id = state.scrape_id
        if state.source_mode == "extension-direct":
            if _captured_payload_is_complete(state.captured_payload):
                trace_node(
                    state, "StartScrape", "SkipBrowserTier", started,
                    {"source_mode": state.source_mode, "fast_path": True},
                )
                return SkipBrowserTier()
            # Defensive bail — incomplete payload on a claimed extension-direct
            # scrape means the api-side validator regressed. We refuse to
            # silently degrade to the browser tier because the user already
            # rendered the page and the extension promised the payload; if
            # we re-fetch with Camoufox we may land on an auth wall or
            # different content, polluting the JobPost. ExtractFail
            # produces a debug artifact + leaves the scrape in
            # status='failed' for operator inspection.
            from . import nodes_extract
            state.failure_reason = (
                "extension-direct: captured_payload missing required "
                "title/company/description"
            )
            trace_node(
                state, "StartScrape", "ExtractFail", started,
                {
                    "source_mode": state.source_mode,
                    "payload_present": state.captured_payload is not None,
                    "reason": "incomplete_captured_payload",
                },
            )
            return nodes_extract.ExtractFail()
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


def _propagate_canonical_to_parent_jp(
    parent_scrape_id: int, resolved_canonical_url: str
) -> None:
    """Best-effort: when a parent scrape's URL resolves through a
    redirect chain, propagate the resolved canonical URL back to the
    JobPost (if any) that the parent scrape was attached to.

    Motivating case (2026-05-28 jp 3036 / jp 3040 incident): cc_auto's
    email pipeline created jp 3036 with `link` = SendGrid tracker
    `u52508838.ct.sendgrid.net/ls/click?upn=…`. SendGrid's `upn` is an
    opaque hashed token, so canonicalize_link can't decode it — jp
    3036.canonical_link stayed pinned to the wrapper. Later, scrape
    477 ran on the resolved destination `hiring.cafe/job/<id>`
    directly and created a fresh jp 3040 — the two rows represent the
    same opening but the canonical_link comparison couldn't merge
    them. Without this propagation step, every wrapped-URL JobPost
    accumulates an unmerged duplicate the first time the destination
    is captured directly.

    Behavior:
    - GET the parent scrape; if no `job_post` relationship, no-op.
    - GET the JobPost; if its canonical_link already equals the
      resolved URL (idempotent re-run), no-op.
    - Otherwise PATCH the JobPost's `canonical_link`. The api's PATCH
      path leaves `link` alone (audit / wrapper preserved) and the
      JobPost.save() guard only re-derives canonical_link when it's
      empty, so the value we send lands verbatim.

    All exceptions are swallowed and logged — this is enhancement
    for downstream dedupe, not load-bearing for the scrape itself.
    The redirect-handoff to the child scrape proceeds either way.
    """
    try:
        resp = httpx.get(
            f"{_api_base()}/api/v1/scrapes/{parent_scrape_id}/",
            headers=_api_headers(),
            timeout=5.0,
        )
        if resp.status_code != 200:
            return
        rel = (
            ((resp.json() or {}).get("data") or {})
            .get("relationships", {})
            .get("job-post")
            or {}
        )
        jp_data = rel.get("data")
        if not jp_data:
            return
        jp_id = jp_data.get("id")
        if not jp_id:
            return

        get_resp = httpx.get(
            f"{_api_base()}/api/v1/job-posts/{jp_id}/",
            headers=_api_headers(),
            timeout=5.0,
        )
        if get_resp.status_code != 200:
            return
        current_canonical = (
            ((get_resp.json() or {}).get("data") or {})
            .get("attributes", {})
            .get("canonical_link")
        )
        if current_canonical == resolved_canonical_url:
            return

        httpx.patch(
            f"{_api_base()}/api/v1/job-posts/{jp_id}/",
            json={
                "data": {
                    "type": "job-post",
                    "id": str(jp_id),
                    "attributes": {"canonical_link": resolved_canonical_url},
                }
            },
            headers={**_api_headers(), "Content-Type": "application/json"},
            timeout=5.0,
        )
    except Exception:
        logger.warning(
            "ResolveFinalUrl: canonical_link propagation to parent JobPost failed",
            exc_info=True,
        )


def _resolve_final_url_body(state: ScrapeGraphState) -> None:
    """Synchronous core of ResolveFinalUrl — split out so we can wrap
    the whole thing in a thread + asyncio.wait_for budget below.

    Mutates state in place: canonical_url, did_redirect, scrape_id (on
    redirect handoff). Idempotent on the no-redirect path.
    """
    landed = state.final_url or state.submitted_url
    state.canonical_url = canonicalize_url(landed)
    if not urls_differ(state.submitted_url, landed):
        return
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
                # Terminal-close the parent before swapping
                # state.scrape_id to the child. Without this, the
                # parent stays at status='running' forever — every
                # subsequent trace_node / _patch_scrape_status call
                # below targets the child, so no terminal PATCH
                # ever lands on the parent and the poller keeps
                # re-dispatching it. Provenance is preserved via
                # the child's source_scrape FK (set on POST above).
                _patch_scrape_status(
                    state.scrape_id, "completed",
                    note=(
                        f"redirected to scrape {new_id} "
                        f"({state.canonical_url})"
                    ),
                )
                # Propagate the resolved canonical_link back to the
                # JobPost (if any) attached to the parent scrape. Runs
                # against the still-parent state.scrape_id BEFORE the
                # swap below. Best-effort — failures don't block the
                # handoff. See _propagate_canonical_to_parent_jp for
                # the JP 3036 motivating incident.
                _propagate_canonical_to_parent_jp(
                    state.scrape_id, state.canonical_url
                )
                state.scrape_id = int(new_id)
    except Exception:
        logger.warning("ResolveFinalUrl: child-scrape create failed", exc_info=True)


@dataclass
class ResolveFinalUrl(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> CheckLinkDedup:
        started = time.time()
        state = ctx.state
        # Settle JS / meta-refresh redirects that fire AFTER Navigate's
        # `domcontentloaded` cutoff. Email-tracker URLs (ZipRecruiter
        # /km/<token>, SendGrid /ls/click, Mailgun /c/) all drive the
        # redirect from a meta-refresh tag or window.location.replace,
        # which executes after the HTML parses but before the page
        # finishes loading. Navigate captured state.final_url at
        # domcontentloaded so it's frozen at the tracker URL — re-read
        # page.url here once the network goes idle so the redirect
        # destination is what we hand to _resolve_final_url_body. 5s cap
        # so a hung tracker can't eat the 15s outer budget.
        #
        # The page.url re-read is INTENTIONALLY outside the networkidle
        # try/except. LinkedIn (and other tracker-heavy SPAs) keep the
        # network perpetually busy with heartbeat / telemetry beacons,
        # so wait_for_load_state("networkidle") routinely times out even
        # though navigation completed and the URL is up-to-date. The
        # 2026-05-28 JP 715 incident: ObstacleRememberMe logged the user
        # into LinkedIn, the browser navigated to the real job page, the
        # screenshot at Capture-time confirmed the job content rendered
        # — but state.final_url stayed at the /uas/login wrapper because
        # the timeout exception jumped over the page.url read. Result:
        # CheckLinkDedup compared the login wrapper against existing
        # canonical_links and missed an obvious dup against JP 2963.
        # Two separate try/except blocks so networkidle-timeout never
        # blocks the URL read.
        page = getattr(state, "_browser_page", None)
        if page is not None:
            try:
                await page.wait_for_load_state("networkidle", timeout=5_000)
            except Exception:
                pass  # best-effort; the URL read below handles the fallback
            try:
                state.final_url = page.url
            except Exception:
                pass  # truly hopeless — keep Navigate's capture as last resort
        # Wrap the sync body in a thread + wait_for so a wedged httpx
        # call (api unreachable, tracker host hanging on a half-open
        # connection) can't park the parent scrape in `running`
        # forever. On timeout: best-effort close the parent so the
        # poller stops re-dispatching, then continue to CheckLinkDedup
        # with whatever state we managed to set. canonical_url is
        # written eagerly inside the body before any blocking call, so
        # downstream nodes still have something to work with.
        timed_out = False
        try:
            await asyncio.wait_for(
                asyncio.to_thread(_resolve_final_url_body, state),
                timeout=_RESOLVE_FINAL_URL_BUDGET_S,
            )
        except asyncio.TimeoutError:
            timed_out = True
            logger.warning(
                "ResolveFinalUrl: budget exceeded scrape_id=%s budget_s=%s — "
                "closing parent and continuing",
                state.scrape_id, _RESOLVE_FINAL_URL_BUDGET_S,
            )
            # Best-effort terminal close so the parent doesn't sit in
            # `running`. Note routes the operator to the timeout cause.
            try:
                _patch_scrape_status(
                    state.scrape_id, "failed",
                    note=(
                        f"ResolveFinalUrl timeout after "
                        f"{_RESOLVE_FINAL_URL_BUDGET_S:g}s"
                    ),
                )
            except Exception:
                logger.warning(
                    "ResolveFinalUrl: post-timeout PATCH failed scrape_id=%s",
                    state.scrape_id, exc_info=True,
                )
            # Make sure canonical_url is at least the submitted URL so
            # CheckLinkDedup has something non-empty to filter on.
            if not state.canonical_url:
                state.canonical_url = state.submitted_url
        trace_node(
            state,
            "ResolveFinalUrl",
            "CheckLinkDedup",
            started,
            {
                "did_redirect": state.did_redirect,
                "canonical_url": state.canonical_url,
                "timed_out": timed_out,
                "budget_s": _RESOLVE_FINAL_URL_BUDGET_S,
            },
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


# Per-selector wait budget. Short enough that one full pass over a
# ~20-candidate list completes in <10s, leaving room for the outer
# loop (below) to cycle the list multiple times within the total
# budget. Hydration races on SDUI sites (LinkedIn) typically resolve
# in seconds, so a short per-attempt + retry-soon beats a long
# per-attempt + try-once.
_READY_SELECTOR_PER_TIMEOUT_MS = 500

# Overall WaitReadySelector budget. The inner loop iterates the
# selector list repeatedly, exiting on first match or when this
# budget is exhausted. ~30s gives ~3 passes over a 20-selector list
# at 500ms each.
_READY_SELECTOR_TOTAL_BUDGET_MS = 30_000

# Below this per-pass wall time we assume the wait_for is faked and
# bail to avoid spinning the budget loop. In production, even an
# all-miss pass against a single selector spends ~per-timeout ms
# (~500ms), well above this threshold.
_READY_SELECTOR_MIN_PASS_MS = 50


@dataclass
class WaitReadySelector(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    """Loop the profile's ready_selector list until one matches.

    Why iterate (vs one `wait_for_selector` on a comma-joined string):
    Playwright's CSS parser rejects comma lists that mix standard CSS
    with text-engine selectors (`:has-text(...)`), and a single bad
    fragment can silently neutralize the whole list. Iterating with
    `locator(s).first.wait_for(state="visible")` also gives us
    visibility semantics (the LinkedIn description container exists
    in DOM before it paints) and per-attempt telemetry so a failing
    profile is debuggable from the trace alone.

    Why loop (vs one pass): SDUI hydration is a race. A selector that
    misses on pass 1 (because that card hasn't hydrated yet) may land
    by pass 2. Re-checking the whole list on a tight cycle catches
    late-arriving content sooner than waiting a long per-attempt
    timeout once.
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
        matched_pass: int | None = None
        passes = 0
        budget_s = _READY_SELECTOR_TOTAL_BUDGET_MS / 1000.0
        if page and selectors:
            while (
                matched_selector is None
                and (time.time() - started) < budget_s
            ):
                passes += 1
                pass_started = time.time()
                for idx, sel in enumerate(selectors):
                    if (time.time() - started) >= budget_s:
                        break
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
                        "pass": passes,
                        "duration_ms": int((time.time() - attempt_started) * 1000),
                        "matched": ok,
                        "error": error,
                    })
                    if ok:
                        matched_selector = sel
                        matched_index = idx
                        matched_pass = passes
                        break
                # If the entire pass returned in negligible wall time
                # (no real Playwright sleeping), we're in test/fake
                # mode — bail to avoid spinning the budget loop.
                if (
                    matched_selector is None
                    and (time.time() - pass_started) * 1000 < _READY_SELECTOR_MIN_PASS_MS
                ):
                    break

        payload: dict = {
            "matched_selector": matched_selector,
            "matched_index": matched_index,
            "matched_pass": matched_pass,
            "passes": passes,
            "timed_out": matched_selector is None and bool(selectors),
            "selector_count": len(selectors),
            "total_duration_ms": int((time.time() - started) * 1000),
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
    ) -> "DetectClosedState":
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
        # DetectClosedState runs while DOM is still live so the CSS path
        # can probe the page; passes through to PersistScrape regardless
        # of verdict (closed-state is metadata, never terminal).
        trace_node(state, "Capture", "DetectClosedState", started)
        return DetectClosedState()


async def _screenshot_and_upload(page, state: ScrapeGraphState) -> None:
    """Happy-path screenshot for the scrapes/:id/screenshots/ viewer.

    Delegates to the shared in-memory uploader in ``_artifacts``. Until
    2026-05-05 this had its own disk-round-trip implementation that
    imported ``mcp_servers.browser_server.SCREENSHOT_DIR`` and called
    ``page.screenshot(full_page=True)`` with no explicit timeout —
    Playwright's 30s default fired on every LinkedIn login-wall page
    and the screenshot silently dropped (scrape 322 was the last
    happy-path screenshot to land before the regression became
    visible). The shared helper uses a viewport-only snap with a 5s
    timeout so brittle pages still produce a usable artifact.
    """
    from ._artifacts import upload_page_screenshot
    await upload_page_screenshot(page, state)


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
class DetectClosedState(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    """Probe captured page (live DOM + text) for closed-state signal.

    Three paths in priority order:
      1. CSS selectors against the live Playwright page (deterministic,
         host-curated). Runs in full so the trace records every
         attempted selector — first hit wins the verdict.
      2. text_phrases + learned_phrases regex/substring scan against
         state.job_content (deterministic).
      3. Haiku LLM fallback — only when no host config defined AND
         len(job_content) >= min_chars_for_llm. Quote is verbatim-
         validated against text. Hits are PROMOTED back to the host's
         closed_state_config.learned_phrases for future runs.

    Always routes to PersistScrape — closed-state is metadata, not a
    graph terminal. The verdict + evidence land on Scrape.detected_*
    columns via the next node's PATCH; downstream JobPostExtractor
    reads them as the priority-1 channel for posting_status flips.
    """
    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> "PersistScrape":
        from .closed_state_detector import (
            DEFAULT_MIN_CHARS_FOR_LLM,
            TRACE_QUOTE_MAX,
            detect_via_css,
            detect_via_phrases,
            detect_via_llm,
        )
        started = time.time()
        state = ctx.state
        page = getattr(state, "_browser_page", None)
        text = state.job_content or ""

        # Pull config out of the existing css_selectors JSONB blob, same
        # shape pattern as apply_resolver_config. None of the keys are
        # required; missing/empty → fall through.
        profile = state.profile if isinstance(state.profile, dict) else {}
        css_blob = profile.get("css_selectors") if isinstance(profile.get("css_selectors"), dict) else {}
        cfg = (css_blob or {}).get("closed_state_config") or {}
        if not isinstance(cfg, dict):
            cfg = {}
        css_sels = cfg.get("css_selectors") or []
        phrases  = cfg.get("text_phrases") or []
        learned  = cfg.get("learned_phrases") or []
        min_chars = int(cfg.get("min_chars_for_llm") or DEFAULT_MIN_CHARS_FOR_LLM)

        config_present = {
            "css_selectors": len(css_sels) if isinstance(css_sels, list) else 0,
            "text_phrases":  len(phrases)  if isinstance(phrases,  list) else 0,
            "learned_phrases": len(learned) if isinstance(learned, list) else 0,
            "min_chars_for_llm": min_chars,
        }

        css_block: dict | None = None
        phrase_block: dict | None = None
        llm_block: dict | None = None
        promoted_block: dict | None = None

        verdict: str | None = None
        evidence: str | None = None
        method: str = "no_config"

        # Path 1: CSS — runs all selectors so the trace records full
        # attempt history (per the "Run both, log both" choice).
        if css_sels and page:
            attempts, hit = await detect_via_css(page, list(css_sels))
            css_block = {"ran": True, "attempts": attempts, "matched_selector": hit["selector"] if hit else None}
            if hit:
                verdict = "closed"
                evidence = hit.get("snippet") or hit["selector"]
                method = "css"

        # Path 2: phrases — also runs even if CSS hit, so the trace can
        # show whether the cheaper phrase path WOULD have caught it
        # too (regret-analysis signal). Phrase result only becomes the
        # verdict if CSS didn't hit.
        if (phrases or learned):
            phrase_hit = detect_via_phrases(
                text,
                list(phrases) if isinstance(phrases, list) else [],
                learned if isinstance(learned, list) else [],
            )
            phrase_block = {"ran": True, **(phrase_hit or {"matched": False})}
            if phrase_hit and verdict is None:
                verdict = "closed"
                evidence = phrase_hit.get("matched_substring")
                method = "phrase"

        # Path 3: LLM bootstrap — only fires when host has NO config
        # at all AND captured text is substantive. Cost guard prevents
        # degraded auth-walled chrome captures from triggering the
        # call (jp 1532 incident: 756 chars → would have flipped).
        if not css_sels and not phrases and not learned:
            if len(text) >= min_chars:
                llm_result = await detect_via_llm(text)
                quote = (llm_result.get("evidence_quote") or "").strip()
                quote_validated = bool(quote and quote in text)
                llm_block = {
                    "ran": True,
                    "model": llm_result.get("model"),
                    "captured_chars": len(text),
                    "min_chars": min_chars,
                    "verdict": "closed" if llm_result.get("is_closed") else "open",
                    "evidence_quote": (quote or "")[:TRACE_QUOTE_MAX] or None,
                    "quote_validated": quote_validated,
                    "duration_ms": llm_result.get("duration_ms", 0),
                    "error": llm_result.get("error"),
                }
                if llm_result.get("is_closed") and quote_validated:
                    verdict = "closed"
                    evidence = quote
                    method = "llm"
                    promoted_block = self._promote(state, quote)
            else:
                method = "skipped_thin_capture"
                llm_block = {
                    "ran": False,
                    "reason": "below_min_chars",
                    "captured_chars": len(text),
                    "min_chars": min_chars,
                }
        elif (css_sels or phrases or learned) and verdict is None:
            method = "no_signal"

        if llm_block is None and (css_sels or phrases or learned):
            llm_block = {"ran": False, "reason": "config_present"}

        state.detected_posting_status = verdict
        state.detected_closed_evidence = evidence
        state.closed_detection_method = method

        payload = {
            "method": method,
            "verdict": verdict,
            "evidence": (evidence or "")[:TRACE_QUOTE_MAX] or None,
            "config_present": config_present,
            "css": css_block,
            "phrase": phrase_block,
            "llm": llm_block,
            "promoted_learned_phrase": (promoted_block or {}).get("phrase") if promoted_block else None,
        }
        note = self._note(method, verdict, evidence, css_block, phrase_block)
        trace_node(state, "DetectClosedState", "PersistScrape", started, payload=payload, note=note)
        if promoted_block:
            # Surface the side-effect as its own ScrapeStatus row so the
            # trace shows BOTH the verdict AND the learning that came
            # from it. Useful when auditing whether a learned phrase is
            # responsible for a later false positive.
            try:
                httpx.post(
                    f"{_api_base()}/api/v1/scrapes/{state.scrape_id}/graph-transition/",
                    json={
                        "graph_node": "DetectClosedState",
                        "graph_payload": {
                            "routed_to": "PersistScrape",
                            "duration_ms": 0,
                            "method": "promote_learned_phrase",
                            **promoted_block,
                        },
                        "note": (
                            f"Promoted learned phrase to ScrapeProfile "
                            f"#{promoted_block.get('promoted_to_profile_id')}: "
                            f"{(promoted_block.get('phrase') or '')[:80]!r}"
                        ),
                    },
                    headers={**_api_headers(), "Content-Type": "application/json"},
                    timeout=5.0,
                )
            except Exception:
                logger.warning("DetectClosedState: promote-status post failed", exc_info=True)
        return PersistScrape()

    def _note(
        self,
        method: str,
        verdict: str | None,
        evidence: str | None,
        css_block: dict | None,
        phrase_block: dict | None,
    ) -> str:
        if verdict == "closed":
            if method == "css":
                sel = (css_block or {}).get("matched_selector") or "?"
                return f"DetectClosedState: closed via css {sel!r}"
            if method == "phrase":
                pat = (phrase_block or {}).get("matched_pattern") or "?"
                src = (phrase_block or {}).get("source") or "?"
                return f"DetectClosedState: closed via {src} phrase {pat!r}"
            if method == "llm":
                ev = (evidence or "")[:60]
                return f"DetectClosedState: closed via llm — {ev!r}"
            return f"DetectClosedState: closed ({method})"
        if method == "skipped_thin_capture":
            return "DetectClosedState: open — skipped LLM (capture too thin)"
        if method == "no_config":
            return "DetectClosedState: open — no host config, capture too thin for LLM"
        return f"DetectClosedState: open ({method})"

    def _promote(self, state, phrase: str) -> dict | None:
        """Append the LLM-validated phrase to the host's
        ``closed_state_config.learned_phrases`` so future scrapes use
        the cheap phrase path. PATCH against the api; failure is logged
        but doesn't fail the node.
        """
        from datetime import datetime, timezone

        profile = state.profile if isinstance(state.profile, dict) else {}
        profile_id = profile.get("id") or profile.get("profile_id")
        if not profile_id or not phrase:
            return None
        css_blob = profile.get("css_selectors") if isinstance(profile.get("css_selectors"), dict) else {}
        existing_cfg = (css_blob or {}).get("closed_state_config") or {}
        existing_learned = (existing_cfg.get("learned_phrases") or []) if isinstance(existing_cfg.get("learned_phrases"), list) else []
        # Idempotency: don't re-append a phrase already learned.
        for entry in existing_learned:
            if isinstance(entry, dict) and entry.get("phrase") == phrase:
                return None
        new_entry = {
            "phrase": phrase,
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "from_scrape_id": state.scrape_id,
        }
        new_cfg = {**existing_cfg, "learned_phrases": existing_learned + [new_entry]}
        new_blob = {**(css_blob or {}), "closed_state_config": new_cfg}
        try:
            resp = httpx.patch(
                f"{_api_base()}/api/v1/scrape-profiles/{profile_id}/",
                json={
                    "data": {
                        "type": "scrape-profile",
                        "id": str(profile_id),
                        "attributes": {"css_selectors": new_blob},
                    }
                },
                headers={**_api_headers(), "Content-Type": "application/json"},
                timeout=10.0,
            )
            patch_succeeded = resp.status_code < 400
        except Exception:
            logger.warning("DetectClosedState: profile patch failed", exc_info=True)
            patch_succeeded = False
        return {
            "phrase": phrase,
            "promoted_to_profile_id": profile_id,
            "patch_succeeded": patch_succeeded,
        }


@dataclass
class PersistScrape(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> Union["StartExtract", "ResolveApplyUrl"]:  # noqa: F821 — resolved via graph.py namespace
        from . import nodes_extract
        started = time.time()
        state = ctx.state

        # Last-chance DOM capture. Capture sets `state.html` from
        # `page.content()`, but that call can fail (frame detached,
        # "execution context destroyed" on a still-navigating SPA) while
        # the `inner_text("body")` read just before it succeeds — leaving
        # `state.html` empty but `state.job_content` populated. The Fail
        # path recovers via `_artifacts.capture_debug_artifact`'s re-grab;
        # the success path had no such recovery, so a browser scrape whose
        # `Capture` content() hiccuped persisted null html and the DOM that
        # `inspect_scrape_html` / `find_selectors_for_text` read was lost.
        # Re-grab here from the still-live page (now settled past
        # navigation) so every browser scrape that reaches PersistScrape
        # lands its raw HTML at least once. Best-effort: never raises.
        page = getattr(state, "_browser_page", None)
        if not state.html and page is not None:
            try:
                state.html = await page.content()
            except Exception:
                logger.debug(
                    "PersistScrape: html re-grab failed scrape_id=%s",
                    state.scrape_id, exc_info=True,
                )

        attributes: dict = {
            "job_content": state.job_content,
            "status": "extracting",
            "note": (
                f"Content delivered ({len(state.job_content or '')} chars)"
            ),
            "detected_posting_status": state.detected_posting_status,
            "detected_closed_evidence": state.detected_closed_evidence,
        }
        # Only send `html` when we actually captured a non-empty DOM.
        # Sending an empty html would clobber a DOM persisted on an earlier
        # run (lease-sweep re-dispatch) or by the debug-artifact backfill.
        # Composes with the api anti-clobber guard (which drops a falsy
        # `html` from PATCHes) without relying on it.
        if state.html:
            attributes["html"] = state.html

        try:
            httpx.patch(
                f"{_api_base()}/api/v1/scrapes/{state.scrape_id}/",
                json={
                    "data": {
                        "type": "scrape",
                        "id": str(state.scrape_id),
                        "attributes": attributes,
                    }
                },
                headers={**_api_headers(), "Content-Type": "application/json"},
                timeout=30.0,
            )
        except Exception:
            logger.warning("PersistScrape: patch failed", exc_info=True)
        # skip_extract scrapes (staff "Resolve & dedupe" path) bypass
        # the extraction chain: page is loaded, redirect resolved by
        # ResolveFinalUrl, dup check ran in CheckLinkDedup. The only
        # remaining work is apply-URL capture before End. The
        # originating JobPost's title/description/etc. are intentionally
        # left untouched.
        if state.skip_extract:
            trace_node(state, "PersistScrape", "ResolveApplyUrl", started)
            return nodes_extract.ResolveApplyUrl()
        trace_node(state, "PersistScrape", "StartExtract", started)
        return nodes_extract.StartExtract()


@dataclass
class SkipBrowserTier(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    """Fast-path entry for source_mode='extension-direct' scrapes.

    The extension content-script already extracted title + company +
    description from the user-rendered DOM (and optionally apply_url +
    location + extraction_hints). This node copies those fields onto
    state.parsed (the ParsedJobData shape PersistJobPost POSTs to the
    api's /persist-extraction/ endpoint) so the dedupe-first invariant
    is preserved — the api's JobPostExtractor.parse_scrape still owns
    canonical_link / fingerprint / sticky-closed handling. No browser
    tier nodes run.

    Routing — per the canonical plan ("Calls ResolveApplyUrl ONLY if
    captured_payload.apply_url is null"):
    - apply_url present in payload → route to PersistJobPost directly
      (the apply_url rides through as parsed['link'], and the JobPost's
      apply_url column is populated by ParsedJobData persistence).
    - apply_url null → route to ResolveApplyUrl, which on the fast path
      no-ops (no browser page) and chains into PersistJobPost so the
      JobPost still lands.

    PATCHes /scrapes/:id/ to status='extracting' before routing so the
    state machine matches the browser-tier path's transition into the
    persistence phase.
    """

    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> "Union[PersistJobPost, ResolveApplyUrl]":  # noqa: F821 — resolved via graph.py namespace
        from . import nodes_extract
        started = time.time()
        state = ctx.state
        payload = state.captured_payload or {}

        # Map captured_payload → ParsedJobData. The api's ParsedJobData
        # uses company_name / link / description; the extension's
        # payload uses company / apply_url for clarity. Translate here
        # so persist-extraction never sees the extension's vocabulary.
        parsed: dict = {
            "title": (payload.get("title") or "").strip(),
            "company_name": (payload.get("company") or "").strip(),
            "description": (payload.get("description") or "").strip(),
        }
        if payload.get("location"):
            parsed["location"] = payload["location"]
        apply_url = payload.get("apply_url")
        if apply_url:
            # ParsedJobData stores the apply destination under `link`.
            parsed["link"] = apply_url
        state.parsed = parsed

        # PATCH the scrape status to 'extracting' so the row reflects
        # progress past the (skipped) capture phase. Mirrors what
        # PersistScrape does for the browser-tier path; the status
        # state machine still expects this transition before
        # /persist-extraction/ writes the JobPost.
        try:
            httpx.patch(
                f"{_api_base()}/api/v1/scrapes/{state.scrape_id}/",
                json={
                    "data": {
                        "type": "scrape",
                        "id": str(state.scrape_id),
                        "attributes": {
                            "status": "extracting",
                            "note": (
                                "extension-direct fast path: captured "
                                f"{len(parsed.get('description') or '')} desc chars"
                            ),
                        },
                    }
                },
                headers={**_api_headers(), "Content-Type": "application/json"},
                timeout=15.0,
            )
        except Exception:
            logger.warning(
                "SkipBrowserTier: status PATCH failed scrape_id=%s",
                state.scrape_id, exc_info=True,
            )

        # Branch: apply_url known → straight to PersistJobPost.
        # apply_url null → ResolveApplyUrl (which on fast path no-ops
        # because there's no browser page, then chains to PersistJobPost).
        if apply_url:
            trace_node(
                state, "SkipBrowserTier", "PersistJobPost", started,
                {
                    "source_mode": state.source_mode,
                    "payload_fields": sorted(payload.keys()),
                    "apply_url_present": True,
                    "routed_through_resolve_apply": False,
                },
            )
            return nodes_extract.PersistJobPost()
        trace_node(
            state, "SkipBrowserTier", "ResolveApplyUrl", started,
            {
                "source_mode": state.source_mode,
                "payload_fields": sorted(payload.keys()),
                "apply_url_present": False,
                "routed_through_resolve_apply": True,
            },
        )
        return nodes_extract.ResolveApplyUrl()


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
    "SkipBrowserTier",
]
