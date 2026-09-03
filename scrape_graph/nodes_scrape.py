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

from browser.resident import is_driver_closed
from .landing_page_detector import detect_landing_page
from .state import ScrapeGraphState
from .tracing import trace_node
from .url_canonicalize import apply_url_rewrites, canonicalize_url, urls_differ

logger = logging.getLogger(__name__)

# CC-226. The distinct terminal note for a page that is a search-landing /
# interstitial rather than a job detail page. Deliberately NOT the generic
# "graph run exceeded 240s cap" the runner writes on a timeout, nor
# ExtractFail's "extraction" — the whole point of the ticket is that these
# are queryable apart from real timeouts and real extraction failures.
_LANDING_PAGE_FAILURE_REASON = "landing_page_not_detail"


def _reraise_if_driver_closed(exc: BaseException) -> None:
    """Re-raise ``exc`` when it is a Playwright driver-connection-dead error.

    CC-160: a Camoufox/Playwright driver death that happens MID-scrape (after
    ``open_tab`` succeeded — the seam CC-141's open_tab guard can't cover)
    surfaces as "Connection closed while reading from the driver" on the next
    Playwright call, whichever node is running. The browser-tier nodes below
    catch every per-selector / per-op exception best-effort (a page that just
    hasn't hydrated a selector yet is normal), which means a dead-driver error
    would otherwise be swallowed as one more "not matched": the graph marches
    over a dead page, captures 0 bytes, and terminates at ``ExtractFail`` →
    the row is wrongly marked ``failed`` (and the ExtractFail screenshot fires
    against a corpse → no screenshot, no DOM).

    Call this at the TOP of every ``except`` that wraps a Playwright call. A
    driver-closed error propagates out of the node and out of
    ``run_scrape_graph``; the runner's ``_run_graph`` already treats it as
    infra-death — relaunch (CC-141 ``open_tab``) + re-queue the scrape as
    ``hold`` (never ``failed``) — so the eventual real failure happens on a
    live, re-navigated page where the screenshot invariant can actually fire.

    A genuine "no selector matched after 30s" (or any non-driver error) does
    NOT carry the marker phrase, so it is left to be swallowed as before and
    routes onward to SettleWait/…/ExtractFail exactly like today.
    """
    if is_driver_closed(exc):
        raise exc

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


class LandingPageFail(BaseNode[ScrapeGraphState, None, dict]):
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
                # CC-160: a driver death at goto is infra-death, not a
                # navigation content failure — propagate so the runner
                # relaunches + re-queues hold instead of marching on.
                _reraise_if_driver_closed(exc)
                state.failure_reason = f"navigate_failed: {exc}"
        from . import nodes_obstacle
        trace_node(state, "Navigate", "DetectObstacle", started)
        return nodes_obstacle.DetectObstacle()


def _propagate_canonical_to_parent_jp(
    parent_scrape_id: str, landed_url: str
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

    Takes the RAW landed URL and lets the api canonicalize it. This
    module has its own canonicalizer whose rules DIFFER from the api's
    — the param sets are disjoint, not nested, it applies no
    ScrapeProfile url_rewrites, and it strips `src`, which the api
    deliberately keeps. Sending our form put values into the primary
    dedupe key that the api's own matcher would never reproduce for the
    same input. The api canonicalizes an inbound canonical_link at
    write, so there is exactly one owner of those rules and it is not
    this file.

    Behavior:
    - GET the parent scrape; if no `job_post` relationship, no-op.
    - GET the JobPost; if its canonical_link already equals what we are
      about to send, no-op.
    - Otherwise PATCH the JobPost's `canonical_link`. The api's PATCH
      path leaves `link` alone (audit / wrapper preserved).

    NOTE on the idempotence check: the stored value is canonical and
    `landed_url` is raw, so they now compare equal only when the URL
    was already canonical. A re-run on a URL that needed normalizing
    therefore re-PATCHes with the same end result rather than
    short-circuiting — one redundant, idempotent write on a
    best-effort path. Reproducing the api's canonicalization here to
    avoid it would reintroduce exactly the divergence this change
    removes.

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
        if current_canonical == landed_url:
            return

        httpx.patch(
            f"{_api_base()}/api/v1/job-posts/{jp_id}/",
            json={
                "data": {
                    "type": "job-post",
                    "id": str(jp_id),
                    "attributes": {"canonical_link": landed_url},
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
                        # RAW landed URL, not our canonicalized form. This
                        # becomes the child scrape's url and then JobPost.link,
                        # which the api treats as the untouched original and
                        # exact-matches on in four places. Sending our version
                        # put an agents-shaped string into that column: this
                        # module strips `src`, which the api DELIBERATELY keeps
                        # (worksourcewa encodes part of the job id there — see
                        # api job_post_dedupe._TRACKING_PARAMS), so distinct
                        # jobs could collapse on the way in.
                        "url": landed,
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
                # Send the RAW landed URL. The api canonicalizes an inbound
                # canonical_link at write now, so it owns the rules and this
                # side does not have to know them.
                _propagate_canonical_to_parent_jp(state.scrape_id, landed)
                # Scrape ids are NanoID strings (CC-77) — int() would raise
                # ValueError, get swallowed by the except below, and the
                # child-scrape swap would silently fail (the child's outcome
                # then misattributed to the parent row).
                state.scrape_id = new_id
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
        # Send the RAW landed URL, not our canonicalized form. The api
        # canonicalizes the filter value itself and builds a four-leg OR
        # (link, apply_url, canonical_link, canonicalized apply_url), so
        # handing it a pre-canonicalized string meant the composition was
        # api(agents(u)) rather than api(u) — and since the stored `link` is
        # the untouched original, the exact-`link` leg could then only match
        # by luck. The two param sets are disjoint, not nested, so neither is
        # a superset of the other.
        canonical = state.final_url or state.submitted_url
        non_stub_id: str | None = None
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
                    # JobPost id is a NanoID string (CC-77); int() would
                    # raise, get swallowed by the bare except below, and
                    # DuplicateShortCircuit would never fire — re-creating a
                    # duplicate JobPost and violating dedupe-first.
                    non_stub_id = row["id"]
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
                        # CC-160: a mid-scrape driver death surfaces here as a
                        # per-selector wait_for exception. Re-raise it BEFORE
                        # recording it as "not matched" so the runner can
                        # relaunch + re-queue hold instead of marching over a
                        # dead page to a wrong `failed`. Non-driver errors
                        # (timeout, parse) fall through and are recorded.
                        _reraise_if_driver_closed(exc)
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
                        # CC-226: record the hit on state so Capture's
                        # landing-page guard knows the detail anchor
                        # appeared and must not refuse the page.
                        state.matched_ready_selector = sel
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
                    except Exception as exc:
                        # CC-160: don't let a mid-scroll driver death look like
                        # a benign scroll error — propagate so the runner
                        # relaunches + re-queues hold.
                        _reraise_if_driver_closed(exc)
                        break
                    await asyncio.sleep(_SCROLL_TICK_MS / 1000.0)
                    for sel in selectors:
                        try:
                            handle = await page.query_selector(sel)
                            if handle is not None:
                                matched = sel
                                # CC-226: a late hydration hit still means
                                # this is a detail page — tell Capture.
                                state.matched_ready_selector = sel
                                break
                        except Exception as exc:
                            _reraise_if_driver_closed(exc)
                    if matched:
                        break
                    try:
                        height = await page.evaluate(
                            "document.body.scrollHeight"
                        )
                    except Exception as exc:
                        _reraise_if_driver_closed(exc)
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
                except Exception as exc:
                    _reraise_if_driver_closed(exc)
                    final_y = 0
                await asyncio.sleep(_SCROLL_POST_SETTLE_MS / 1000.0)
            except Exception as exc:
                # Outer guard: a driver-closed error re-raised by any inner
                # handler must not be re-swallowed here (CC-160).
                _reraise_if_driver_closed(exc)
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
            except Exception as exc:
                _reraise_if_driver_closed(exc)  # CC-160
                logger.debug("ExpandTruncations failed", exc_info=True)
        trace_node(ctx.state, "ExpandTruncations", "Capture", started)
        return Capture()


@dataclass
class Capture(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> Union["DetectClosedState", "LandingPageFail"]:
        started = time.time()
        state = ctx.state
        page = getattr(state, "_browser_page", None)
        canonical_trace: dict = {}
        if page:
            # CC-248: read the page's own declared canonical while the DOM
            # is in hand. Pure querySelector work — no extra fetch, no
            # LLM, no per-host config on the common path.
            #
            # ORDER IS LOAD-BEARING — this runs BEFORE the content reads,
            # not after. `_adopt_declared_canonical` re-raises a
            # driver-closed error (the CC-160 convention for every except
            # that wraps a Playwright call), and CC-160's contract is
            # "fail BEFORE we bank a capture, so the runner relaunches and
            # re-queues the scrape as hold". Below the reads, the same
            # raise throws away a complete, usable `job_content` + `html`
            # and pays for a whole re-scrape — a raise site Capture did
            # not have before CC-248 (`_screenshot_and_upload` never
            # raises by contract, `_discover_selectors` swallows
            # everything). Above the reads it adds no new failure mode at
            # all: any driver death that kills this query_selector kills
            # `page.inner_text` on the very next line.
            canonical_trace = await _adopt_declared_canonical(page, state)
            try:
                state.job_content = await page.inner_text("body")
                state.html = await page.content()
            except Exception as exc:
                # CC-160: a driver death between ResolveFinalUrl and here
                # makes both reads throw. Propagate rather than persist a
                # 0-byte capture and march to ExtractFail — the runner
                # relaunches + re-queues hold so the retry captures a live
                # page (and the eventual real ExtractFail screenshots it).
                _reraise_if_driver_closed(exc)
                state.failure_reason = f"capture_failed: {exc}"

            await _screenshot_and_upload(page, state)
            await _discover_selectors(page, state)

        # CC-226 — search-landing / interstitial fast-fail. The screenshot
        # above is already banked, so the post-mortem is intact; what we
        # are cutting is everything AFTER this point on a page that will
        # never yield a posting: DetectClosedState's LLM leg, a full-DOM
        # PATCH, and the tier ladder whose two LLM rungs carry 120s
        # timeouts each and can reach the runner's 240s graph cap on their
        # own. Costs one regex pass over text we already hold.
        selectors = _normalize_ready_selectors(
            (state.profile or {}).get("ready_selector")
        )
        landing = detect_landing_page(
            state.job_content or "",
            url=state.final_url or state.canonical_url or state.submitted_url or "",
            ready_selector_configured=bool(selectors),
            ready_selector_matched=bool(state.matched_ready_selector),
        )
        capture_payload = {
            "canonical_source": state.canonical_source,
            "canonical_url": state.canonical_url,
            "matched_ready_selector": state.matched_ready_selector,
            **canonical_trace,
        }
        if landing:
            state.failure_reason = _LANDING_PAGE_FAILURE_REASON
            trace_node(
                state, "Capture", "LandingPageFail", started,
                {**capture_payload, "landing_page": landing},
            )
            return LandingPageFail()

        # DetectClosedState runs while DOM is still live so the CSS path
        # can probe the page; passes through to PersistScrape regardless
        # of verdict (closed-state is metadata, never terminal).
        trace_node(
            state, "Capture", "DetectClosedState", started, capture_payload,
        )
        return DetectClosedState()


# ---------------------------------------------------------------------------
# CC-248 — the declared-canonical ladder.
# ---------------------------------------------------------------------------

# Generic path segments that mean "you are at the front door, not at a
# posting". A declared canonical whose path contains one of these is an
# auth-wall or account shell declaring ITSELF, not the job we came for —
# adopting it would replace a usable identity with a URL that identifies
# nothing.
#
# This is a CLOSED SET OF WEB CONVENTIONS applied uniformly to every host,
# the same shape as `_STRIP_EXACT` in url_canonicalize.py. It is NOT
# per-host tuning: no hostname appears anywhere in this module's
# canonical code, and adding one here would be the signal to stop and ask
# rather than to extend the set.
#
# Matched on WHOLE, lowercased path segments so ordinary job slugs that
# merely contain one of these words survive — "/jobs/account-manager" and
# "/careers/registered-nurse" are unaffected.
_AUTH_PATH_SEGMENTS = frozenset({
    "login", "signin", "sign-in", "sign_in",
    "signup", "sign-up", "sign_up",
    "register", "registration",
    "auth", "authwall", "authenticate", "oauth", "sso",
    "account", "session", "logout", "signout", "sign-out",
})


def _host_key(url: str) -> str:
    """Lowercased, `www.`-stripped hostname — the repo-wide convention for
    comparing hosts (LoadProfile does the same strip before the profile
    lookup, so a profile keyed `example.com` serves `www.example.com`).
    Returns "" when the URL has no parseable host.
    """
    from urllib.parse import urlparse
    try:
        host = (urlparse(url or "").hostname or "").lower()
    except ValueError:
        return ""
    return host.removeprefix("www.")


def _declared_canonical_is_sane(candidate: str, landed: str) -> tuple[bool, str]:
    """Junk filter for a page-declared canonical. Returns (ok, reason).

    A declaration is untrusted input: we did not compute it, and job
    boards routinely emit garbage — an SPA shell with a hard-coded
    `<link rel="canonical" href="https://site.com/">`, an auth-wall
    declaring its own login URL, a syndicated listing pointing at an
    aggregator on another host. Adopting any of those would destroy the
    only identity we had (the resolved URL), so a declaration is only
    allowed to win when it is at least as SPECIFIC as what we already
    hold.

    Three host-agnostic gates:

    1. absolute, with an http/https scheme. Rules out `javascript:`,
       `about:blank`, mailto:, and any relative leftover that urljoin
       failed to resolve.
    2. same host as `landed`. A cross-host declaration is a syndication
       pointer, not this document's identity: we never fetched that URL
       and cannot verify it names this job. This gate also has a second,
       quieter job — `UpdateProfile` derives the profile host key from
       `state.canonical_url` (nodes_extract.py:903), so pinning the host
       guarantees an adopted declaration can never cause profile
       learning to be written under the wrong hostname.
    3. path is non-empty, is not `/`, and contains no `_AUTH_PATH_SEGMENTS`
       segment. A host root cannot identify a posting; a login path
       identifies the wall, not what is behind it.

    Deliberately NOT gated: same-host SEARCH/listing pages
    (`/jobs?q=engineer`). Recognizing those means guessing at path
    vocabulary per site, which is exactly the bespoke per-host work this
    ticket forbids, and the failure is soft — a listing URL is still the
    right host and still round-trips. If it ever bites, it is a ruling to
    get, not a heuristic to sneak in.
    """
    from urllib.parse import urlparse
    if not candidate:
        return False, "empty"
    try:
        parsed = urlparse(candidate)
    except ValueError:
        return False, "unparseable"
    if parsed.scheme not in ("http", "https"):
        return False, "bad_scheme"
    cand_host = _host_key(candidate)
    if not cand_host:
        return False, "no_host"
    landed_host = _host_key(landed)
    if not landed_host or cand_host != landed_host:
        return False, "cross_host"
    path = parsed.path or ""
    if not path.strip("/"):
        return False, "host_root"
    segments = {seg.lower() for seg in path.split("/") if seg}
    if segments & _AUTH_PATH_SEGMENTS:
        return False, "auth_path"
    return True, "ok"


# Rungs 1 and 2 of the ladder — the two web standards, cheapest first.
# `(selector, attributes to try in order, source label)`.
#
# `rel~="canonical"`, NOT `rel="canonical"`: `rel` is a
# space-separated TOKEN LIST, and `<link rel="canonical alternate">` is
# both legal and real on syndicated / i18n pages. `~=` is the CSS
# operator for "contains this token"; `=` is an exact whole-value match
# that silently misses the multi-token form. Playwright's selector
# parser accepts `~=` (driver `lib/utils/isomorphic/selectorParser.js`
# allows `=, *=, ^=, $=, |=, ~=`), and `~=` matches everything `=`
# matched, so this is a strict widening.
_STANDARD_CANONICAL_RUNGS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ('link[rel~="canonical"]', ("href",), "link_rel"),
    ('meta[property="og:url"]', ("content",), "og_url"),
)


async def _read_declared_canonical(
    page, profile: dict | None, landed: str
) -> tuple[tuple[str, str] | None, list[dict]]:
    """Walk the canonical ladder cheapest rung first. Returns
    ``(accepted, rejections)``.

    ``accepted`` is ``(absolute_url, source)`` for the first rung whose
    declaration SURVIVES ``_declared_canonical_is_sane``, else ``None``.
    ``rejections`` is the audit trail — one
    ``{"source", "reason", "candidate"}`` dict per rung that produced a
    value we then threw away — so a host being systematically rejected
    shows up on the trace instead of only at ``logger.debug``.

    1. ``link[rel~="canonical"]``  — a web standard, zero config, every host.
    2. ``meta[property="og:url"]`` — same.
    3. ``ScrapeProfile.extension_selectors.canonical_link_selectors`` — the
       escape hatch, reached only when 1 and 2 yield nothing USABLE.

    THE GATE RUNS PER RUNG, INSIDE THE LADDER, and a rejected rung falls
    through to the next one. This is the whole reason there are three
    rungs. The escape hatch exists for a host that "emits neither
    standard tag OR EMITS A WRONG ONE", and the wrong-one half is the
    common case: an SPA shell with a hard-coded
    ``<link rel="canonical" href="https://site/">`` on every page is
    precisely the motivating anti-pattern for this ticket. Gate the
    ladder's OUTPUT once at the caller instead and that junk rung 1
    short-circuits rungs 2 and 3 — the operator's profile entry can
    never rescue the exact case they configured it for. Reviewed and
    fixed on CC-248 before merge; do not "simplify" the gate back out to
    the caller.

    Exceptions follow the same rule: every rung reads inside its own
    try/except, so one throwing selector cannot blind the rungs below
    it. A driver-closed error still propagates (CC-160) — that is infra
    death, not a bad selector, and every rung below it would throw too.

    On rung 3, `extension_selectors` is read NESTED off the profile, not
    flattened: `_flatten_profile_attrs` lifts the keys of `css_selectors`
    only, and the api serializes attribute names with underscores
    (BaseSerializer.to_resource emits the raw field names), so the bundle
    arrives at `state.profile["extension_selectors"]`. VERIFIED against
    api serializers.py ScrapeProfileSerializer.attributes.
    """
    from urllib.parse import urljoin

    async def _first_attr(selector: str, attrs: tuple[str, ...]) -> str:
        el = await page.query_selector(selector)
        if not el:
            return ""
        for attr in attrs:
            value = await el.get_attribute(attr)
            if value and value.strip():
                return value.strip()
        return ""

    def _rungs():
        """Lazy so rung 3 is never even assembled when rung 1 wins."""
        yield from _STANDARD_CANONICAL_RUNGS
        extension = (profile or {}).get("extension_selectors")
        selectors = (extension or {}).get("canonical_link_selectors") or []
        if isinstance(selectors, str):
            selectors = [selectors]
        for selector in selectors:
            if not isinstance(selector, str) or not selector.strip():
                continue
            yield selector.strip(), ("href", "content"), "profile_selector"

    rejections: list[dict] = []
    for selector, attrs, source in _rungs():
        try:
            raw = await _first_attr(selector, attrs)
        except Exception as exc:
            _reraise_if_driver_closed(exc)  # CC-160
            # One throwing selector must not blind the rungs below it.
            logger.debug(
                "Capture: canonical rung failed sel=%s source=%s",
                selector, source, exc_info=True,
            )
            continue
        if not raw:
            continue
        try:
            # Relative hrefs are legal in link[rel=canonical]; resolve
            # against the landed URL before the host gate can see them,
            # or every `href="/jobs/123"` would fail as "no_host".
            candidate = urljoin(landed, raw)
        except ValueError:
            rejections.append(
                {"source": source, "reason": "unjoinable", "candidate": raw}
            )
            continue
        ok, reason = _declared_canonical_is_sane(candidate, landed)
        if ok:
            return (candidate, source), rejections
        logger.debug(
            "Capture: rejected declared canonical source=%s reason=%s "
            "candidate=%s landed=%s",
            source, reason, candidate, landed,
        )
        rejections.append(
            {"source": source, "reason": reason, "candidate": candidate}
        )
    return None, rejections


async def _adopt_declared_canonical(page, state: ScrapeGraphState) -> dict:
    """Overwrite ``state.canonical_url`` with the page's OWN declared
    canonical when the page makes one and it survives the junk filter.

    Returns a (possibly empty) dict of extras for Capture's trace
    payload. ``canonical_source`` alone reads "resolved" identically for
    "the page declared nothing" and "the page declared something we
    threw away", so a host being systematically rejected by
    ``_AUTH_PATH_SEGMENTS`` would be invisible in production — the
    rejections ride here as ``canonical_rejected`` so the trace can
    answer that without a re-scrape.

    PRECEDENCE — the declaration wins over the resolved URL. Reasoning,
    because this is the one open design question on CC-248:

    ``ResolveFinalUrl`` computes a TRANSPORT fact — where the browser
    ended up after the redirect chain, minus known tracker params. It is
    an excellent answer to "what did we fetch" and a mediocre answer to
    "what job is this", because it cannot know which of the surviving
    query params are identity and which are presentation:
    `/jobs/view/123?position=2&pageNum=0` and `/jobs/view/123` are one
    posting and `_STRIP_EXACT` has no way to learn that.
    ``link[rel="canonical"]`` is an IDENTITY fact — the URL the site
    itself publishes as the name of this document. Everything downstream
    that consumes `JobPost.canonical_link` (the api's duplicate-candidate
    scoring) is asking the identity question, so the identity signal is
    the one worth storing. When the two disagree, the site is the better
    authority about its own URL space.

    The gates in ``_declared_canonical_is_sane`` are what make "declared
    wins" safe rather than reckless: the declaration only wins when it is
    at least as specific as the resolved URL we would be discarding.

    Comparison is against the LANDED url (``final_url or submitted_url``),
    never the SUBMITTED one. After a tracker redirect the landed host is
    the real host; comparing to the submitted tracker host would reject
    every legitimate declaration on precisely the redirect scrapes this
    work exists for. Do not let this get "simplified".

    A missing or garbage canonical is the normal case for most of the
    web — on any rejection or ordinary error ``state.canonical_url`` is
    left exactly as ``ResolveFinalUrl`` set it and
    ``state.canonical_source`` stays "resolved". The ONE exception is a
    driver-closed error, which propagates per CC-160; Capture calls this
    BEFORE the content reads precisely so that raise costs us nothing
    (see the comment at the call site).
    """
    landed = state.final_url or state.submitted_url or ""
    try:
        accepted, rejections = await _read_declared_canonical(
            page, state.profile, landed,
        )
    except Exception as exc:
        _reraise_if_driver_closed(exc)  # CC-160
        logger.debug("Capture: declared-canonical read failed", exc_info=True)
        return {"canonical_read_failed": True}
    extras: dict = {}
    if rejections:
        extras["canonical_rejected"] = rejections
    if accepted is None:
        return extras
    absolute, source = accepted
    # Adopt the declaration AS THE PAGE STATED IT. This used to run through
    # this module's canonicalize_url first, and the result was PATCHed
    # straight onto JobPost.canonical_link by ReviewCompleteness — putting an
    # agents-shaped value into the api's primary dedupe key, which its own
    # matcher would never reproduce for the same input (disjoint param sets,
    # no ScrapeProfile url_rewrites, and we strip `src` where the api
    # deliberately keeps it). The api canonicalizes an inbound canonical_link
    # at write, so it owns those rules and this file no longer guesses at them.
    #
    # Safe for the other readers: outside the persist hop and the trace
    # payload, `state.canonical_url` is only ever read for its hostname
    # (nodes_extract.py, nodes_obstacle.py), and canonicalization never
    # touched the host. The sanity gates already ran on this candidate,
    # per-rung, inside _read_declared_canonical.
    state.canonical_url = absolute
    state.canonical_source = source
    logger.info(
        "Capture: adopted declared canonical source=%s url=%s",
        source, state.canonical_url,
    )
    return extras


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
class LandingPageFail(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    """Terminal: the page we landed on is a search-listing / interstitial,
    not a job detail page (CC-226).

    Same shape as ObstacleFail — snapshot first, then finalize — with one
    difference that matters: Capture has already banked a screenshot, and
    PersistScrape never ran, so `scrape.html` is still empty. That is
    exactly the case `capture_debug_artifact`'s write-to-empty branch
    exists for, which means the admin UI's "view raw html" link works on
    these rows without the full-DOM PATCH the happy path pays for.

    The failure_reason is its own string rather than a reuse of
    "extraction" or the runner's timeout note, because the ticket's ask is
    that these become queryable APART from real timeouts: before this
    node, a landing page and a genuinely stuck node were the same row.
    """

    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> End[dict]:
        from ._artifacts import capture_debug_artifact
        started = time.time()
        state = ctx.state
        state.outcome = "failure"
        state.failure_reason = state.failure_reason or _LANDING_PAGE_FAILURE_REASON

        page = getattr(state, "_browser_page", None)
        artifact_info: dict = {}
        try:
            artifact_info = await capture_debug_artifact(
                page, state, reason="landing_page",
            )
        except Exception:
            logger.warning(
                "LandingPageFail: debug artifact capture failed scrape_id=%s",
                state.scrape_id, exc_info=True,
            )

        _patch_scrape_status(state.scrape_id, "failed", note=state.failure_reason)
        trace_node(
            state, "LandingPageFail", "End", started,
            payload=artifact_info or None,
        )
        return End({
            "outcome": "failure",
            "failure_reason": state.failure_reason,
            "scrape_id": state.scrape_id,
        })


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
        # `html` from PATCHes) without relying on it. Cap at _MAX_DOM_BYTES
        # via the same shared helper the Fail path uses — a LinkedIn DOM is
        # multi-MB and the success path would otherwise PATCH it uncapped.
        # Covers both the Capture-set html and the re-grab above, since both
        # land in state.html before this gate.
        if state.html:
            from ._artifacts import truncate_dom
            attributes["html"] = truncate_dom(state.html)

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


def _patch_scrape_status(scrape_id: str, status: str, note: str | None = None) -> None:
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
    "LandingPageFail",
    "PersistScrape",
    "SkipBrowserTier",
]
