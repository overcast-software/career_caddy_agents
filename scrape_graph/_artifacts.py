"""Shared debug-artifact helper: snapshot the page for any scrape outcome.

Two callers:
- ``capture_debug_artifact`` — invoked by every Fail terminal
  (``ObstacleFail`` / ``ExtractFail``) BEFORE ``_patch_scrape_status``.
  Snapshots the page + DOM so the admin UI has something to render
  post-mortem.
- ``Capture._screenshot_and_upload`` (nodes_scrape.py) — happy-path
  capture so the /scrapes/:id/screenshots/ viewer keeps working when
  the scrape succeeds.

Both call ``upload_page_screenshot`` underneath. Until 2026-05-05 the
happy path had its own disk-round-trip implementation that imported
``mcp_servers.browser_server.SCREENSHOT_DIR`` and called
``page.screenshot(full_page=True)`` with no timeout — Playwright's 30s
default fired on slow LinkedIn pages and the screenshot never landed
(scrapes 320, 321, 323, 324, 325 all silently dropped). The shared
helper uses ``full_page=False`` + ``timeout=5_000`` so the same
viewport-fast snap that worked on the failure path now works on the
happy path too. See notes.org Operations/Scrape Log 2026-05-05.

Both paths are tolerant of detached pages, closed browsers, upload
errors — never raises, never blocks the caller.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)


_MAX_DOM_BYTES = 200_000
_DOM_TRUNCATION_TRAILER = "\n<!-- [truncated] -->"
_DOM_NOISE_STRIPPED_TRAILER = (
    "\n<!-- [script/style stripped to fit the capture cap] -->"
)
# Loud marker for the one outcome that used to be silent: the cap landing
# before <body>, so the persisted DOM is a <head> and nothing else. Probing
# for it is a substring check on the stored html — no size arithmetic.
# The marker text deliberately spells the tag out rather than writing
# "<body>", so that probing the stored html for "<body" (here and in the
# enhancer) cannot match the marker announcing the tag's absence.
_DOM_BODY_LOST_TRAILER = (
    "\n<!-- [truncated BEFORE the opening BODY tag"
    " — NO body markup persisted] -->"
)

# Hard ceiling on what we hand the api, matching the historical maximum
# (`_MAX_DOM_BYTES` + the truncation trailer = 200_021 bytes) so the 200KB
# column expectation from CC-134 holds no matter which trailer we append.
_MAX_PERSISTED_BYTES = _MAX_DOM_BYTES + len(_DOM_TRUNCATION_TRAILER)


def _has_body_markup(html: str) -> bool:
    """True when ``html`` reaches the opening ``<body>`` tag."""
    return "<body" in html.lower()


def truncate_dom(dom: str, *, scrape_id: object = None) -> str:
    """Cap a captured DOM at ``_MAX_DOM_BYTES``, keeping ``<body>`` if we can.

    Shared by every path that persists ``scrape.html`` — the Fail-path
    debug-artifact backfill (``capture_debug_artifact`` below) and the
    success-path persist (``nodes_scrape.PersistScrape``). A LinkedIn /
    Cloudflare DOM can be multiple MB; without a cap the success path
    would PATCH the whole thing on every scrape. Keeping the cap + the
    trailer convention in one place means both paths stay in lockstep.

    The cap is a PREFIX slice, which on a head-heavy host threw the whole
    document away: jobright.ai is a Next.js app whose 27 stylesheet links
    plus inline Ant Design / emotion critical CSS blow 200KB before
    ``<body>`` is ever reached, so the persisted DOM held zero ``div`` /
    ``a`` / ``button`` elements while ``html_size_bytes`` reported a
    healthy 200,021. Nothing said so — every selector probe just returned
    ``match_count: 0``. That is PACA CC-284.

    So an oversized DOM now takes three steps, not one:

    1. Strip ``<script>`` / ``<style>`` (JSON-LD kept — see
       ``lib.scrape_inspector.strip_dom_noise``). This is the read-time
       prune behind ``inspect_scrape_html`` moved earlier, and it takes
       jobright from 200KB+ to a few KB, so the entire body fits.
    2. If the stripped DOM fits, persist all of it.
    3. Only if it STILL does not fit, slice — and if the slice lands
       before ``<body>``, say so loudly: a distinct trailer on the stored
       html and a WARNING log line, rather than a healthy-looking row.

    Under-cap DOMs are returned byte-for-byte unchanged, as before; the
    strip only ever runs on documents that would otherwise be cut.
    """
    if len(dom) <= _MAX_DOM_BYTES:
        return dom

    original_len = len(dom)
    try:
        from lib.scrape_inspector import strip_dom_noise

        stripped = strip_dom_noise(dom)
    except Exception:
        # Best-effort: a bs4 parse failure must never cost us the capture.
        logger.warning(
            "truncate_dom: noise strip failed scrape_id=%s", scrape_id,
            exc_info=True,
        )
        stripped = dom

    if len(stripped) <= _MAX_DOM_BYTES - len(_DOM_NOISE_STRIPPED_TRAILER):
        logger.info(
            "truncate_dom: script/style strip fit the cap scrape_id=%s "
            "%d -> %d bytes",
            scrape_id, original_len, len(stripped),
        )
        return stripped + _DOM_NOISE_STRIPPED_TRAILER

    dom = stripped if len(stripped) < original_len else dom
    cut = dom[:_MAX_DOM_BYTES]
    if _has_body_markup(cut):
        return cut + _DOM_TRUNCATION_TRAILER

    logger.warning(
        "truncate_dom: cap landed BEFORE the opening BODY tag scrape_id=%s "
        "— persisting a head-only DOM "
        "(%d bytes captured, %d after script/style strip). "
        "Tier-0 selector work on this host is impossible until the pre-body "
        "markup shrinks. See PACA CC-284.",
        scrape_id, original_len, len(stripped),
    )
    # Reserve room for the longer trailer so the persisted value never
    # exceeds the historical `_MAX_PERSISTED_BYTES` ceiling.
    cut = dom[: _MAX_PERSISTED_BYTES - len(_DOM_BODY_LOST_TRAILER)]
    return cut + _DOM_BODY_LOST_TRAILER


def _api_base() -> str:
    return os.environ.get("CC_API_BASE_URL", "").rstrip("/")


def _api_headers() -> dict[str, str]:
    token = os.environ.get("CC_API_TOKEN", "")
    return {"Authorization": f"Bearer {token}"} if token else {}


async def upload_page_screenshot(
    page,
    state,
    *,
    reason: str | None = None,
    full_page: bool = False,
    timeout_ms: int = 5_000,
) -> str | None:
    """Snap a screenshot in-memory and POST to /api/v1/scrapes/:id/screenshots/.

    Returns the uploaded filename on success, None on any failure
    (screenshot exception, upload exception, non-2xx response, missing
    page or scrape_id). Records ``state.screenshot_name`` on success so
    ``graph_payload`` can reference it.

    Best-effort: never raises. Logs warnings on failure.

    Filename schema:
        with reason → ``{host}_{reason}_{YYYYMMDD_HHMMSS}.png``
        without    → ``{host}_{YYYYMMDD_HHMMSS}.png``

    ``full_page`` defaults to False and ``timeout_ms`` defaults to 5s
    deliberately — Playwright's default 30s "wait for fonts" stalls on
    LinkedIn / Cloudflare-gated pages where the page DOM never settles.
    Both callers (failure path AND Capture's happy path) use these
    defaults; the LinkedIn login-wall pages were the case that
    motivated the unification.
    """
    if page is None or not getattr(state, "scrape_id", None):
        return None

    host = (
        urlparse(state.canonical_url or state.submitted_url or "").hostname
        or "unknown"
    ).lower()
    if host.startswith("www."):
        host = host[4:]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{host}_{reason}_{ts}.png" if reason else f"{host}_{ts}.png"

    try:
        png_bytes: bytes | None = await page.screenshot(
            full_page=full_page, timeout=timeout_ms,
        )
    except Exception:
        logger.warning(
            "upload_page_screenshot: screenshot failed scrape_id=%s reason=%s",
            state.scrape_id, reason, exc_info=True,
        )
        return None

    if not png_bytes:
        return None

    try:
        resp = httpx.post(
            f"{_api_base()}/api/v1/scrapes/{state.scrape_id}/screenshots/",
            files={"file": (name, png_bytes, "image/png")},
            headers=_api_headers(),
            timeout=30.0,
        )
    except Exception:
        logger.warning(
            "upload_page_screenshot: upload exception scrape_id=%s",
            state.scrape_id, exc_info=True,
        )
        return None

    if resp.status_code >= 400:
        logger.warning(
            "upload_page_screenshot: upload %s: %s",
            resp.status_code, resp.text[:200],
        )
        return None

    try:
        state.screenshot_name = name
    except Exception:
        pass
    return name


async def capture_debug_artifact(
    page, state, *, reason: str,
) -> dict:
    """Screenshot + DOM snapshot for post-mortem. Best-effort.

    Returns a small dict describing what we captured so the terminal node
    can include it in its `graph_payload` for `dump_graph_traces` training
    data. All keys are always present with bool / str values.
    """
    result = {
        "screenshot_uploaded": False,
        "dom_saved": False,
        "reason": reason,
    }

    if page is None or not getattr(state, "scrape_id", None):
        return result

    uploaded = await upload_page_screenshot(page, state, reason=reason)
    result["screenshot_uploaded"] = uploaded is not None

    # ── DOM snapshot (first _MAX_DOM_BYTES bytes) ─────────────────────────
    # Only write to scrape.html if it's empty — we don't want to clobber a
    # legit captured DOM. Write-to-empty is the whole point: Fail paths
    # that short-circuit Capture leave scrape.html null, which makes the
    # admin UI's "view raw html" link useless.
    try:
        dom = await page.content()
    except Exception:
        dom = None

    if dom:
        dom = truncate_dom(dom, scrape_id=state.scrape_id)
        # Use the api to check if scrape.html is already set. Read-first-
        # then-maybe-write costs a round trip but keeps us from silently
        # overwriting a successful captured DOM. On any read/write
        # exception, skip — this is post-mortem data, not critical.
        try:
            get_resp = httpx.get(
                f"{_api_base()}/api/v1/scrapes/{state.scrape_id}/",
                headers=_api_headers(),
                timeout=10.0,
            )
            attrs = (
                (get_resp.json() or {}).get("data", {}).get("attributes", {})
                if get_resp.status_code == 200
                else {}
            )
            existing_html = attrs.get("html") or ""
        except Exception:
            existing_html = ""

        if not existing_html:
            try:
                patch_resp = httpx.patch(
                    f"{_api_base()}/api/v1/scrapes/{state.scrape_id}/",
                    json={
                        "data": {
                            "type": "scrape",
                            "id": str(state.scrape_id),
                            "attributes": {"html": dom},
                        }
                    },
                    headers={**_api_headers(), "Content-Type": "application/vnd.api+json"},
                    timeout=30.0,
                )
                if patch_resp.status_code < 400:
                    result["dom_saved"] = True
                else:
                    logger.warning(
                        "capture_debug_artifact: DOM patch %s: %s",
                        patch_resp.status_code, patch_resp.text[:200],
                    )
            except Exception:
                logger.warning(
                    "capture_debug_artifact: DOM patch exception scrape_id=%s",
                    state.scrape_id, exc_info=True,
                )

    return result
