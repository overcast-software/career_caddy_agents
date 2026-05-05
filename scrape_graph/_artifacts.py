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
        if len(dom) > _MAX_DOM_BYTES:
            dom = dom[:_MAX_DOM_BYTES] + "\n<!-- [truncated for debug artifact] -->"
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
