"""Shared debug-artifact helper: snapshot the page on any terminal failure.

Capture's happy path (nodes_scrape.py) already takes a screenshot, uploads
it, and PATCHes the scrape's html + job_content. But failure paths short-
circuit before Capture — `ObstacleFail` fires when we can't clear a login
wall, `ExtractFail` fires when all tiers produce junk. At those moments
we have nothing to look at post-mortem except a cryptic failure_reason.

`capture_debug_artifact` is the invariant: every Fail terminal calls it
before `_patch_scrape_status`. Takes a viewport screenshot (best-effort),
uploads it to the scrape's screenshots endpoint, and snapshots the HTML
into scrape.html if that column is still empty. Tolerant of detached
pages, closed browsers, upload errors — never raises, never blocks the
Fail terminal from finalizing the scrape.

Why viewport over full-page: full-page can be multi-MB on LinkedIn and
sometimes takes >5s to render. On a failure path we're already in a
degraded state; we want a quick visual, not a perfect one.
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

    # ── Screenshot ────────────────────────────────────────────────────────
    host = (
        urlparse(state.canonical_url or state.submitted_url or "").hostname
        or "unknown"
    ).lower()
    if host.startswith("www."):
        host = host[4:]
    name = f"{host}_{reason}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

    # Capture's happy path writes the screenshot to SCREENSHOT_DIR on disk
    # first. Here we keep it purely in-memory — the helper fires on failure
    # paths where disk cleanup adds risk, and the upload accepts bytes
    # directly. The extra disk round-trip Capture does is cosmetic, not
    # required.

    png_bytes: bytes | None = None
    try:
        png_bytes = await page.screenshot(full_page=False)
    except Exception:
        logger.warning(
            "capture_debug_artifact: screenshot failed scrape_id=%s reason=%s",
            state.scrape_id, reason, exc_info=True,
        )

    if png_bytes:
        try:
            resp = httpx.post(
                f"{_api_base()}/api/v1/scrapes/{state.scrape_id}/screenshots/",
                files={"file": (name, png_bytes, "image/png")},
                headers=_api_headers(),
                timeout=30.0,
            )
            if resp.status_code < 400:
                result["screenshot_uploaded"] = True
                # Record on state so graph_payload can reference it.
                try:
                    state.screenshot_name = name
                except Exception:
                    pass
            else:
                logger.warning(
                    "capture_debug_artifact: upload %s: %s",
                    resp.status_code, resp.text[:200],
                )
        except Exception:
            logger.warning(
                "capture_debug_artifact: upload exception scrape_id=%s",
                state.scrape_id, exc_info=True,
            )

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
