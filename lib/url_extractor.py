"""Canonicalize inbound URLs: unwrap tracker redirects, strip tracking params,
drop dead-listing pages.

Pure-Python (httpx + stdlib). No LLM dependency. Safe to import from
anywhere — API viewset, MCP tool, daemon, test.

Shape:
    canonical = await canonicalize_url(url)
    # str — canonical URL (tracking params stripped, trackers unwrapped)
    # None — tracker was dead or the listing page advertises "expired"

Ported from career_caddy_automation/src/agents/url_extractor.py. The
automation repo keeps its own copy since the email pipeline there calls it
directly; this file is the core copy, consumed by anything in-repo that
accepts URLs from users or external ingest.
"""
from __future__ import annotations

import logging
import re
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx

logger = logging.getLogger(__name__)


# Hosts that always redirect to the real job URL. Unique per recipient, so
# two users getting the same role see different tracker URLs — server-side
# dedup on `link` can't see they match without unwrapping.
_TRACKER_HOST_RE = re.compile(
    r"(?i)^("
    r"url\d*\.alerts\.jobot\.com"
    r"|click\.ziprecruiter\.com"
    r"|email\.mg\d*\.ziprecruiter\.com"
    r"|email\.mg\.ziprecruiter\.com"
    r"|url\d*\.mailmunch\.co"
    r"|email\.[a-z0-9-]+\.mailgun\.org"
    r"|links?\.[a-z0-9.-]+\.sendgrid\.net"
    r"|trk\.[a-z0-9.-]+"
    r"|click\.[a-z0-9.-]+"
    r"|t\.[a-z0-9.-]+"
    r")$"
)

# Query params to strip from canonical URLs. Safe to drop — none affect which
# job listing the URL points at.
_TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_content",
    "utm_term",
    "gclid",
    "fbclid",
    "mc_cid",
    "mc_eid",
    "trackingId",
    "refId",
    "lipi",
    "eid",
    "midToken",
    "midSig",
    "otpToken",
    "trk",
    "trkEmail",
    "tsid",
    "ssid",
    "fmid",
    "email_source",
    "email_token",
}

_DEAD_LINK_MARKERS = re.compile(
    r"(?i)"
    r"wrong link"
    r"|invalid link"
    r"|you have clicked on an invalid"
    r"|this (job|position|posting) (is )?(no longer|has been) (available|removed|filled)"
    r"|job (not found|expired|has been removed)"
    r"|expired job"
    r"|posting (no longer|has been) (available|active)"
    r"|page not found"
    r"|job you.re looking for"
    r"|we.re sorry,? but this job has expired"
)

# Hosts that serve "listing expired" pages at HTTP 200 rather than 404 — a
# bare status check misses them. For these we do a bounded GET and scan the
# body for _DEAD_LINK_MARKERS.
_DEAD_CHECK_HOST_RE = re.compile(
    r"(?i)^("
    r"(www\.)?hiring\.cafe"
    r"|(www\.)?hiringcafe\.com"
    r")$"
)


def strip_tracking_params(url: str) -> str:
    """Drop known tracking query params; preserve everything else."""
    try:
        p = urlparse(url)
    except ValueError:
        return url
    if not p.query:
        return url
    kept = [
        (k, v)
        for k, v in parse_qsl(p.query, keep_blank_values=True)
        if k not in _TRACKING_PARAMS
    ]
    return urlunparse(p._replace(query=urlencode(kept)))


async def _peek_for_dead_marker(client: httpx.AsyncClient, url: str) -> bool:
    try:
        async with client.stream(
            "GET",
            url,
            follow_redirects=True,
            timeout=3.0,
            headers={"User-Agent": "Mozilla/5.0 (CareerCaddyResolver)"},
        ) as r:
            if r.status_code >= 400:
                return False
            body = b""
            async for chunk in r.aiter_bytes():
                body += chunk
                if len(body) >= 16_384:
                    break
    except (TimeoutError, httpx.HTTPError) as exc:
        logger.debug("dead-marker peek failed for %s: %s", url, exc)
        return False
    return bool(_DEAD_LINK_MARKERS.search(body.decode("utf-8", errors="ignore")))


async def canonicalize_url(url: str, client: httpx.AsyncClient | None = None) -> str | None:
    """Return a canonical URL or None if the URL is a dead listing.

    - Strips known tracking params.
    - If host is a tracker (SendGrid/Jobot/etc), follows redirects and
      returns the resolved URL; drops if the tracker returns 4xx or the
      body matches a dead-listing marker.
    - If host is on the "dead-check" list (pages that 200 with 'expired'
      text), peeks at the body and drops if a marker matches.

    Network errors on trackers return the stripped-params version rather
    than dropping — transient failures shouldn't lose a real posting.
    """
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient()

    try:
        try:
            host = urlparse(url).netloc
        except ValueError:
            return url

        if not _TRACKER_HOST_RE.match(host):
            canonical = strip_tracking_params(url)
            if _DEAD_CHECK_HOST_RE.match(host):
                if await _peek_for_dead_marker(client, canonical):
                    logger.info("%s serves an expired-listing page — dropping", canonical)
                    return None
            return canonical

        try:
            async with client.stream(
                "GET",
                url,
                follow_redirects=True,
                timeout=3.0,
                headers={"User-Agent": "Mozilla/5.0 (CareerCaddyResolver)"},
            ) as r:
                if r.status_code >= 400:
                    logger.info("tracker %s returned %d — dropping", url, r.status_code)
                    return None
                final_url = str(r.url)
                body = b""
                async for chunk in r.aiter_bytes():
                    body += chunk
                    if len(body) >= 16_384:
                        break
        except (TimeoutError, httpx.HTTPError) as exc:
            logger.debug("tracker resolve failed, keeping raw URL %s: %s", url, exc)
            return strip_tracking_params(url)

        text = body.decode("utf-8", errors="ignore")
        if _DEAD_LINK_MARKERS.search(text):
            logger.info("tracker %s resolved to error page — dropping", url)
            return None

        if _TRACKER_HOST_RE.match(urlparse(final_url).netloc):
            logger.info("tracker %s never redirected off tracker domain — dropping", url)
            return None

        return strip_tracking_params(final_url)
    finally:
        if owns_client:
            await client.aclose()
