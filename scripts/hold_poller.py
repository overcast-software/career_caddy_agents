#!/usr/bin/env python3
"""Poll the Career Caddy API for hold scrapes, scrape locally, push results back.

The worker only runs the browser — extraction, job post creation, and scrape
profile updates are handled by the API when it receives the scraped content.

Usage:
    CC_API_BASE_URL=https://careercaddy.online \
    CC_API_TOKEN=<token> \
    uv run caddy-poller
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from pathlib import Path

# Ensure the ai/ root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.api_tools import (
    ApiClient, get_scrapes, get_scrape_profile, update_scrape,
)
from lib.browser.engine import (
    configure as configure_engine,
    get_engine,
    launch_browser,
)
from lib.browser.resident import ResidentBrowser
from lib.url_unwrap import unwrap_url

# Module-level resident browser; set by the attended main() before the poll loop.
_RESIDENT: ResidentBrowser | None = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from lib.logfire_setup import setup_logfire  # noqa: E402

setup_logfire("hold_poller")

logger = logging.getLogger("hold_poller")

POLL_INTERVAL = int(os.environ.get("HOLD_POLL_INTERVAL", "30"))


def _parse_hostname(url: str) -> str:
    """Extract and normalize hostname from a URL."""
    from urllib.parse import urlparse
    from lib.browser.credentials import Credentials
    raw = urlparse(url).hostname or ""
    return Credentials.normalize_domain(raw) if raw else ""


async def _fetch_profile(api: ApiClient, hostname: str) -> dict | None:
    """Fetch the scrape profile for a hostname. Returns parsed profile dict or None."""
    if not hostname:
        return None
    raw = await get_scrape_profile(api, hostname)
    wrapper = json.loads(raw)
    body = wrapper.get("data") or {}
    profiles = body.get("data", [])
    if not profiles:
        return None
    p = profiles[0] if isinstance(profiles, list) else profiles
    return {
        "id": int(p["id"]),
        "css_selectors": (p.get("attributes") or {}).get("css-selectors") or {},
    }


async def process_scrape(api: ApiClient, scrape: dict) -> bool:
    """Process a single hold scrape through the pydantic-graph pipeline.

    The poller is a thin dispatcher: it opens a Playwright page, builds
    a ScrapeGraphState, and hands control to run_scrape_graph. Each
    graph node writes its own trace + owns its side effects (PATCH
    scrape, upload screenshot, push profile selectors, create JobPost).
    """
    scrape_id = int(scrape["id"])
    attrs = scrape.get("attributes", {})
    url = attrs.get("url")

    if not url:
        logger.warning("Scrape %s has no URL, skipping", scrape_id)
        await update_scrape(api, scrape_id, status="failed", note="No URL provided")
        return False

    unwrapped = unwrap_url(url)
    if unwrapped != url:
        logger.info("Scrape %s: unwrapped tracker URL\n  from: %s\n  to:   %s", scrape_id, url, unwrapped)
        url = unwrapped

    logger.info("Processing scrape %s: %s", scrape_id, url)

    hostname = _parse_hostname(url)
    profile = await _fetch_profile(api, hostname)
    if profile:
        logger.info("Scrape %s: loaded profile id=%s for %s", scrape_id, profile["id"], hostname)
    else:
        logger.info("Scrape %s: no profile for %s", scrape_id, hostname)

    await update_scrape(api, scrape_id, status="running", note="Poller picked up")
    return await _run_graph(api, scrape_id, url, hostname, profile)


async def _run_graph(
    api: ApiClient, scrape_id: int, url: str, hostname: str, profile: dict | None,
) -> bool:
    """Run the pydantic-graph against a live Playwright page."""
    from lib.scrape_graph import ScrapeGraphState
    from lib.scrape_graph.runner import run_scrape_graph

    css_selectors = (profile or {}).get("css_selectors") or {}
    cookies = []
    try:
        from mcp_servers.browser_server import _resolve_scrape_inputs
        _, _, cookies, _ = _resolve_scrape_inputs(url, profile)
    except Exception:
        logger.debug("primary: cookie resolution failed", exc_info=True)

    state = ScrapeGraphState(
        scrape_id=scrape_id,
        submitted_url=url,
        original_scrape_id=scrape_id,
        profile=css_selectors,
        source="poller",
    )

    try:
        if _RESIDENT is not None:
            # Attended: reuse the domain's resident tab so manual
            # login state persists across scrapes.
            async with _RESIDENT.lock_for(hostname):
                page = await _RESIDENT.page_for(hostname, seed_cookies=cookies)
                await run_scrape_graph(state, browser_page=page, has_browser=True)
        else:
            async with launch_browser(get_engine(), _is_headless()) as browser:
                ctx = await browser.new_context()
                if cookies:
                    try:
                        await ctx.add_cookies(cookies)
                    except Exception:
                        logger.debug("primary: cookie seed failed", exc_info=True)
                page = await ctx.new_page()
                try:
                    await run_scrape_graph(state, browser_page=page, has_browser=True)
                finally:
                    try:
                        await ctx.close()
                    except Exception:
                        pass
    except Exception as exc:
        logger.exception("Scrape %s failed inside graph run", scrape_id)
        await update_scrape(api, scrape_id, status="failed", note=str(exc)[:200])
        return False

    # Persist cookies after each scrape so Ctrl-C doesn't lose manually-
    # warmed login state.
    if _RESIDENT is not None:
        try:
            await _RESIDENT.save_sessions()
        except Exception:
            logger.debug("save_sessions after scrape %s failed", scrape_id, exc_info=True)

    outcome = state.outcome or "failure"
    logger.info(
        "Scrape %s graph done: outcome=%s job_post_id=%s tiers=%d trace=%d",
        scrape_id, outcome, state.job_post_id,
        len(state.tier_attempts), len(state.node_trace),
    )
    return outcome in ("success", "duplicate")


def _is_headless() -> bool:
    """Respect the engine's configured headless flag."""
    from lib.browser.engine import get_headless
    return bool(get_headless())


async def poll_once(api: ApiClient) -> int:
    """Poll for hold scrapes and process them. Returns count processed."""
    raw = await get_scrapes(api, status="hold", sort="id")
    data = json.loads(raw)

    if not data.get("success"):
        logger.error("API error: %s", data.get("error"))
        return 0

    scrapes = data.get("data", {}).get("data", [])
    if not scrapes:
        return 0

    logger.info("Found %d hold scrape(s)", len(scrapes))

    processed = 0
    for scrape in scrapes:
        success = await process_scrape(api, scrape)
        if success:
            processed += 1

    return processed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Poll for hold scrapes and process them")
    parser.add_argument(
        "--engine", choices=["camoufox", "chrome"], default=None,
        help="Browser engine (default: BROWSER_ENGINE env or 'camoufox')",
    )
    parser.add_argument("--headless", action="store_true", default=None, help="Run headless")
    parser.add_argument("--headed", dest="headless", action="store_false", help="Run headed")
    parser.add_argument(
        "--attended", action="store_true",
        help="Launch a single headed browser with per-domain tabs. Solve captchas "
             "once in the open tabs; state persists across scrapes. Implies --headed.",
    )
    parser.add_argument(
        "--attended-delay", type=int, nargs="?", const=5, default=0, metavar="N",
        help="Seconds to wait after the browser launches before preseeding "
             "tabs (attended mode only). Gives you a chance to move the "
             "first Camoufox window to a dedicated workspace before the "
             "~10 preseed tabs spawn. Omit the flag for no delay; pass "
             "--attended-delay (no value) for 5 seconds; pass a number "
             "to override. No-op when --attended is off.",
    )
    return parser.parse_args()


def _attended_preseed_domains() -> list[str]:
    """Domains to pre-open tabs for — read from secrets.yml."""
    try:
        from lib.browser.credentials import Credentials
        creds = Credentials.load()
        return sorted(creds.domains.keys())
    except Exception as exc:
        logger.warning("Could not load secrets.yml for preseed: %s", exc)
        return []


async def _preflight_auth(api: ApiClient) -> bool:
    """Hit an auth-required endpoint once to confirm the token works.

    Uses GET /api/v1/scrapes/ — a 200 means auth passed. We don't use
    the unauthenticated /healthcheck/ because the whole point is to
    verify the credential, not just reachability.
    """
    try:
        raw = await get_scrapes(api, status="hold", sort="id")
    except Exception as exc:
        logger.error("Pre-flight request raised: %s", exc)
        return False
    try:
        body = json.loads(raw)
    except Exception:
        logger.error("Pre-flight returned non-JSON: %s", raw[:200])
        return False
    if body.get("success"):
        return True
    logger.error("Pre-flight failed: %s", body.get("error") or body)
    return False


async def _run_poll_loop(api: ApiClient, running_flag):
    while running_flag():
        try:
            count = await poll_once(api)
            if count:
                logger.info("Processed %d scrape(s)", count)
        except Exception:
            logger.warning("Poll cycle failed, will retry", exc_info=True)
        await asyncio.sleep(POLL_INTERVAL)


async def main():
    global _RESIDENT
    args = _parse_args()
    # Attended mode forces headed.
    headless = False if args.attended else args.headless
    configure_engine(engine=args.engine, headless=headless)

    base_url = os.environ.get("CC_API_BASE_URL")
    token = os.environ.get("CC_API_TOKEN")

    if not base_url or not token:
        logger.error("CC_API_BASE_URL and CC_API_TOKEN are required")
        sys.exit(1)

    # Loud startup banner — so a wrong base_url is visible before a
    # single scrape runs. Warning level so it shows under default
    # logging without LOG_LEVEL tweaks.
    logger.warning(
        "poller boot: base_url=%s engine=%s headless=%s attended=%s poll_interval=%ds",
        base_url, args.engine, headless, bool(args.attended), POLL_INTERVAL,
    )

    api = ApiClient(base_url=base_url, token=token)

    # Pre-flight: verify the token is accepted before we spin up a browser.
    # A bad token means no scrape will ever succeed; don't burn a browser
    # launch (and on attended mode, a real user's workflow) to find out.
    if not await _preflight_auth(api):
        logger.error(
            "Pre-flight auth check failed against %s — aborting before browser launch. "
            "Verify CC_API_TOKEN is a valid API key for this API (`jh_…` prefix) and "
            "that CC_API_BASE_URL points at the right host.",
            base_url,
        )
        sys.exit(2)

    running = True

    def stop(*_):
        nonlocal running
        running = False
        logger.info("Shutting down...")

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    from lib.browser.engine import get_headless
    mode = "attended" if args.attended else "ephemeral"
    logger.info(
        "Hold poller started (mode=%s, interval=%ds, api=%s, engine=%s, headless=%s)",
        mode, POLL_INTERVAL, base_url, get_engine(), get_headless(),
    )

    if args.attended:
        # Wrap the whole attended block so SIGINT reaching Camoufox's
        # subprocess (which breaks the Playwright pipe) doesn't produce
        # a crash on shutdown. We've already persisted sessions after
        # each scrape, so a broken close is cosmetic.
        try:
            async with launch_browser(get_engine(), headless=False) as browser:
                _RESIDENT = ResidentBrowser(browser)
                preseed = _attended_preseed_domains()
                # Give the user a chance to move the first Camoufox window
                # (or set up compositor rules, move monitors, whatever)
                # before the preseed burst spawns another ~10 windows. Opt-in
                # via --attended-delay N; silent when N<=0.
                if args.attended_delay and args.attended_delay > 0:
                    for remaining in range(args.attended_delay, 0, -1):
                        sys.stderr.write(
                            f"\rattended: preseed in {remaining}s "
                            f"({len(preseed)} tabs)... "
                        )
                        sys.stderr.flush()
                        await asyncio.sleep(1)
                    sys.stderr.write("\n")
                logger.info("Attended: preseeding tabs for %s", preseed)
                await _RESIDENT.preseed(preseed)
                logger.info(
                    "Attended: tabs open. Solve any captchas/logins in the browser; "
                    "scrapes will start on the next poll tick."
                )
                try:
                    await _run_poll_loop(api, lambda: running)
                finally:
                    try:
                        saved = await _RESIDENT.save_sessions()
                        logger.info("Attended: saved sessions for %d domain(s)", saved)
                    except Exception:
                        logger.warning("Attended: save_sessions failed", exc_info=True)
                    try:
                        await _RESIDENT.close()
                    except Exception:
                        logger.debug("Attended: resident close raised", exc_info=True)
                    _RESIDENT = None
        except Exception as exc:
            # Camoufox / Playwright can raise here if the subprocess
            # received SIGINT ahead of us — the pipe is already closed
            # so browser.close() has nothing to reply. Session state is
            # already on disk; no user-visible loss.
            logger.info("Attended: browser shutdown finished with: %s", exc)
    else:
        await _run_poll_loop(api, lambda: running)


def main_sync():
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
