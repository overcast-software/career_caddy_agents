#!/usr/bin/env python3
"""Career Caddy scrape runner — claims hold scrapes via the api, runs the
browser locally, pushes results back.

Runs on any host with outbound HTTPS to the Career Caddy api + a long-lived
API key + a Camoufox/Playwright install. N runners can coexist safely against
the same api — POST /api/v1/scrapes/claim-next/ uses SELECT FOR UPDATE
SKIP LOCKED to hand each scrape to exactly one runner.

The runner only runs the browser — extraction, job post creation, and scrape
profile updates are handled by the api when it receives the scraped content.

Naming: was ``pollers/hold_poller.py`` (and the ``caddy-poller`` console-script
entry) up through 2026-05-30. The name was always a misnomer — it's a
runner, not a poller-of-hold-status alone. Renamed as Phase 1 of
``notes.org/Plans/Scrape runner — harden hold-poller``. The ``caddy-poller``
entry stays as an alias for one release.

Usage:
    CC_API_BASE_URL=https://careercaddy.online \
    CC_API_TOKEN=<token> \
    CC_RUNNER_NAME=omarchy \
    uv run caddy-runner
"""

import argparse
import asyncio
import logging

import yaml
import os
import signal
import socket
import sys
from pathlib import Path

# Ensure the ai/ root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.api_tools import (
    ApiClient,
    claim_next_scrape,
    get_scrape_profile,
    get_scrapes,
    update_scrape,
)
from browser.engine import (
    configure as configure_engine,
    get_engine,
    launch_browser,
)
from browser.resident import ResidentBrowser, ResidentDriverDead, is_driver_closed
from lib.url_unwrap import unwrap_url

# Module-level resident browser; set by the headed main() before the poll loop.
_RESIDENT: ResidentBrowser | None = None

# How many consecutive driver-death events (across claims) the runner
# tolerates before it stops claiming and backs off hard. A dead resident
# whose relaunch keeps failing would otherwise poll claim-next forever and
# drain the hold queue into `failed` rows (CC-141). Small bound, loud error.
_DRIVER_DEATH_BACKOFF_THRESHOLD = 3
# Seconds to sleep after the threshold trips, on top of POLL_INTERVAL, to
# give a supervisor (systemd Restart / operator) time to respawn the browser.
_DRIVER_DEATH_BACKOFF_SECONDS = 120


class DriverDeath(Exception):
    """Signal raised up through the runner when a claimed scrape failed only
    because the resident browser driver was dead (infra-death), NOT because
    of page/content. The scrape is left re-queued as ``hold`` (never
    ``failed``); the poll loop counts these and backs off rather than
    draining the queue.
    """


def format_scrape_label(entry_id, final_id) -> str:
    """Format a scrape id for terminal log lines.

    ResolveFinalUrl can fork a child scrape and swap ``state.scrape_id``;
    downstream nodes (and the resulting JobPost) belong to the child, not
    the entry row. Logging the entry id alone misattributes the child's
    outcome to the parent — a logfire search for "Scrape 302 graph done"
    lights up even when row 302 itself produced no JobPost.

    Returns ``"entry→final"`` whenever the swap happened, ``"entry"``
    otherwise. ``None``/zero final_id falls back to entry.
    """
    if final_id and final_id != entry_id:
        return f"{entry_id}→{final_id}"
    return str(entry_id)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from lib.logfire_setup import setup_logfire  # noqa: E402

setup_logfire("scrape_runner")

logger = logging.getLogger("scrape_runner")

POLL_INTERVAL = int(os.environ.get("HOLD_POLL_INTERVAL", "30"))


def _parse_hostname(url: str) -> str:
    """Extract and normalize hostname from a URL."""
    from urllib.parse import urlparse
    from browser.credentials import Credentials
    raw = urlparse(url).hostname or ""
    return Credentials.normalize_domain(raw) if raw else ""


async def _fetch_profile(api: ApiClient, hostname: str) -> dict | None:
    """Fetch the scrape profile for a hostname. Returns parsed profile dict or None."""
    if not hostname:
        return None
    raw = await get_scrape_profile(api, hostname)
    body = yaml.safe_load(raw)
    if not isinstance(body, dict) or body.get("error"):
        return None
    profiles = body.get("data", [])
    if not profiles:
        return None
    p = profiles[0] if isinstance(profiles, list) else profiles
    return {
        "id": p["id"],
        "css_selectors": (p.get("attributes") or {}).get("css-selectors") or {},
    }


async def process_scrape(api: ApiClient, scrape: dict) -> bool:
    """Process a single hold scrape through the pydantic-graph pipeline.

    The poller is a thin dispatcher: it opens a Playwright page, builds
    a ScrapeGraphState, and hands control to run_scrape_graph. Each
    graph node writes its own trace + owns its side effects (PATCH
    scrape, upload screenshot, push profile selectors, create JobPost).
    """
    scrape_id = scrape["id"]
    attrs = scrape.get("attributes", {})
    url = attrs.get("url")
    skip_extract = bool(attrs.get("skip_extract"))
    # Phase B — Extension direct-POST plan. The api's claim-next response
    # serializes both source_mode (CharField, default 'browser') and
    # captured_payload (JSONField, nullable). When source_mode is
    # 'extension-direct' and the payload is present, the scrape-graph's
    # StartScrape node branches to SkipBrowserTier so no browser-tier
    # node ever runs. The Phase-A api-side validator already enforced
    # title/company/description presence at write time; the graph
    # double-checks defensively before branching.
    source_mode = attrs.get("source_mode") or "browser"
    captured_payload = attrs.get("captured_payload")

    if not url:
        logger.warning("Scrape %s has no URL, skipping", scrape_id)
        await update_scrape(api, scrape_id, status="failed", note="No URL provided")
        return False

    unwrapped = unwrap_url(url)
    if unwrapped != url:
        logger.info("Scrape %s: unwrapped tracker URL\n  from: %s\n  to:   %s", scrape_id, url, unwrapped)
        url = unwrapped

    logger.info(
        "Processing scrape %s: %s (source_mode=%s)",
        scrape_id, url, source_mode,
    )

    hostname = _parse_hostname(url)
    profile = await _fetch_profile(api, hostname)
    if profile:
        logger.info("Scrape %s: loaded profile id=%s for %s", scrape_id, profile["id"], hostname)
    else:
        logger.info("Scrape %s: no profile for %s", scrape_id, hostname)

    await update_scrape(api, scrape_id, status="running", note="Poller picked up")
    return await _run_graph(
        api, scrape_id, url, hostname, profile, skip_extract,
        source_mode=source_mode, captured_payload=captured_payload,
    )


async def _run_graph(
    api: ApiClient,
    scrape_id: str,
    url: str,
    hostname: str,
    profile: dict | None,
    skip_extract: bool = False,
    *,
    source_mode: str = "browser",
    captured_payload: dict | None = None,
) -> bool:
    """Run the pydantic-graph against a live Playwright page."""
    from scrape_graph import ScrapeGraphState
    from scrape_graph.runner import run_scrape_graph

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
        skip_extract=skip_extract,
        source_mode=source_mode,
        captured_payload=captured_payload,
    )

    # Hard cap on a single graph run. Without this, a node that hangs
    # (e.g. an auth interstitial that doesn't fire `load`, an httpx
    # call that ignores its own timeout) wedges the poller on one
    # scrape forever — the row stays `running` and the queue stops
    # advancing. 4 minutes is well above the worst observed happy-path
    # run; anything beyond that is a stuck node, not slow work.
    GRAPH_RUN_TIMEOUT_S = 240.0

    async def _drive() -> None:
        # Phase B fast-path: extension-direct scrapes never need a
        # browser page — StartScrape branches to SkipBrowserTier and the
        # graph runs persistence only. Skip the entire browser launch
        # so a Camoufox/Chromium process is never spawned for these
        # scrapes. has_browser=True keeps the entry at StartScrape (the
        # extract-only graph would skip past StartScrape entirely).
        if source_mode == "extension-direct":
            await run_scrape_graph(state, browser_page=None, has_browser=True)
            return
        if _RESIDENT is not None:
            page = await _RESIDENT.open_tab(domain=hostname, seed_cookies=cookies)
            try:
                await run_scrape_graph(state, browser_page=page, has_browser=True)
            finally:
                await _RESIDENT.close_tab(page)
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

    try:
        await asyncio.wait_for(_drive(), timeout=GRAPH_RUN_TIMEOUT_S)
    except asyncio.TimeoutError:
        logger.warning(
            "Scrape %s exceeded %.0fs graph cap",
            format_scrape_label(scrape_id, state.scrape_id),
            GRAPH_RUN_TIMEOUT_S,
        )
        await update_scrape(
            api, scrape_id, status="failed",
            note=f"graph run exceeded {int(GRAPH_RUN_TIMEOUT_S)}s cap",
        )
        return False
    except (ResidentDriverDead, Exception) as exc:
        # Infra-death guard (CC-141). A dead browser driver surfaces as
        # ResidentDriverDead (resident path, already relaunched-and-failed)
        # or a raw Playwright "Connection closed while reading from the
        # driver" error anywhere in the graph run. Either way this is NOT a
        # content failure: the page never got a fair scrape. Marking it
        # `failed` here is what silently drains the hold queue into failed
        # rows while the browser is dead. Instead re-queue it as `hold`
        # (distinctly tagged) and raise DriverDeath so the poll loop backs
        # off instead of claim-looping on a dead browser.
        if isinstance(exc, ResidentDriverDead) or is_driver_closed(exc):
            label = format_scrape_label(scrape_id, state.scrape_id)
            logger.error(
                "Scrape %s hit dead browser driver — re-queueing as hold, "
                "NOT failing (CC-141 infra-death)",
                label,
            )
            try:
                await update_scrape(
                    api, scrape_id, status="hold",
                    note="[driver-death] browser driver connection closed; re-queued",
                )
            except Exception:
                logger.warning(
                    "Scrape %s: re-queue PATCH failed after driver death", label,
                    exc_info=True,
                )
            raise DriverDeath(str(exc)[:200]) from exc
        logger.exception(
            "Scrape %s failed inside graph run",
            format_scrape_label(scrape_id, state.scrape_id),
        )
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
        format_scrape_label(scrape_id, state.scrape_id),
        outcome,
        state.job_post_id,
        len(state.tier_attempts),
        len(state.node_trace),
    )
    return outcome in ("success", "duplicate")


def _is_headless() -> bool:
    """Respect the engine's configured headless flag."""
    from browser.engine import get_headless
    return bool(get_headless())


async def poll_once(api: ApiClient) -> int:
    """Claim and process hold scrapes one at a time via the runner-safe
    POST /api/v1/scrapes/claim-next/ endpoint (Plans/Scrape runner
    Phase 1). Each call atomically picks the oldest hold row, flips it
    to status='running', and returns it; SELECT FOR UPDATE SKIP LOCKED
    means N concurrent runners on different hosts (omarchy + pibu +
    future) split the queue without racing.

    The hold queue is a single FIFO — a scrape is a scrape. Headed vs
    headless is a property of the runner (``--headed``), not a flag
    stamped on each scrape, so there is no per-scrape claim partition.

    Loops until the api returns 204 (queue empty). Returns count
    processed.
    """
    runner_name = os.environ.get("CC_RUNNER_NAME") or socket.gethostname()
    processed = 0
    while True:
        raw = await claim_next_scrape(api, runner_name=runner_name)
        data = yaml.safe_load(raw)

        if not isinstance(data, dict) or data.get("error"):
            logger.error(
                "API error: %s",
                (data or {}).get("error") if isinstance(data, dict) else raw[:200],
            )
            return processed

        scrape = data.get("data")
        if scrape is None:
            # 204 — no hold scrapes available. Caller's sleep loop
            # kicks in for the next interval.
            return processed

        logger.info("Claimed scrape %s (runner=%s)", scrape.get("id"), runner_name)
        # DriverDeath propagates: the browser is dead, so stop claiming this
        # cycle rather than pulling the next hold row and failing it too. The
        # scrape we just claimed was re-queued as `hold` by _run_graph, so it
        # stays available for a healthy runner / the next cycle. The poll loop
        # counts these and backs off.
        if await process_scrape(api, scrape):
            processed += 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Poll for hold scrapes and process them")
    parser.add_argument(
        "--engine", choices=["camoufox", "chrome"], default=None,
        help="Browser engine (default: BROWSER_ENGINE env or 'camoufox')",
    )
    parser.add_argument("--headless", action="store_true", default=None, help="Run headless (default)")
    parser.add_argument(
        "--headed", dest="headed", action="store_true",
        help="Launch a single resident headed browser; spawn an ephemeral tab "
             "per scrape in the same window and close it on completion. Cookies "
             "persist on the shared context across scrapes, so a login/captcha "
             "you solve once stays warm. Use it to watch the runner work and "
             "screenshot to verify. Forces headless off.",
    )
    # Deprecated alias of --headed, retained so the live pibu systemd unit and
    # tmuxinator invocations keep working through the rollout. It launches the
    # same resident headed browser; it no longer partitions the scrape queue
    # (a scrape is a scrape). main() logs a one-line deprecation warning.
    parser.add_argument(
        "--attended", dest="headed", action="store_true", help=argparse.SUPPRESS,
    )
    return parser.parse_args()


async def _preflight_auth(api: ApiClient) -> bool:
    """Hit an auth-required endpoint once to confirm the token works.

    Uses GET /api/v1/scrapes/ (read-only) — a 200 means auth passed.
    We don't use the unauthenticated /healthcheck/ because the whole
    point is to verify the credential, not just reachability. And we
    deliberately don't use POST /scrapes/claim-next/ here because that
    endpoint has side effects (claiming a scrape) — preflight must be
    idempotent.
    """
    try:
        raw = await get_scrapes(api, status="hold", sort="-scraped_at", per_page=1)
    except Exception as exc:
        logger.error("Pre-flight request raised: %s", exc)
        return False
    try:
        body = yaml.safe_load(raw)
    except Exception:
        logger.error("Pre-flight returned unparseable response: %s", raw[:200])
        return False
    if isinstance(body, dict) and body.get("error"):
        logger.error("Pre-flight failed: %s", body.get("error"))
        return False
    return True


async def _run_poll_loop(api: ApiClient, running_flag):
    consecutive_driver_deaths = 0
    while running_flag():
        backoff = 0
        try:
            count = await poll_once(api)
            if count:
                logger.info("Processed %d scrape(s)", count)
            consecutive_driver_deaths = 0
        except DriverDeath as exc:
            # The resident browser is dead and could not be rebuilt. The
            # claimed scrape was already re-queued as `hold` (not failed).
            # Stop hammering claim-next: count the death and, once we cross
            # the threshold, back off hard with a loud ERROR so the outage is
            # visible in logfire instead of silently draining the queue.
            consecutive_driver_deaths += 1
            if consecutive_driver_deaths >= _DRIVER_DEATH_BACKOFF_THRESHOLD:
                backoff = _DRIVER_DEATH_BACKOFF_SECONDS
                logger.error(
                    "Browser driver dead %d cycles running (%s) — backing off "
                    "%ds and NOT claiming. Holds are left claimable; restart "
                    "the runner/browser to recover (CC-141).",
                    consecutive_driver_deaths, exc, backoff,
                )
            else:
                logger.warning(
                    "Browser driver death %d/%d — pausing this cycle (%s)",
                    consecutive_driver_deaths,
                    _DRIVER_DEATH_BACKOFF_THRESHOLD, exc,
                )
        except Exception:
            logger.warning("Poll cycle failed, will retry", exc_info=True)
            consecutive_driver_deaths = 0
        await asyncio.sleep(POLL_INTERVAL + backoff)


async def main():
    global _RESIDENT
    args = _parse_args()
    if "--attended" in sys.argv:
        logger.warning(
            "--attended is deprecated; it is now just an alias for --headed "
            "(launch the resident headed browser). It no longer partitions the "
            "scrape queue — a scrape is a scrape. Switch the invocation to --headed."
        )
    # Headed mode forces headless off.
    headless = False if args.headed else args.headless
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
        "poller boot: base_url=%s engine=%s headless=%s headed=%s poll_interval=%ds",
        base_url, args.engine, headless, bool(args.headed), POLL_INTERVAL,
    )

    api = ApiClient(base_url=base_url, token=token)

    # Pre-flight: verify the token is accepted before we spin up a browser.
    # A bad token means no scrape will ever succeed; don't burn a browser
    # launch (and on headed mode, a real user's workflow) to find out.
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

    from browser.engine import get_headless
    mode = "headed" if args.headed else "ephemeral"
    logger.info(
        "Hold poller started (mode=%s, interval=%ds, api=%s, engine=%s, headless=%s)",
        mode, POLL_INTERVAL, base_url, get_engine(), get_headless(),
    )

    if args.headed:
        # Wrap the whole headed block so SIGINT reaching Camoufox's
        # subprocess (which breaks the Playwright pipe) doesn't produce
        # a crash on shutdown. We've already persisted sessions after
        # each scrape, so a broken close is cosmetic.
        try:
            async with launch_browser(get_engine(), headless=False) as browser:
                _RESIDENT = ResidentBrowser(browser)
                logger.info(
                    "Headed: ready. Each scrape opens a tab in the resident "
                    "window and closes it on completion. Solve captchas in the "
                    "live tab as scrapes arrive."
                )
                try:
                    await _run_poll_loop(api, lambda: running)
                finally:
                    try:
                        saved = await _RESIDENT.save_sessions()
                        logger.info("Headed: saved sessions for %d domain(s)", saved)
                    except Exception:
                        logger.warning("Headed: save_sessions failed", exc_info=True)
                    try:
                        await _RESIDENT.close()
                    except Exception:
                        logger.debug("Headed: resident close raised", exc_info=True)
                    _RESIDENT = None
        except Exception as exc:
            # Camoufox / Playwright can raise here if the subprocess
            # received SIGINT ahead of us — the pipe is already closed
            # so browser.close() has nothing to reply. Session state is
            # already on disk; no user-visible loss.
            logger.info("Headed: browser shutdown finished with: %s", exc)
    else:
        await _run_poll_loop(api, lambda: running)


def main_sync():
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
