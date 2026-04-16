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

from lib.api_tools import ApiClient, get_scrapes, update_scrape, upload_screenshot
from lib.browser.engine import configure as configure_engine
from mcp_servers.browser_server import scrape_page

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("hold_poller")

POLL_INTERVAL = int(os.environ.get("HOLD_POLL_INTERVAL", "30"))


async def process_scrape(api: ApiClient, scrape: dict) -> bool:
    """Process a single hold scrape. Returns True on success."""
    scrape_id = int(scrape["id"])
    attrs = scrape.get("attributes", {})
    url = attrs.get("url")

    if not url:
        logger.warning("Scrape %s has no URL, skipping", scrape_id)
        await update_scrape(api, scrape_id, status="failed", note="No URL provided")
        return False

    logger.info("Processing scrape %s: %s", scrape_id, url)

    # Mark as running
    await update_scrape(api, scrape_id, status="running", note="Poller picked up")

    try:
        # Direct scrape — no LLM, just browser
        result_json = await scrape_page(url)
        result = json.loads(result_json)

        if result.get("error") == "login_wall_detected":
            msg = result.get("message", "Login wall detected")
            logger.warning("Scrape %s: %s", scrape_id, msg)
            await update_scrape(api, scrape_id, status="failed", note=msg)
            return False

        if result.get("error"):
            logger.error("Scrape %s error: %s", scrape_id, result["error"])
            await update_scrape(api, scrape_id, status="failed", note=result["error"])
            return False

        content = result.get("content", "")
        if not content.strip():
            logger.warning("Scrape %s: empty content", scrape_id)
            await update_scrape(api, scrape_id, status="failed", note="Empty content returned")
            return False

        # Upload screenshot to API
        screenshot_name = result.get("screenshot")
        if screenshot_name:
            from mcp_servers.browser_server import SCREENSHOT_DIR
            screenshot_path = SCREENSHOT_DIR / screenshot_name
            if screenshot_path.exists():
                await upload_screenshot(api, scrape_id, screenshot_path)
                screenshot_path.unlink()
                logger.info("Screenshot uploaded for scrape %s", scrape_id)

        # Push content back — the API handles extraction + profile update
        await update_scrape(api, scrape_id, status="completed", job_content=content,
                            note=f"Content delivered ({len(content)} chars)")
        logger.info("Scrape %s: content delivered (%d chars), API will extract", scrape_id, len(content))

        return True

    except Exception as exc:
        logger.exception("Scrape %s failed", scrape_id)
        await update_scrape(api, scrape_id, status="failed", note=str(exc)[:200])
        return False


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
    return parser.parse_args()


async def main():
    args = _parse_args()
    configure_engine(engine=args.engine, headless=args.headless)

    base_url = os.environ.get("CC_API_BASE_URL")
    token = os.environ.get("CC_API_TOKEN")

    if not base_url or not token:
        logger.error("CC_API_BASE_URL and CC_API_TOKEN are required")
        sys.exit(1)

    api = ApiClient(base_url=base_url, token=token)

    running = True

    def stop(*_):
        nonlocal running
        running = False
        logger.info("Shutting down...")

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    from lib.browser.engine import get_engine, get_headless
    logger.info(
        "Hold poller started (interval=%ds, api=%s, engine=%s, headless=%s)",
        POLL_INTERVAL,
        base_url,
        get_engine(),
        get_headless(),
    )

    while running:
        count = await poll_once(api)
        if count:
            logger.info("Processed %d scrape(s)", count)
        await asyncio.sleep(POLL_INTERVAL)


def main_sync():
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
