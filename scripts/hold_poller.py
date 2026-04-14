#!/usr/bin/env python3
"""Poll the Career Caddy API for hold scrapes, scrape locally, push results back.

Usage:
    BROWSER_HEADLESS=false \
    CC_API_BASE_URL=https://careercaddy.online \
    CC_API_TOKEN=<token> \
    JOB_EXTRACTOR_MODEL=anthropic:claude-haiku-4-5-20251001 \
    uv run caddy-poller
"""

import asyncio
import json
import logging
import os
import signal
import sys
from pathlib import Path

# Ensure the ai/ root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.api_tools import ApiClient, get_scrapes, update_scrape
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
        await update_scrape(api, scrape_id, status="failed")
        return False

    logger.info("Processing scrape %s: %s", scrape_id, url)

    # Mark as running
    await update_scrape(api, scrape_id, status="running")

    try:
        # Direct scrape — no LLM, just Camoufox
        result_json = await scrape_page(url)
        result = json.loads(result_json)

        if result.get("error") == "login_wall_detected":
            msg = result.get("message", "Login wall detected")
            logger.warning("Scrape %s: %s", scrape_id, msg)
            await update_scrape(api, scrape_id, status="failed")
            return False

        if result.get("error"):
            logger.error("Scrape %s error: %s", scrape_id, result["error"])
            await update_scrape(api, scrape_id, status="failed")
            return False

        content = result.get("content", "")
        if not content.strip():
            logger.warning("Scrape %s: empty content", scrape_id)
            await update_scrape(api, scrape_id, status="failed")
            return False

        # Rename screenshot to include scrape_id
        screenshot_name = result.get("screenshot")
        if screenshot_name:
            from mcp_servers.browser_server import SCREENSHOT_DIR
            old_path = SCREENSHOT_DIR / screenshot_name
            new_name = f"scrape_{scrape_id}_{screenshot_name}"
            new_path = SCREENSHOT_DIR / new_name
            if old_path.exists():
                old_path.rename(new_path)
                logger.info("Screenshot: %s", new_path)

        # Update scrape with content
        await update_scrape(api, scrape_id, status="completed", job_content=content)
        logger.info("Scrape %s: content saved (%d chars)", scrape_id, len(content))

        # Extract job data with LLM (Haiku) and create job post (gpt-4o-mini)
        try:
            from agents.agent_factory import register_defaults
            from agents.job_extractor_agent import extract_job_from_content
            from agents.career_caddy_agent import add_job_post

            register_defaults()
            api_token = os.environ.get("CC_API_TOKEN", "")

            job_data = await extract_job_from_content(
                content, url=url, api_token=api_token
            )
            logger.info("Extracted: %s at %s", job_data.title, job_data.company_name)

            caddy_result = await add_job_post(job_data, api_token=api_token)
            logger.info("Job post created: %s", json.dumps(caddy_result, indent=2))
        except Exception:
            logger.exception("Scrape %s: extraction/creation failed (content was saved)", scrape_id)

        return True

    except Exception:
        logger.exception("Scrape %s failed", scrape_id)
        await update_scrape(api, scrape_id, status="failed")
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


async def main():
    base_url = os.environ.get("CC_API_BASE_URL")
    token = os.environ.get("CC_API_TOKEN")

    if not base_url or not token:
        logger.error("CC_API_BASE_URL and CC_API_TOKEN are required")
        sys.exit(1)

    # Use Api-Key auth for the poller (not JWT Bearer)
    api = ApiClient(base_url=base_url, token=token)
    api._headers["Authorization"] = f"Api-Key {token}"

    running = True

    def stop(*_):
        nonlocal running
        running = False
        logger.info("Shutting down...")

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    logger.info(
        "Hold poller started (interval=%ds, api=%s, headless=%s)",
        POLL_INTERVAL,
        base_url,
        os.environ.get("BROWSER_HEADLESS", "true"),
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
