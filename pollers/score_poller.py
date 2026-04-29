#!/usr/bin/env python3
"""Auto-score daemon.

Polls Career Caddy for scrapes in ``status=completed`` whose linked JobPost
has no ``Score`` yet, then calls the scoring endpoint for each. Scoring is
an expensive LLM operation on the server; running this daemon is opt-in
per deployment (``CC_AUTOSCORE_ENABLED=1``) and opt-in per user
(``Profile.auto_score``).

Usage:
    CC_API_BASE_URL=http://localhost:8000 \\
    CC_API_TOKEN=<token> \\
    uv run caddy-score                 # loop every 30 minutes
    uv run caddy-score --once          # single run
    uv run caddy-score --limit 5       # max posts scored per run
    uv run caddy-score --interval 15   # minutes between runs

The daemon itself only sees the caller's scopes (via the API token). Each
deployment typically runs one instance with a service-account token that
has list access across users; the server enforces per-user
``Profile.auto_score`` when deciding whether to actually score a given post.
"""

import argparse
import asyncio
import logging
import os
import sys

import yaml
from datetime import datetime
from pathlib import Path

# Ensure ai/ root is on sys.path for imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.api_tools import (  # noqa: E402
    ApiClient,
    get_scrapes,
    score_job_post,
)
from lib.logfire_setup import setup_logfire  # noqa: E402

setup_logfire("score_poller")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("score_poller")


def _api_client() -> ApiClient:
    token = os.environ["CC_API_TOKEN"]
    base_url = os.environ.get("CC_API_BASE_URL", "http://localhost:8000")
    return ApiClient(base_url, token)


def _job_post_id_from_scrape(row: dict) -> int | None:
    rels = row.get("relationships") or {}
    jp = (rels.get("job-post") or rels.get("job_post") or {}).get("data") or {}
    raw = jp.get("id")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


async def _collect_candidates(api: ApiClient, limit: int) -> list[int]:
    """Return job_post ids whose scrape completed but have no score yet.

    Relies on the API's ``filter[has_score]=false`` on the scrapes endpoint —
    server-side join to Score keeps the daemon from N+1ing its way through a
    per-post score lookup.
    """
    try:
        raw = await get_scrapes(
            api,
            status="completed",
            has_score=False,
            sort="-scraped_at",
            per_page=limit * 2,
        )
        resp = yaml.safe_load(raw)
    except Exception as exc:
        logger.error("Scrape query failed: %s", exc)
        return []

    if not isinstance(resp, dict) or resp.get("error"):
        logger.warning("Scrape query unsuccessful: %s", (resp or {}).get("error") if isinstance(resp, dict) else raw[:200])
        return []

    rows = resp.get("data") or []
    out: list[int] = []
    seen: set[int] = set()
    for row in rows:
        post_id = _job_post_id_from_scrape(row)
        if post_id is None or post_id in seen:
            continue
        seen.add(post_id)
        out.append(post_id)
        if len(out) >= limit:
            break
    return out


async def _score_one(api: ApiClient, job_post_id: int) -> bool:
    try:
        raw = await score_job_post(api, job_post_id)
        resp = yaml.safe_load(raw)
    except Exception:
        logger.exception("Scoring raised for job_post %s", job_post_id)
        return False
    if not isinstance(resp, dict) or resp.get("error"):
        logger.warning(
            "Scoring failed for job_post %s: %s",
            job_post_id,
            (resp or {}).get("error") if isinstance(resp, dict) else raw[:200],
        )
        return False
    logger.info("  scored job_post %s", job_post_id)
    return True


async def run_once(limit: int = 5) -> str:
    logger.info("Starting score_poller run (limit=%d)", limit)
    api = _api_client()
    candidates = await _collect_candidates(api, limit=limit)
    logger.info("Found %d post(s) eligible for scoring", len(candidates))
    if not candidates:
        return "No posts to score."

    ok = 0
    for post_id in candidates:
        if await _score_one(api, post_id):
            ok += 1

    summary = f"Scored {ok}/{len(candidates)} job-post(s)"
    logger.info("%s", summary)
    return summary


async def loop(interval_minutes: int, limit: int) -> None:
    while True:
        start = datetime.now()
        try:
            await run_once(limit=limit)
        except Exception:
            logger.exception("Run failed")
        elapsed = (datetime.now() - start).total_seconds()
        sleep_secs = max(0.0, interval_minutes * 60 - elapsed)
        logger.info("Next run in %.0f minutes", sleep_secs / 60)
        try:
            await asyncio.sleep(sleep_secs)
        except asyncio.CancelledError:
            break


def run():
    parser = argparse.ArgumentParser(description="Auto-score job-posts whose scrape has completed.")
    parser.add_argument("--once", action="store_true", help="Run a single pass and exit")
    parser.add_argument("--limit", type=int, default=5, metavar="N", help="Max posts per run (default: 5)")
    parser.add_argument("--interval", type=int, default=30, metavar="MINUTES", help="Loop interval (default: 30)")
    args = parser.parse_args()

    try:
        if args.once:
            asyncio.run(run_once(limit=args.limit))
        else:
            asyncio.run(loop(args.interval, limit=args.limit))
    except KeyboardInterrupt:
        logger.info("Interrupted — exiting.")


if __name__ == "__main__":
    run()
