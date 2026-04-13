"""Fire-and-forget AI usage reporting to the Career Caddy API."""

import logging
import os
from urllib.parse import urljoin

import httpx

logger = logging.getLogger(__name__)

CC_API_BASE_URL = os.environ.get("CC_API_BASE_URL", "http://localhost:8000")


async def report_usage(
    api_token: str,
    agent_name: str,
    model_name: str,
    usage,
    trigger: str,
    pipeline_run_id: str | None = None,
    base_url: str | None = None,
) -> None:
    """POST usage data to /api/v1/ai-usages/. Errors are logged, never raised."""
    base = base_url or CC_API_BASE_URL
    payload = {
        "data": {
            "type": "ai-usage",
            "attributes": {
                "agent_name": agent_name,
                "model_name": model_name,
                "request_tokens": getattr(usage, "request_tokens", 0) or 0,
                "response_tokens": getattr(usage, "response_tokens", 0) or 0,
                "total_tokens": getattr(usage, "total_tokens", 0) or 0,
                "request_count": getattr(usage, "requests", 1) or 1,
                "trigger": trigger,
                "pipeline_run_id": str(pipeline_run_id) if pipeline_run_id else None,
            },
        }
    }
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/vnd.api+json",
    }
    logger.info(
        "Reporting usage: agent=%s model=%s tokens=%s/%s/%s trigger=%s",
        agent_name, model_name,
        payload["data"]["attributes"]["request_tokens"],
        payload["data"]["attributes"]["response_tokens"],
        payload["data"]["attributes"]["total_tokens"],
        trigger,
    )
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                urljoin(base, "/api/v1/ai-usages/"),
                json=payload,
                headers=headers,
                timeout=10.0,
            )
            if resp.status_code not in (200, 201):
                logger.warning(
                    "Usage report failed: %s %s", resp.status_code, resp.text[:200]
                )
            else:
                logger.info("Usage report success: %s", resp.status_code)
    except Exception:
        logger.exception("Failed to report AI usage")
