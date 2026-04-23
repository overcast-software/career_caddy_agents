"""Extract-side nodes. Phase 1b skeleton — tier nodes call api's
llm-extract endpoint (lands in Phase 1c when wired); EvaluateExtraction
escalates through Tier1→Tier2; PersistJobPost POSTs parsed data back
to api for dedup/posted_date/stub-merge handling.
"""
# ruff: noqa: F811
# The forward-declare-then-redefine pattern below is how we give
# pydantic-graph's get_type_hints enough info to resolve Union[...]
# return annotations at class-body time. The second definition is the
# real node; the stubs are intentional.
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Union
from urllib.parse import urlparse

import httpx
from pydantic_graph import BaseNode, End, GraphRunContext

from .state import ScrapeGraphState, TierAttempt
from .tracing import trace_node

logger = logging.getLogger(__name__)


_TIER1_MODEL = os.environ.get("SCRAPE_GRAPH_TIER1_MODEL", "openai:gpt-4o-mini")
_TIER2_MODEL = os.environ.get("SCRAPE_GRAPH_TIER2_MODEL", "anthropic:claude-haiku-4-5")
_TIER3_MODEL = os.environ.get("SCRAPE_GRAPH_TIER3_MODEL", "anthropic:claude-sonnet-4-6")
_STUB_MIN_WORDS = 60


def _api_base() -> str:
    return os.environ.get("CC_API_BASE_URL", "").rstrip("/")


def _api_headers() -> dict[str, str]:
    token = os.environ.get("CC_API_TOKEN", "")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _call_llm_extract(
    state: ScrapeGraphState, tier_label: str, model_spec: str,
) -> dict | None:
    """Call api's llm-extract endpoint (Phase 1c), records TierAttempt.

    Returns parsed dict on success, None on failure. The endpoint
    doesn't exist yet in Phase 1a; 404 is recorded as a soft failure
    so Phase 1b skeleton tests can run without a live api sidecar.
    """
    t0 = time.time()
    error: str | None = None
    parsed_dict: dict | None = None
    try:
        resp = httpx.post(
            f"{_api_base()}/api/v1/scrapes/{state.scrape_id}/llm-extract/",
            json={"model": model_spec},
            headers={**_api_headers(), "Content-Type": "application/json"},
            timeout=120.0,
        )
        if resp.status_code == 200:
            parsed_dict = (resp.json() or {}).get("data", {}).get("attributes") or None
        elif resp.status_code == 404:
            error = "llm-extract endpoint not deployed yet"
        else:
            error = f"HTTP {resp.status_code}"
    except Exception as exc:
        error = repr(exc)
        logger.warning("%s LLM call failed: %s", tier_label, exc)
    duration_ms = int((time.time() - t0) * 1000)
    state.tier_attempts.append(
        TierAttempt(
            tier=tier_label,
            model=model_spec,
            duration_ms=duration_ms,
            produced_output=parsed_dict is not None,
            error=error,
        )
    )
    return parsed_dict


# Forward refs
class Tier0CSS(BaseNode[ScrapeGraphState, None, dict]):
    pass


class Tier1Mini(BaseNode[ScrapeGraphState, None, dict]):
    pass


class Tier2Haiku(BaseNode[ScrapeGraphState, None, dict]):
    pass


class Tier3Sonnet(BaseNode[ScrapeGraphState, None, dict]):
    pass


class EvaluateExtraction(BaseNode[ScrapeGraphState, None, dict]):
    pass


class PersistJobPost(BaseNode[ScrapeGraphState, None, dict]):
    pass


class UpdateProfile(BaseNode[ScrapeGraphState, None, dict]):
    pass


class ResolveApplyUrl(BaseNode[ScrapeGraphState, None, dict]):
    pass


class ExtractFail(BaseNode[ScrapeGraphState, None, dict]):
    pass


@dataclass
class StartExtract(BaseNode[ScrapeGraphState, None, dict]):
    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> Tier0CSS:
        started = time.time()
        trace_node(ctx.state, "StartExtract", "Tier0CSS", started)
        return Tier0CSS()


@dataclass
class Tier0CSS(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> Tier1Mini:
        # Phase 1b: Tier0 is still done server-side via legacy
        # parse_scrape; graph skips it and lets Tier1 start. Replace
        # when the api's /tier0-extract/ endpoint ships.
        started = time.time()
        ctx.state.tier_attempts.append(
            TierAttempt(tier="tier0", produced_output=False, model=None)
        )
        trace_node(ctx.state, "Tier0CSS", "Tier1Mini", started)
        return Tier1Mini()


@dataclass
class Tier1Mini(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> EvaluateExtraction:
        started = time.time()
        parsed = _call_llm_extract(ctx.state, "tier1", _TIER1_MODEL)
        if parsed:
            ctx.state.parsed = parsed
        trace_node(ctx.state, "Tier1Mini", "EvaluateExtraction", started)
        return EvaluateExtraction()


@dataclass
class Tier2Haiku(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> EvaluateExtraction:
        started = time.time()
        parsed = _call_llm_extract(ctx.state, "tier2", _TIER2_MODEL)
        if parsed:
            ctx.state.parsed = parsed
        trace_node(ctx.state, "Tier2Haiku", "EvaluateExtraction", started)
        return EvaluateExtraction()


@dataclass
class Tier3Sonnet(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    """Wired but disabled — EvaluateExtraction gates on an env flag."""

    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> EvaluateExtraction:
        started = time.time()
        parsed = _call_llm_extract(ctx.state, "tier3", _TIER3_MODEL)
        if parsed:
            ctx.state.parsed = parsed
        trace_node(ctx.state, "Tier3Sonnet", "EvaluateExtraction", started)
        return EvaluateExtraction()


@dataclass
class EvaluateExtraction(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> Union[PersistJobPost, Tier2Haiku, Tier3Sonnet, ExtractFail]:
        started = time.time()
        state = ctx.state
        parsed = state.parsed or {}
        reasons: list[str] = []
        title = (parsed.get("title") or "").strip()
        company = (parsed.get("company_name") or "").strip()
        description = (parsed.get("description") or "").strip()
        if not title:
            reasons.append("missing_title")
        if not company:
            reasons.append("missing_company")
        if description and len(description.split()) < _STUB_MIN_WORDS:
            reasons.append("thin_description")
        passed = not reasons
        state.evaluation = {"passed": passed, "reasons": reasons}

        last_tier = state.tier_attempts[-1].tier if state.tier_attempts else ""
        tier3_enabled = os.environ.get("SCRAPE_GRAPH_ENABLE_TIER3") == "1"

        if passed:
            trace_node(state, "EvaluateExtraction", "PersistJobPost", started)
            return PersistJobPost()
        if last_tier in ("tier0", "tier1"):
            trace_node(state, "EvaluateExtraction", "Tier2Haiku", started)
            return Tier2Haiku()
        if last_tier == "tier2" and tier3_enabled:
            trace_node(state, "EvaluateExtraction", "Tier3Sonnet", started)
            return Tier3Sonnet()
        trace_node(state, "EvaluateExtraction", "ExtractFail", started)
        return ExtractFail()


@dataclass
class PersistJobPost(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> Union[UpdateProfile, ExtractFail]:
        started = time.time()
        state = ctx.state
        try:
            resp = httpx.post(
                f"{_api_base()}/api/v1/scrapes/{state.scrape_id}/persist-extraction/",
                json={"attributes": state.parsed or {}},
                headers={**_api_headers(), "Content-Type": "application/json"},
                timeout=60.0,
            )
            body = resp.json() if resp.status_code < 500 else {}
            meta = (body or {}).get("meta") or {}
            state.job_post_id = meta.get("job_post_id")
            state.was_duplicate = (meta.get("outcome") == "duplicate")
        except Exception:
            logger.warning("PersistJobPost: post failed", exc_info=True)
            trace_node(state, "PersistJobPost", "ExtractFail", started)
            return ExtractFail()
        trace_node(state, "PersistJobPost", "UpdateProfile", started)
        return UpdateProfile()


@dataclass
class UpdateProfile(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> ResolveApplyUrl:
        started = time.time()
        state = ctx.state
        host = (urlparse(state.canonical_url or state.submitted_url or "").hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        if host:
            try:
                tier0_hit = any(
                    t.tier == "tier0" and t.produced_output
                    for t in state.tier_attempts
                )
                httpx.post(
                    f"{_api_base()}/api/v1/scrape-profiles/{host}/update-from-outcome/",
                    json={
                        "scrape_id": state.scrape_id,
                        "success": bool(state.job_post_id),
                        "tier0_hit": tier0_hit,
                    },
                    headers={**_api_headers(), "Content-Type": "application/json"},
                    timeout=10.0,
                )
            except Exception:
                logger.debug("UpdateProfile: post failed", exc_info=True)
        trace_node(state, "UpdateProfile", "ResolveApplyUrl", started)
        return ResolveApplyUrl()


@dataclass
class ResolveApplyUrl(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    """Phase 1b: no-op. Phase 2 PROJ lands the real resolver here."""

    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> End[dict]:
        started = time.time()
        state = ctx.state
        state.outcome = "success"
        trace_node(state, "ResolveApplyUrl", "End", started)
        return End({
            "outcome": "success",
            "job_post_id": state.job_post_id,
            "scrape_id": state.scrape_id,
        })


@dataclass
class ExtractFail(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> End[dict]:
        started = time.time()
        state = ctx.state
        state.outcome = "failure"
        state.failure_reason = state.failure_reason or "extraction"
        trace_node(state, "ExtractFail", "End", started)
        return End({
            "outcome": "failure",
            "failure_reason": state.failure_reason,
            "scrape_id": state.scrape_id,
        })


__all__ = [
    "StartExtract",
    "Tier0CSS",
    "Tier1Mini",
    "Tier2Haiku",
    "Tier3Sonnet",
    "EvaluateExtraction",
    "PersistJobPost",
    "UpdateProfile",
    "ResolveApplyUrl",
    "ExtractFail",
]
