"""Tier-0 wiring tests for the JSON-LD JobPosting extractor (CC-27).

Tier0CSS now tries the deterministic ($0, no-LLM) JSON-LD parser BEFORE
the per-host CSS selectors:

  1. JSON-LD present + complete → $0 hit, routes to EvaluateExtraction,
     no LLM tier invoked. Needs NO profile selectors.
  2. JSON-LD present alongside CSS selectors → JSON-LD wins (method
     'jsonld' on the trace).
  3. JSON-LD absent + CSS selectors present + complete → CSS fallback
     (method 'css').
  4. JSON-LD absent/partial/invalid + no selectors → soft skip → Tier1Mini
     (preserves the Phase-1b skeleton).

All behind the existing OFF feature flag — nothing in production touches
the graph yet. Network / LLM are mocked; no live api or model call.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from scrape_graph.nodes_extract import (
    EvaluateExtraction,
    Tier0CSS,
    Tier1Mini,
)
from scrape_graph.state import ScrapeGraphState


def _run(node, state: ScrapeGraphState):
    ctx = SimpleNamespace(state=state)
    return asyncio.run(node.run(ctx))


def _state(profile: dict | None = None, html: str | None = None) -> ScrapeGraphState:
    state = ScrapeGraphState(scrape_id=1, submitted_url="https://acme.com/job/1")
    state.profile = profile
    state.html = html
    return state


_JSONLD_HTML = (
    "<html><head>"
    '<script type="application/ld+json">'
    '{"@context": "https://schema.org", "@type": "JobPosting",'
    ' "title": "Senior Backend Engineer",'
    ' "description": "<p>We are looking for a backend engineer with 5+ years'
    ' building distributed systems and APIs.</p>",'
    ' "hiringOrganization": {"@type": "Organization", "name": "Acme Corp"},'
    ' "datePosted": "2026-06-01"}'
    "</script>"
    "</head><body><h1>chrome</h1></body></html>"
)

# CSS-only fixture (no ld+json) + the job_data selector map that resolves it.
_CSS_HTML = """
<html><body>
  <h1 class="job-title">CSS Title</h1>
  <div class="company-name">CSS Company</div>
  <section class="job-description">
    A real description with enough words to be a believable job posting body.
  </section>
</body></html>
"""
_CSS_JOB_DATA = {
    "title": "h1.job-title",
    "company_name": ".company-name",
    "description": ".job-description",
}


def test_jsonld_present_extracts_at_tier0_no_profile_no_llm():
    """JSON-LD present, NO profile selectors → Tier0 parses at $0,
    sets state.parsed, records a tier0 hit, makes NO LLM call, and routes
    to EvaluateExtraction."""
    state = _state(profile={"preferred_tier": "0"}, html=_JSONLD_HTML)
    with patch("scrape_graph.nodes_extract.httpx.post") as mock_post, patch(
        "scrape_graph.nodes_extract._call_llm_extract"
    ) as mock_llm:
        mock_post.return_value.status_code = 200
        next_node = _run(Tier0CSS(), state)

    assert isinstance(next_node, EvaluateExtraction)
    assert mock_llm.call_count == 0, "JSON-LD Tier0 must not invoke any LLM tier"
    assert state.parsed["title"] == "Senior Backend Engineer"
    assert state.parsed["company_name"] == "Acme Corp"
    assert state.parsed["description"].startswith("We are looking for")
    assert state.parsed["posted_date"] == "2026-06-01"

    tier0 = [t for t in state.tier_attempts if t.tier == "tier0"]
    assert len(tier0) == 1
    assert tier0[0].produced_output is True
    assert tier0[0].cost_usd == 0.0
    assert tier0[0].model is None
    # Trace records which deterministic path paid off.
    assert state.node_trace[-1].payload.get("method") == "jsonld"


def test_jsonld_wins_over_css_when_both_present():
    """When both JSON-LD and css_selectors.job_data resolve, JSON-LD is
    tried first and wins (method 'jsonld')."""
    state = _state(
        profile={"preferred_tier": "0", "job_data": _CSS_JOB_DATA},
        html=_JSONLD_HTML,
    )
    with patch("scrape_graph.nodes_extract.httpx.post") as mock_post, patch(
        "scrape_graph.nodes_extract._call_llm_extract"
    ) as mock_llm:
        mock_post.return_value.status_code = 200
        next_node = _run(Tier0CSS(), state)

    assert isinstance(next_node, EvaluateExtraction)
    assert mock_llm.call_count == 0
    assert state.parsed["title"] == "Senior Backend Engineer"  # from JSON-LD
    assert state.node_trace[-1].payload.get("method") == "jsonld"


def test_css_fallback_when_jsonld_absent():
    """No JSON-LD JobPosting in the HTML → fall through to the per-host
    CSS selector path (method 'css')."""
    state = _state(
        profile={"preferred_tier": "0", "job_data": _CSS_JOB_DATA},
        html=_CSS_HTML,
    )
    with patch("scrape_graph.nodes_extract.httpx.post") as mock_post, patch(
        "scrape_graph.nodes_extract._call_llm_extract"
    ) as mock_llm:
        mock_post.return_value.status_code = 200
        next_node = _run(Tier0CSS(), state)

    assert isinstance(next_node, EvaluateExtraction)
    assert mock_llm.call_count == 0
    assert state.parsed["title"] == "CSS Title"
    assert state.parsed["company_name"] == "CSS Company"
    assert state.node_trace[-1].payload.get("method") == "css"

    tier0 = [t for t in state.tier_attempts if t.tier == "tier0"]
    assert len(tier0) == 1
    assert tier0[0].produced_output is True


def test_jsonld_incomplete_and_no_selectors_skips_to_tier1():
    """A JSON-LD JobPosting missing the description (incomplete) AND no
    css_selectors.job_data → soft skip → Tier1Mini, recording a tier0
    miss (preserves the skeleton behavior)."""
    incomplete = (
        "<html><head>"
        '<script type="application/ld+json">'
        '{"@type": "JobPosting", "title": "Half Posting",'
        ' "hiringOrganization": {"name": "PartialCo"}}'
        "</script>"
        "</head><body></body></html>"
    )
    state = _state(profile={"preferred_tier": "auto"}, html=incomplete)
    next_node = _run(Tier0CSS(), state)

    assert isinstance(next_node, Tier1Mini)
    tier0 = [t for t in state.tier_attempts if t.tier == "tier0"]
    assert len(tier0) == 1
    assert tier0[0].produced_output is False


def test_invalid_jsonld_and_no_selectors_skips_to_tier1():
    """Malformed ld+json + no selectors → fail-soft → Tier1Mini."""
    bad = (
        "<html><head>"
        '<script type="application/ld+json">{ not valid json ,,, }</script>'
        "</head><body></body></html>"
    )
    state = _state(profile=None, html=bad)
    next_node = _run(Tier0CSS(), state)

    assert isinstance(next_node, Tier1Mini)
    assert state.tier_attempts[-1].produced_output is False
