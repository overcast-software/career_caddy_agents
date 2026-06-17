"""Tests for ScrapeProfile.preferred_tier wiring into the extract entry.

CC epic #19 child #21. Two behaviors:

  1. StartExtract routes on the per-host profile's preferred_tier so a
     known-good domain enters at the tier it needs ('0'/'auto'/missing →
     Tier0CSS, '1' → Tier1Mini, '2' → Tier2Haiku, '3' → Tier3Sonnet).
  2. Tier0CSS does real $0 CSS extraction when the profile carries
     css_selectors.job_data AND captured HTML is present; otherwise it
     preserves the Phase-1b skeleton behavior (soft skip → Tier1Mini).

All gated behind the existing OFF feature flag — nothing in production
touches the graph yet.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from scrape_graph.nodes_extract import (
    EvaluateExtraction,
    StartExtract,
    Tier0CSS,
    Tier1Mini,
    Tier2Haiku,
    Tier3Sonnet,
)
from scrape_graph.state import ScrapeGraphState


def _run(node, state: ScrapeGraphState):
    ctx = SimpleNamespace(state=state)
    return asyncio.run(node.run(ctx))


_JOB_HTML = """
<html><body>
  <h1 class="job-title">Senior Backend Engineer</h1>
  <div class="company-name">Acme Corp</div>
  <section class="job-description">
    We are looking for a backend engineer with 5+ years of experience
    building distributed systems and APIs. Strong Python background.
  </section>
  <span class="job-location">Remote — US</span>
</body></html>
"""

_JOB_DATA = {
    "title": "h1.job-title",
    "company_name": ".company-name",
    "description": ".job-description",
    "location": ".job-location",
}


def _state(profile: dict | None = None, html: str | None = None) -> ScrapeGraphState:
    state = ScrapeGraphState(scrape_id=1, submitted_url="https://acme.com/job/1")
    state.profile = profile
    state.html = html
    return state


# --------------------------------------------------------------------------
# StartExtract routing on preferred_tier
# --------------------------------------------------------------------------

def test_start_extract_auto_routes_to_tier0():
    next_node = _run(StartExtract(), _state(profile={"preferred_tier": "auto"}))
    assert isinstance(next_node, Tier0CSS)


def test_start_extract_zero_routes_to_tier0():
    next_node = _run(StartExtract(), _state(profile={"preferred_tier": "0"}))
    assert isinstance(next_node, Tier0CSS)


def test_start_extract_missing_preferred_tier_routes_to_tier0():
    # No profile at all → default ladder entry.
    assert isinstance(_run(StartExtract(), _state(profile=None)), Tier0CSS)
    # Profile present but no preferred_tier key.
    assert isinstance(_run(StartExtract(), _state(profile={})), Tier0CSS)


def test_start_extract_tier1():
    next_node = _run(StartExtract(), _state(profile={"preferred_tier": "1"}))
    assert isinstance(next_node, Tier1Mini)


def test_start_extract_tier2_accepts_int():
    # api may serialize preferred_tier as an int; coercion must hold.
    next_node = _run(StartExtract(), _state(profile={"preferred_tier": 2}))
    assert isinstance(next_node, Tier2Haiku)


def test_start_extract_tier3():
    next_node = _run(StartExtract(), _state(profile={"preferred_tier": "3"}))
    assert isinstance(next_node, Tier3Sonnet)


def test_start_extract_unknown_tier_defaults_to_tier0():
    next_node = _run(StartExtract(), _state(profile={"preferred_tier": "99"}))
    assert isinstance(next_node, Tier0CSS)


def test_start_extract_records_preferred_tier_in_trace():
    state = _state(profile={"preferred_tier": "2"})
    _run(StartExtract(), state)
    last = state.node_trace[-1]
    assert last.payload.get("preferred_tier") == "2"


# --------------------------------------------------------------------------
# Tier0CSS — $0 deterministic CSS extraction
# --------------------------------------------------------------------------

def test_tier0_full_parse_no_llm():
    """preferred_tier='0' + job_data selectors + html → Tier0 parses at
    $0, sets state.parsed, records a tier0 hit, and makes NO LLM call.

    httpx.post is stubbed (tracing fires a best-effort graph-transition
    POST through it) so the test is hermetic; the cost assertion is on
    _call_llm_extract, the only path that spends LLM tokens.
    """
    state = _state(
        profile={"preferred_tier": "0", "job_data": _JOB_DATA},
        html=_JOB_HTML,
    )
    with patch("scrape_graph.nodes_extract.httpx.post") as mock_post, patch(
        "scrape_graph.nodes_extract._call_llm_extract"
    ) as mock_llm:
        mock_post.return_value.status_code = 200
        next_node = _run(Tier0CSS(), state)

    assert isinstance(next_node, EvaluateExtraction)
    assert mock_llm.call_count == 0, "Tier0 must not invoke any LLM tier"
    assert state.parsed["title"] == "Senior Backend Engineer"
    assert state.parsed["company_name"] == "Acme Corp"
    assert state.parsed["description"].startswith("We are looking for")

    tier0 = [t for t in state.tier_attempts if t.tier == "tier0"]
    assert len(tier0) == 1
    assert tier0[0].produced_output is True
    assert tier0[0].cost_usd == 0.0
    assert tier0[0].model is None


def test_tier0_strict_miss_falls_through_to_tier1():
    """preferred_tier='0' with job_data selectors that don't fully resolve
    (description selector misses) records a tier0 miss and falls through
    to Tier1 — it must NOT hard-fail, so the api learning loop can
    auto-demote after sustained misses."""
    html_no_desc = (
        "<html><body>"
        '<h1 class="job-title">Engineer</h1>'
        '<div class="company-name">Acme Corp</div>'
        "</body></html>"
    )
    state = _state(
        profile={"preferred_tier": "0", "job_data": _JOB_DATA},
        html=html_no_desc,
    )
    next_node = _run(Tier0CSS(), state)

    assert isinstance(next_node, Tier1Mini)
    tier0 = [t for t in state.tier_attempts if t.tier == "tier0"]
    assert len(tier0) == 1
    assert tier0[0].produced_output is False


def test_tier0_no_job_data_preserves_skeleton_skip():
    """No job_data selectors → preserve the Phase-1b no-op skip → Tier1
    (regression guard for the existing skeleton)."""
    state = _state(profile={"preferred_tier": "auto"}, html=_JOB_HTML)
    next_node = _run(Tier0CSS(), state)

    assert isinstance(next_node, Tier1Mini)
    tier0 = [t for t in state.tier_attempts if t.tier == "tier0"]
    assert len(tier0) == 1
    assert tier0[0].produced_output is False


def test_tier0_no_html_skips_to_tier1():
    """job_data present but no captured HTML (e.g. paste-ingest with no
    page) → soft skip → Tier1 rather than crashing."""
    state = _state(
        profile={"preferred_tier": "0", "job_data": _JOB_DATA}, html=None,
    )
    next_node = _run(Tier0CSS(), state)

    assert isinstance(next_node, Tier1Mini)
    assert state.tier_attempts[-1].produced_output is False
