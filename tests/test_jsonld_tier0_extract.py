"""Graph-level tests for the JSON-LD Tier-0 extraction path.

Companion to test_preferred_tier_extract_entry.py. A page that carries a
schema.org JobPosting JSON-LD block must extract deterministically at
Tier-0 ($0, no LLM) — even when the per-host profile carries NO
css_selectors.job_data — and JSON-LD must win over CSS when both are
present (it's the higher-fidelity, churn-resistant path).

All gated behind the existing OFF feature flag — nothing in production
touches the graph yet.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from scrape_graph.nodes_extract import EvaluateExtraction, Tier0CSS, Tier1Mini
from scrape_graph.state import ScrapeGraphState


def _run(node, state: ScrapeGraphState):
    ctx = SimpleNamespace(state=state)
    return asyncio.run(node.run(ctx))


def _state(profile: dict | None = None, html: str | None = None) -> ScrapeGraphState:
    state = ScrapeGraphState(scrape_id=1, submitted_url="https://gov.example/job/1")
    state.profile = profile
    state.html = html
    return state


# A governmentjobs.com/NEOGOV-shaped page: full JobPosting JSON-LD, and
# deliberately NO css_selectors.job_data on the profile — the JSON-LD
# path must carry the extraction on its own.
_NEOGOV_HTML = """
<html><head>
<script type="application/ld+json">
{
  "@context": "https://schema.org/",
  "@type": "JobPosting",
  "title": "Civil Engineer II",
  "description": "<p>Design and review municipal infrastructure projects across the county for a growing public works department.</p>",
  "datePosted": "2026-06-18",
  "hiringOrganization": {"@type": "Organization", "name": "City of Springfield"},
  "jobLocation": {"address": {"addressLocality": "Springfield", "addressRegion": "IL"}},
  "baseSalary": {"@type": "MonetaryAmount", "currency": "USD",
    "value": {"@type": "QuantitativeValue", "minValue": 78000, "maxValue": 96000, "unitText": "YEAR"}}
}
</script>
</head><body><h1>Civil Engineer II</h1></body></html>
"""


def test_tier0_jsonld_extracts_without_profile_selectors_no_llm():
    """JSON-LD page, profile with preferred_tier='auto' but NO job_data
    selectors → Tier0 extracts at $0, sets state.parsed (incl. the
    JSON-LD-only salary/posted_date extras), records a tier0 hit, makes
    NO LLM call, and routes to EvaluateExtraction."""
    state = _state(profile={"preferred_tier": "auto"}, html=_NEOGOV_HTML)
    with patch("scrape_graph.nodes_extract.httpx.post") as mock_post, patch(
        "scrape_graph.nodes_extract._call_llm_extract"
    ) as mock_llm:
        mock_post.return_value.status_code = 200
        next_node = _run(Tier0CSS(), state)

    assert isinstance(next_node, EvaluateExtraction)
    assert mock_llm.call_count == 0, "JSON-LD Tier0 must not invoke any LLM tier"
    assert state.parsed["title"] == "Civil Engineer II"
    assert state.parsed["company_name"] == "City of Springfield"
    assert state.parsed["description"].startswith("Design and review")
    assert state.parsed["location"] == "Springfield, IL"
    # JSON-LD-only extras the CSS path can't deliver.
    assert state.parsed["salary_min"] == 78000
    assert state.parsed["salary_max"] == 96000
    assert state.parsed["posted_date"] == "2026-06-18"

    tier0 = [t for t in state.tier_attempts if t.tier == "tier0"]
    assert len(tier0) == 1
    assert tier0[0].produced_output is True
    assert tier0[0].cost_usd == 0.0
    assert tier0[0].model is None


def test_tier0_jsonld_trace_records_method():
    state = _state(profile={"preferred_tier": "auto"}, html=_NEOGOV_HTML)
    with patch("scrape_graph.nodes_extract.httpx.post") as mock_post:
        mock_post.return_value.status_code = 200
        _run(Tier0CSS(), state)
    last = state.node_trace[-1]
    assert last.payload.get("method") == "jsonld"


def test_tier0_jsonld_wins_over_css_when_both_present():
    """When the page has both a JobPosting JSON-LD block AND the profile
    carries job_data selectors, JSON-LD is used (higher fidelity)."""
    html = (
        _NEOGOV_HTML
        + '<div class="legacy-title">STALE CSS TITLE</div>'
    )
    state = _state(
        profile={
            "preferred_tier": "auto",
            "job_data": {
                "title": ".legacy-title",
                "company_name": ".legacy-company",
                "description": ".legacy-desc",
            },
        },
        html=html,
    )
    with patch("scrape_graph.nodes_extract.httpx.post") as mock_post:
        mock_post.return_value.status_code = 200
        next_node = _run(Tier0CSS(), state)

    assert isinstance(next_node, EvaluateExtraction)
    assert state.parsed["title"] == "Civil Engineer II"  # JSON-LD, not the CSS div
    assert state.node_trace[-1].payload.get("method") == "jsonld"


def test_tier0_falls_back_to_css_when_jsonld_absent():
    """No JSON-LD on the page → the CSS-selector path still runs and
    succeeds (regression guard: JSON-LD is additive, not a replacement)."""
    html = (
        "<html><body>"
        '<h1 class="job-title">Backend Engineer</h1>'
        '<div class="company-name">Acme Corp</div>'
        '<section class="job-description">'
        "Build distributed systems with a strong Python background here."
        "</section>"
        "</body></html>"
    )
    state = _state(
        profile={
            "preferred_tier": "auto",
            "job_data": {
                "title": "h1.job-title",
                "company_name": ".company-name",
                "description": ".job-description",
            },
        },
        html=html,
    )
    with patch("scrape_graph.nodes_extract.httpx.post") as mock_post:
        mock_post.return_value.status_code = 200
        next_node = _run(Tier0CSS(), state)

    assert isinstance(next_node, EvaluateExtraction)
    assert state.parsed["title"] == "Backend Engineer"
    assert state.node_trace[-1].payload.get("method") == "css"


def test_tier0_incomplete_jsonld_no_selectors_skips_to_tier1():
    """A JobPosting JSON-LD missing company/description, and no job_data
    selectors → tier0 miss, soft fall-through to Tier1 (no hard fail)."""
    html = (
        '<html><head>'
        '<script type="application/ld+json">'
        '{"@type": "JobPosting", "title": "Only A Title"}</script>'
        '</head><body></body></html>'
    )
    state = _state(profile={"preferred_tier": "auto"}, html=html)
    next_node = _run(Tier0CSS(), state)

    assert isinstance(next_node, Tier1Mini)
    tier0 = [t for t in state.tier_attempts if t.tier == "tier0"]
    assert len(tier0) == 1
    assert tier0[0].produced_output is False
