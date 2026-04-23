"""Minimal tests for the scrape-graph skeleton.

Focuses on state + url canonicalization + graph wiring. LLM-heavy
node tests land with Phase 1c/1d when the api-side /llm-extract/
endpoint is live.
"""
from __future__ import annotations

from lib.scrape_graph import (
    GraphMode,
    ScrapeGraphState,
    canonicalize_url,
    get_mode,
)
from lib.scrape_graph.url_canonicalize import urls_differ


# ----------------------------------------------------------------------
# URL canonicalization
# ----------------------------------------------------------------------


def test_strip_utm_params():
    url = "https://example.com/job?utm_source=email&utm_medium=cpc&id=42"
    assert canonicalize_url(url) == "https://example.com/job?id=42"


def test_strip_linkedin_tracker_params():
    url = (
        "https://www.linkedin.com/comm/jobs/view/4404587081/"
        "?trackingId=abc123&refId=foo&lipi=urn%3Ali"
    )
    cleaned = canonicalize_url(url)
    assert "trackingId" not in cleaned
    assert "refId" not in cleaned
    assert "lipi" not in cleaned
    assert "/comm/jobs/view/4404587081" in cleaned


def test_drops_fragment():
    assert canonicalize_url("https://x.com/path#abc") == "https://x.com/path"


def test_preserves_non_tracker_query():
    url = "https://ats.example/apply/1?locale=en&source=referral"
    assert canonicalize_url(url) == "https://ats.example/apply/1?locale=en&source=referral"


def test_urls_differ_tracker_to_same_destination():
    a = "https://click.example/redirect?trackingId=abc&url=foo"
    b = "https://click.example/redirect?url=foo"
    assert urls_differ(a, b) is False


def test_urls_differ_tracker_to_different_destination():
    tracker = "https://www.linkedin.com/comm/jobs/view/1?trackingId=abc"
    landed = "https://boards.greenhouse.io/acme/jobs/42"
    assert urls_differ(tracker, landed) is True


# ----------------------------------------------------------------------
# Mode reading
# ----------------------------------------------------------------------


def test_mode_defaults_primary(monkeypatch):
    """Post-cutover default is PRIMARY — the pydantic-graph pipeline
    runs for every scrape unless an operator explicitly opts out."""
    monkeypatch.delenv("SCRAPE_GRAPH_MODE", raising=False)
    assert get_mode() is GraphMode.PRIMARY


def test_mode_reads_env(monkeypatch):
    monkeypatch.setenv("SCRAPE_GRAPH_MODE", "shadow")
    assert get_mode() is GraphMode.SHADOW


def test_mode_off_kill_switch(monkeypatch):
    monkeypatch.setenv("SCRAPE_GRAPH_MODE", "off")
    assert get_mode() is GraphMode.OFF


def test_mode_invalid_falls_back_to_primary(monkeypatch):
    monkeypatch.setenv("SCRAPE_GRAPH_MODE", "nope")
    assert get_mode() is GraphMode.PRIMARY


# ----------------------------------------------------------------------
# State dataclass
# ----------------------------------------------------------------------


def test_state_defaults():
    state = ScrapeGraphState(scrape_id=42, submitted_url="https://x.com/")
    assert state.scrape_id == 42
    assert state.tier_attempts == []
    assert state.obstacle_history == []
    assert state.node_trace == []
    assert state.outcome is None


def test_state_payload_serializable():
    import json
    state = ScrapeGraphState(
        scrape_id=1, submitted_url="https://x.com/", original_scrape_id=1,
    )
    json.dumps(state.to_payload())  # should not raise


# ----------------------------------------------------------------------
# Graph wiring — just confirms the module imports + builds a graph
# ----------------------------------------------------------------------


def test_build_scrape_graph_smoke():
    from lib.scrape_graph.graph import build_scrape_graph, build_extract_graph

    full = build_scrape_graph()
    assert full is not None
    extract_only = build_extract_graph()
    assert extract_only is not None
