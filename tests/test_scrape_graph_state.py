"""Minimal tests for the scrape-graph skeleton.

Focuses on state + url canonicalization + graph wiring. LLM-heavy
node tests land with Phase 1c/1d when the api-side /llm-extract/
endpoint is live.
"""
from __future__ import annotations

from lib.scrape_graph import (
    ScrapeGraphState,
    canonicalize_url,
)
from lib.scrape_graph.nodes_scrape import _flatten_profile_attrs
from lib.scrape_graph.url_canonicalize import apply_url_rewrites, urls_differ


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
# Profile-driven URL rewrites
# ----------------------------------------------------------------------


def test_url_rewrite_indeed_vjk_to_viewjob():
    """Real case: Indeed `?vjk=X` tracker lands on the homepage
    instead of the job. Rewriting to `/viewjob?jk=X` fixes scrape 175.
    """
    rules = [
        {
            "match": r"^https?://(?:www\.)?indeed\.com/\?[^#]*\bvjk=([A-Za-z0-9]+)",
            "rewrite": r"https://www.indeed.com/viewjob?jk=\1",
        }
    ]
    url = "https://www.indeed.com/?advn=4035278867423431&vjk=d6b7eeeee82aeb6b"
    assert apply_url_rewrites(url, rules) == "https://www.indeed.com/viewjob?jk=d6b7eeeee82aeb6b"


def test_url_rewrite_no_match_returns_input_unchanged():
    rules = [{"match": r"does-not-match", "rewrite": "/other"}]
    assert apply_url_rewrites("https://x.com/job/1", rules) == "https://x.com/job/1"


def test_url_rewrite_skips_invalid_regex():
    rules = [
        {"match": "[unclosed", "rewrite": "wont-fire"},
        {"match": r"https://x\.com/a", "rewrite": "https://x.com/b"},
    ]
    assert apply_url_rewrites("https://x.com/a", rules) == "https://x.com/b"


def test_url_rewrite_none_and_empty():
    assert apply_url_rewrites("https://x.com/a", None) == "https://x.com/a"
    assert apply_url_rewrites("https://x.com/a", []) == "https://x.com/a"
    assert apply_url_rewrites("", [{"match": ".*", "rewrite": "x"}]) == ""


# ----------------------------------------------------------------------
# State dataclass
# ----------------------------------------------------------------------


def test_flatten_profile_attrs_lifts_css_selectors():
    """LoadProfile's flatten lifts css_selectors keys to the top of profile.

    Regression: scrape 202 hit the linkedin account-chooser, but
    ObstacleRememberMe routed straight to ObstacleWaitRetry with
    duration_ms=0 because the node read profile["rememberme_candidates"]
    while LoadProfile stored the api attributes verbatim — leaving
    rememberme_candidates nested under css_selectors. Flatten on load
    so every reader sees one dict.
    """
    attrs = {
        "hostname": "linkedin.com",
        "css_selectors": {
            "rememberme_candidates": ["button.member-profile__details"],
            "ready_selector": ".jobs-details",
        },
        "apply_resolver_config": {"apply_link_selectors": ["a.jobs-apply-link"]},
        "extraction_hints": "...",
    }
    flat = _flatten_profile_attrs(attrs)
    assert flat["rememberme_candidates"] == ["button.member-profile__details"]
    assert flat["ready_selector"] == ".jobs-details"
    assert flat["apply_resolver_config"] == {"apply_link_selectors": ["a.jobs-apply-link"]}
    assert "css_selectors" not in flat


def test_flatten_profile_attrs_top_level_wins_on_collision():
    """Top-level fields take precedence over a same-named css_selectors entry."""
    attrs = {
        "extraction_hints": "top-level",
        "css_selectors": {"extraction_hints": "nested"},
    }
    flat = _flatten_profile_attrs(attrs)
    assert flat["extraction_hints"] == "top-level"


def test_flatten_profile_attrs_no_css_selectors():
    """Profiles without a css_selectors blob pass through unchanged."""
    attrs = {"hostname": "indeed.com", "url_rewrites": [{"match": "x", "rewrite": "y"}]}
    flat = _flatten_profile_attrs(dict(attrs))
    assert flat == attrs


def test_flatten_profile_attrs_handles_non_dict():
    assert _flatten_profile_attrs(None) == {}  # type: ignore[arg-type]
    assert _flatten_profile_attrs("garbage") == {}  # type: ignore[arg-type]


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
