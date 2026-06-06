"""Phase B — scrape-graph fast path for source_mode='extension-direct'.

When the extension content-script POSTs a Scrape with source_mode=
'extension-direct' and a complete captured_payload (title, company,
description), the graph branches:

    StartScrape → SkipBrowserTier → [ResolveApplyUrl?] → PersistJobPost → End

…instead of the full browser-tier chain (Navigate → DetectObstacle →
… → Tier3Sonnet → ValidateExtraction → PersistJobPost). The api-side
JobPostExtractor.parse_scrape (called via /persist-extraction/) still
owns canonical_link / fingerprint / sticky-closed dedupe — the fast
path does NOT bypass the dedupe-first invariant; it bypasses only the
browser-tier extraction.

Tests cover four contracts the plan calls out:

1. Full payload + apply_url null → StartScrape, SkipBrowserTier,
   ResolveApplyUrl, PersistJobPost run; no browser-tier node enters.
2. Full payload + apply_url present → ResolveApplyUrl is NOT entered;
   apply_url passes through to PersistJobPost as parsed['link'].
3. Default source_mode='browser' (no captured_payload) → existing
   browser path runs end-to-end; SkipBrowserTier never enters.
4. source_mode='extension-direct' but missing captured_payload →
   defensive bail to ExtractFail; the graph does NOT silently
   degrade to the browser tier.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from pydantic_graph import End

from scrape_graph.nodes_extract import (
    ExtractFail,
    PersistJobPost,
    ResolveApplyUrl,
)
from scrape_graph.nodes_scrape import (
    LoadProfile,
    SkipBrowserTier,
    StartScrape,
    _captured_payload_is_complete,
)
from scrape_graph.state import ScrapeGraphState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(node, state: ScrapeGraphState):
    ctx = SimpleNamespace(state=state)
    return asyncio.run(node.run(ctx))


def _persist_response(job_post_id: int = 555, outcome: str = "created") -> MagicMock:
    """Mimic the /persist-extraction/ 200 response shape: the api returns
    {"meta": {"job_post_id": ..., "outcome": ...}, ...}. PersistJobPost
    reads job_post_id + (outcome == "duplicate") from meta."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "meta": {"job_post_id": job_post_id, "outcome": outcome},
    }
    return resp


def _full_payload(apply_url: str | None = None) -> dict:
    payload = {
        "title": "Staff Software Engineer",
        "company": "Acme Corp",
        "description": (
            "We're looking for a Staff Software Engineer to lead our "
            "platform team. You'll design, build, and operate "
            "distributed systems at scale. Responsibilities include "
            "architecting new services, mentoring engineers, and "
            "owning the on-call rotation. Required: 8+ years of "
            "backend experience, deep familiarity with Python, "
            "Go, or Java, and a track record of shipping production "
            "systems. Nice to have: Kubernetes, observability tooling."
        ),
        "location": "Remote — US",
    }
    if apply_url is not None:
        payload["apply_url"] = apply_url
    return payload


def _ext_state(payload: dict | None) -> ScrapeGraphState:
    state = ScrapeGraphState(
        scrape_id=42,
        original_scrape_id=42,
        submitted_url="https://example.com/jobs/abc",
        source="extension",
        source_mode="extension-direct",
        captured_payload=payload,
    )
    return state


# ---------------------------------------------------------------------------
# 1. State / payload-completeness predicate
# ---------------------------------------------------------------------------


def test_state_defaults_source_mode_browser():
    """Backwards-compat: every existing caller that doesn't set
    source_mode gets the legacy browser-tier path."""
    state = ScrapeGraphState(scrape_id=1, submitted_url="https://x.com/")
    assert state.source_mode == "browser"
    assert state.captured_payload is None


def test_captured_payload_is_complete_happy_path():
    assert _captured_payload_is_complete(_full_payload()) is True


def test_captured_payload_is_complete_rejects_missing_field():
    payload = _full_payload()
    del payload["title"]
    assert _captured_payload_is_complete(payload) is False


def test_captured_payload_is_complete_rejects_empty_field():
    payload = _full_payload()
    payload["company"] = "   "
    assert _captured_payload_is_complete(payload) is False


def test_captured_payload_is_complete_rejects_non_string():
    payload = _full_payload()
    payload["description"] = 12345  # type: ignore[assignment]
    assert _captured_payload_is_complete(payload) is False


def test_captured_payload_is_complete_rejects_non_dict():
    assert _captured_payload_is_complete(None) is False
    assert _captured_payload_is_complete("garbage") is False  # type: ignore[arg-type]
    assert _captured_payload_is_complete([]) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 2. StartScrape gate — four cases
# ---------------------------------------------------------------------------


def test_start_scrape_branches_to_skip_browser_tier_on_extension_direct():
    """source_mode='extension-direct' + full payload → SkipBrowserTier.
    LoadProfile is NEVER entered — no browser tier node sees this scrape."""
    state = _ext_state(_full_payload())

    with patch("scrape_graph.tracing._post_transition"):
        next_node = _run(StartScrape(), state)

    assert isinstance(next_node, SkipBrowserTier), (
        f"extension-direct must route to SkipBrowserTier, got {type(next_node).__name__}"
    )
    # The node_trace records the transition so the operator panel and
    # the per-scrape Transitions list show the fast-path branch.
    assert any(
        entry.node == "StartScrape" and entry.routed_to == "SkipBrowserTier"
        for entry in state.node_trace
    ), "StartScrape→SkipBrowserTier must appear in the node_trace"


def test_start_scrape_routes_to_load_profile_on_browser_default():
    """source_mode='browser' (the default) keeps the existing path —
    StartScrape → LoadProfile → Navigate → … browser-tier chain."""
    state = ScrapeGraphState(
        scrape_id=99,
        submitted_url="https://example.com/jobs/abc",
        source="poller",
        # source_mode left at default 'browser', captured_payload None.
    )

    with patch("scrape_graph.tracing._post_transition"):
        next_node = _run(StartScrape(), state)

    assert isinstance(next_node, LoadProfile), (
        f"browser-mode default must route to LoadProfile, got {type(next_node).__name__}"
    )


def test_start_scrape_bails_to_extract_fail_on_missing_payload():
    """Defensive: source_mode='extension-direct' but captured_payload is
    None (the Phase-A serializer should have rejected this at write
    time). The graph refuses to silently degrade to the browser tier
    — we route to ExtractFail so an operator sees the failure instead
    of mysteriously good content from a re-fetch."""
    state = _ext_state(None)

    with patch("scrape_graph.tracing._post_transition"):
        next_node = _run(StartScrape(), state)

    assert isinstance(next_node, ExtractFail), (
        f"missing payload must bail to ExtractFail, got {type(next_node).__name__}"
    )
    assert "extension-direct" in (state.failure_reason or "")
    assert "captured_payload" in (state.failure_reason or "")


def test_start_scrape_bails_to_extract_fail_on_incomplete_payload():
    """Same defensive bail when the payload is present but missing one
    of the required title/company/description fields."""
    payload = _full_payload()
    del payload["description"]
    state = _ext_state(payload)

    with patch("scrape_graph.tracing._post_transition"):
        next_node = _run(StartScrape(), state)

    assert isinstance(next_node, ExtractFail), (
        f"incomplete payload must bail to ExtractFail, got {type(next_node).__name__}"
    )
    assert "captured_payload" in (state.failure_reason or "")


# ---------------------------------------------------------------------------
# 3. SkipBrowserTier — payload mapping + routing
# ---------------------------------------------------------------------------


def test_skip_browser_tier_maps_payload_to_parsed_with_apply_url():
    """SkipBrowserTier translates captured_payload → ParsedJobData shape.
    Extension uses 'company' / 'apply_url'; api expects 'company_name' /
    'link'. The mapping happens here so persist-extraction sees the
    canonical vocabulary."""
    state = _ext_state(_full_payload(apply_url="https://acme.com/apply/abc"))

    with patch("scrape_graph.tracing._post_transition"), \
         patch("scrape_graph.nodes_scrape.httpx.patch") as mock_patch:
        next_node = _run(SkipBrowserTier(), state)

    assert state.parsed is not None
    assert state.parsed["title"] == "Staff Software Engineer"
    assert state.parsed["company_name"] == "Acme Corp"
    assert state.parsed["description"].startswith("We're looking for")
    assert state.parsed["location"] == "Remote — US"
    assert state.parsed["link"] == "https://acme.com/apply/abc"
    # apply_url present → straight to PersistJobPost; ResolveApplyUrl
    # is NOT entered.
    assert isinstance(next_node, PersistJobPost), (
        f"apply_url-present must route to PersistJobPost, got {type(next_node).__name__}"
    )
    # The scrape gets status='extracting' so the state machine matches
    # the browser-tier transition into the persistence phase.
    assert mock_patch.called


def test_skip_browser_tier_routes_through_resolve_apply_url_when_apply_url_null():
    """When the payload has no apply_url, route to ResolveApplyUrl so
    the graph has a chance to fill it from the browser tier — except
    there's no browser page on the fast path, so ResolveApplyUrl
    no-ops and chains to PersistJobPost."""
    state = _ext_state(_full_payload(apply_url=None))

    with patch("scrape_graph.tracing._post_transition"), \
         patch("scrape_graph.nodes_scrape.httpx.patch"):
        next_node = _run(SkipBrowserTier(), state)

    # state.parsed has no 'link' because payload had no apply_url.
    assert "link" not in state.parsed
    assert isinstance(next_node, ResolveApplyUrl), (
        f"apply_url-null must route to ResolveApplyUrl, got {type(next_node).__name__}"
    )


def test_skip_browser_tier_omits_optional_fields_when_absent():
    """Bare-minimum payload — only the three required fields. Optional
    location / apply_url are absent from the payload AND from
    state.parsed (no echo of missing fields)."""
    payload = {
        "title": "Engineer",
        "company": "Acme",
        "description": "A long enough description to pass the gate. " * 5,
    }
    state = _ext_state(payload)

    with patch("scrape_graph.tracing._post_transition"), \
         patch("scrape_graph.nodes_scrape.httpx.patch"):
        next_node = _run(SkipBrowserTier(), state)

    assert state.parsed["title"] == "Engineer"
    assert state.parsed["company_name"] == "Acme"
    assert "location" not in state.parsed
    assert "link" not in state.parsed
    # apply_url absent → ResolveApplyUrl route.
    assert isinstance(next_node, ResolveApplyUrl)


# ---------------------------------------------------------------------------
# 4. PersistJobPost — fast-path terminal
# ---------------------------------------------------------------------------


def test_persist_job_post_terminates_at_end_on_fast_path():
    """On the extension-direct fast path, PersistJobPost terminates the
    graph at End — ReviewCompleteness / UpdateProfile / ResolveApplyUrl
    do NOT enter. They're either browser-tier (selector probation,
    apply-button resolution) or already-handled upstream."""
    state = _ext_state(_full_payload(apply_url="https://acme.com/apply"))
    state.parsed = {
        "title": "Engineer", "company_name": "Acme",
        "description": "x" * 200, "link": "https://acme.com/apply",
    }

    with patch("scrape_graph.tracing._post_transition"), \
         patch("scrape_graph.nodes_extract.httpx.post",
               return_value=_persist_response(job_post_id=777)), \
         patch("scrape_graph.nodes_scrape.httpx.patch"):
        result = _run(PersistJobPost(), state)

    assert isinstance(result, End), (
        f"fast-path PersistJobPost must terminate at End, got {type(result).__name__}"
    )
    assert state.outcome == "success"
    assert state.job_post_id == 777
    assert result.data.get("fast_path") is True


def test_persist_job_post_browser_path_routes_to_review_completeness():
    """Browser-mode PersistJobPost keeps the existing routing —
    ReviewCompleteness → UpdateProfile → ResolveApplyUrl → End."""
    from scrape_graph.nodes_extract import ReviewCompleteness

    state = ScrapeGraphState(
        scrape_id=88, submitted_url="https://example.com/job",
        source_mode="browser",
    )
    state.parsed = {"title": "Engineer", "company_name": "Acme"}
    state.job_content = "x" * 500

    with patch("scrape_graph.tracing._post_transition"), \
         patch("scrape_graph.nodes_extract.httpx.post",
               return_value=_persist_response()):
        next_node = _run(PersistJobPost(), state)

    assert isinstance(next_node, ReviewCompleteness), (
        f"browser path must route to ReviewCompleteness, got {type(next_node).__name__}"
    )


def test_persist_job_post_fast_path_records_duplicate_outcome():
    """When the api reports outcome='duplicate' (canonical_link or
    fingerprint match), the fast path End payload reflects it. The
    dedupe-first invariant from api-side parse_scrape is preserved on
    the fast write path."""
    state = _ext_state(_full_payload(apply_url="https://acme.com/apply"))
    state.parsed = {
        "title": "Engineer", "company_name": "Acme",
        "description": "x" * 200,
    }

    with patch("scrape_graph.tracing._post_transition"), \
         patch("scrape_graph.nodes_extract.httpx.post",
               return_value=_persist_response(job_post_id=99, outcome="duplicate")), \
         patch("scrape_graph.nodes_scrape.httpx.patch"):
        result = _run(PersistJobPost(), state)

    assert isinstance(result, End)
    assert state.was_duplicate is True
    assert state.outcome == "duplicate"
    assert result.data["outcome"] == "duplicate"


# ---------------------------------------------------------------------------
# 5. ResolveApplyUrl — fast-path routing
# ---------------------------------------------------------------------------


def test_resolve_apply_url_routes_to_persist_job_post_on_fast_path():
    """On the extension-direct fast path, ResolveApplyUrl no-ops (no
    browser page) and routes to PersistJobPost. On the browser path it
    terminates the graph at End. The branch is on source_mode, not on
    browser_page presence — so a future browserless scrape that's not
    extension-direct still terminates as before."""
    state = _ext_state(_full_payload(apply_url=None))
    # _browser_page absent → resolver returns no_page status
    # (resolve_apply_url checks page is None and bails).

    with patch("scrape_graph.tracing._post_transition"), \
         patch("scrape_graph.nodes_extract.httpx.patch"):
        next_node = _run(ResolveApplyUrl(), state)

    assert isinstance(next_node, PersistJobPost), (
        f"fast-path ResolveApplyUrl must route to PersistJobPost, got {type(next_node).__name__}"
    )


def test_resolve_apply_url_browser_path_terminates_at_end():
    """Browser path with no page (paste/chat ingest) still terminates
    at End — resolve_apply_url returns no_page, ResolveApplyUrl PATCHes
    the scrape and ends the graph."""
    state = ScrapeGraphState(
        scrape_id=33, submitted_url="https://example.com/job",
        source_mode="browser",
    )

    with patch("scrape_graph.tracing._post_transition"), \
         patch("scrape_graph.nodes_extract.httpx.patch"), \
         patch("scrape_graph.nodes_scrape.httpx.patch"):
        result = _run(ResolveApplyUrl(), state)

    assert isinstance(result, End), (
        f"browser path with no page must End, got {type(result).__name__}"
    )


# ---------------------------------------------------------------------------
# 6. End-to-end through the full graph — assert which nodes ran
# ---------------------------------------------------------------------------


def _record_graph_run(state: ScrapeGraphState) -> list[str]:
    """Run the full scrape-graph against a state and return the ordered
    list of node names that recorded a transition. node_trace is the
    canonical source — every node calls trace_node() at the end of
    run() (the only way to advance), so the trace is the authoritative
    list of which nodes executed."""
    from scrape_graph.runner import run_scrape_graph

    with patch("scrape_graph.tracing._post_transition"), \
         patch("scrape_graph.nodes_extract.httpx.post",
               return_value=_persist_response(job_post_id=42)), \
         patch("scrape_graph.nodes_scrape.httpx.patch"), \
         patch("scrape_graph.nodes_extract.httpx.patch"):
        asyncio.run(run_scrape_graph(state, browser_page=None, has_browser=True))
    return [entry.node for entry in state.node_trace]


def test_full_graph_extension_direct_with_apply_url_only_runs_fast_path_nodes():
    """End-to-end: extension-direct with apply_url present should run
    EXACTLY StartScrape → SkipBrowserTier → PersistJobPost. No
    browser-tier node enters; ResolveApplyUrl is bypassed because the
    apply_url is already known."""
    state = _ext_state(_full_payload(apply_url="https://acme.com/apply/abc"))

    nodes_run = _record_graph_run(state)

    # Required nodes
    assert "StartScrape" in nodes_run
    assert "SkipBrowserTier" in nodes_run
    assert "PersistJobPost" in nodes_run
    # Forbidden nodes — browser tier
    for forbidden in (
        "LoadProfile", "Navigate", "DetectObstacle", "ResolveFinalUrl",
        "CheckLinkDedup", "WaitReadySelector", "SettleWait",
        "ScrollToLoad", "ExpandTruncations", "Capture",
        "DetectClosedState", "PersistScrape",
        "StartExtract", "Tier0CSS", "Tier1Mini", "Tier2Haiku",
        "Tier3Sonnet", "EvaluateExtraction", "ValidateExtraction",
        "ReviewCompleteness", "UpdateProfile",
    ):
        assert forbidden not in nodes_run, (
            f"{forbidden} must NOT enter on the fast path; got {nodes_run}"
        )
    # apply_url known → ResolveApplyUrl is skipped too
    assert "ResolveApplyUrl" not in nodes_run, (
        f"ResolveApplyUrl must be skipped when payload carries apply_url; "
        f"got {nodes_run}"
    )


def test_full_graph_extension_direct_without_apply_url_routes_through_resolve_apply():
    """End-to-end: extension-direct with apply_url null runs
    StartScrape → SkipBrowserTier → ResolveApplyUrl → PersistJobPost.
    ResolveApplyUrl enters specifically because the apply_url was null
    and the plan calls for a resolver pass (no-op without a browser
    page, but still part of the recorded trace)."""
    state = _ext_state(_full_payload(apply_url=None))

    nodes_run = _record_graph_run(state)

    assert "StartScrape" in nodes_run
    assert "SkipBrowserTier" in nodes_run
    assert "ResolveApplyUrl" in nodes_run
    assert "PersistJobPost" in nodes_run
    # Browser-tier nodes still forbidden
    for forbidden in (
        "LoadProfile", "Navigate", "DetectObstacle", "Capture",
        "Tier1Mini", "ValidateExtraction", "ReviewCompleteness",
        "UpdateProfile",
    ):
        assert forbidden not in nodes_run, (
            f"{forbidden} must NOT enter on the fast path; got {nodes_run}"
        )


def test_full_graph_extension_direct_with_missing_payload_bails_to_extract_fail():
    """Defensive end-to-end: extension-direct + missing payload bails
    to ExtractFail; the browser tier is NEVER entered. ExtractFail
    produces a debug artifact + closes the scrape failed."""
    state = _ext_state(payload=None)

    with patch("scrape_graph.tracing._post_transition"), \
         patch("scrape_graph.nodes_scrape.httpx.patch"), \
         patch("scrape_graph.nodes_extract.httpx.patch"), \
         patch("scrape_graph._artifacts.capture_debug_artifact",
               return_value={}):
        from scrape_graph.runner import run_scrape_graph
        asyncio.run(run_scrape_graph(state, browser_page=None, has_browser=True))

    nodes_run = [entry.node for entry in state.node_trace]
    assert "StartScrape" in nodes_run
    assert "ExtractFail" in nodes_run
    # No browser tier ran — we did NOT silently degrade
    for forbidden in (
        "LoadProfile", "Navigate", "Capture", "PersistJobPost",
        "SkipBrowserTier",
    ):
        assert forbidden not in nodes_run, (
            f"{forbidden} must NOT enter on the defensive-bail path; "
            f"got {nodes_run}"
        )
    assert state.outcome == "failure"


def test_full_graph_browser_mode_still_runs_browser_tier():
    """The fast path is opt-in via source_mode. A scrape entered
    without source_mode (or with source_mode='browser') still runs the
    legacy browser-tier chain. SkipBrowserTier never enters."""
    # No browser page → runner falls back to extract-only graph (the
    # paste/chat ingest path). We just want to confirm SkipBrowserTier
    # is NOT entered when source_mode='browser', which proves the
    # source_mode gate doesn't accidentally fire on legacy callers.
    from scrape_graph.state import TierAttempt

    state = ScrapeGraphState(
        scrape_id=11,
        submitted_url="https://example.com/jobs/raw",
        source="paste",
        # source_mode left at default 'browser'
    )
    state.job_content = "x" * 500  # paste path needs content to extract

    # Mock the LLM-extract chain so the tier nodes don't make real api
    # calls. Match the real function's side-effect (TierAttempt append)
    # so EvaluateExtraction's escalation ladder (tier0→tier1→tier2→
    # ExtractFail) advances; otherwise it would infinite-loop on the
    # same tier label.
    def fake_call_llm_extract(state, tier_label, model_spec):
        state.tier_attempts.append(
            TierAttempt(tier=tier_label, model=model_spec,
                        produced_output=False, error="mocked")
        )
        return None

    with patch("scrape_graph.tracing._post_transition"), \
         patch("scrape_graph.nodes_extract._call_llm_extract",
               side_effect=fake_call_llm_extract), \
         patch("scrape_graph.nodes_extract.httpx.post",
               return_value=_persist_response()), \
         patch("scrape_graph.nodes_extract.httpx.patch"), \
         patch("scrape_graph.nodes_scrape.httpx.patch"), \
         patch("scrape_graph._artifacts.capture_debug_artifact",
               return_value={}):
        from scrape_graph.runner import run_scrape_graph
        # Browser-mode + no page → extract-only graph (StartExtract entry).
        asyncio.run(run_scrape_graph(state, browser_page=None, has_browser=False))

    nodes_run = [entry.node for entry in state.node_trace]
    assert "SkipBrowserTier" not in nodes_run, (
        f"SkipBrowserTier must NOT enter on a browser-mode scrape; "
        f"got {nodes_run}"
    )
    # The extract-only graph entered at StartExtract.
    assert "StartExtract" in nodes_run


# ---------------------------------------------------------------------------
# 7. Dedupe-first invariant — fast path still hits /persist-extraction/
# ---------------------------------------------------------------------------


def test_fast_path_calls_persist_extraction_endpoint():
    """The plan's non-negotiable: dedupe-first. PersistJobPost on the
    fast path MUST POST to /scrapes/:id/persist-extraction/ so the
    api's JobPostExtractor.parse_scrape (which calls find_duplicate)
    owns canonical_link / fingerprint / sticky-closed handling. The
    fast path bypasses the LLM tiers, NOT the api-side dedupe."""
    state = _ext_state(_full_payload(apply_url="https://acme.com/apply"))
    state.parsed = {
        "title": "Engineer", "company_name": "Acme",
        "description": "x" * 200, "link": "https://acme.com/apply",
    }

    with patch("scrape_graph.tracing._post_transition"), \
         patch("scrape_graph.nodes_extract.httpx.post",
               return_value=_persist_response()) as mock_post, \
         patch("scrape_graph.nodes_scrape.httpx.patch"):
        _run(PersistJobPost(), state)

    # The POST URL must be /persist-extraction/ — the dedupe entrypoint.
    assert mock_post.called, "PersistJobPost must POST to the api"
    call_url = mock_post.call_args[0][0]
    assert call_url.endswith(
        f"/api/v1/scrapes/{state.scrape_id}/persist-extraction/"
    ), f"fast path must hit persist-extraction; got {call_url}"
    # And the payload it sends is the mapped ParsedJobData (not the
    # raw captured_payload) — sanity-check the body shape.
    body = mock_post.call_args[1]["json"]
    assert body == {"attributes": state.parsed}


# ---------------------------------------------------------------------------
# 8. Trace recording — fast-path transitions land on Scrape.node_trace
# ---------------------------------------------------------------------------


def test_fast_path_records_source_mode_in_trace_payload():
    """The StartScrape→SkipBrowserTier transition records source_mode +
    fast_path in the trace payload so the per-scrape graph-trace
    introspection UI shows operators which branch fired."""
    state = _ext_state(_full_payload())

    with patch("scrape_graph.tracing._post_transition"):
        _run(StartScrape(), state)

    branch_entries = [
        e for e in state.node_trace
        if e.node == "StartScrape" and e.routed_to == "SkipBrowserTier"
    ]
    assert branch_entries, "StartScrape→SkipBrowserTier must trace"
    payload = branch_entries[0].payload
    assert payload.get("source_mode") == "extension-direct"
    assert payload.get("fast_path") is True
