"""Tests for the DetectClosedState graph node.

The node orchestrates three detection paths (CSS, phrase, LLM) over
host config in ``ScrapeProfile.css_selectors.closed_state_config``,
always routing to PersistScrape. Verdict + evidence land on
state.detected_posting_status / detected_closed_evidence; downstream
JobPostExtractor reads those as the priority-1 channel.

Critical regression: the JP 1532 / scrape 414 incident (2026-05-13).
A 756-char LinkedIn chrome capture with NO host config existed →
must NOT trigger the LLM (cost guard) → no false flip to closed.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from scrape_graph.nodes_scrape import DetectClosedState, PersistScrape
from scrape_graph.state import ScrapeGraphState


def _run(node, state: ScrapeGraphState):
    ctx = SimpleNamespace(state=state)
    return asyncio.run(node.run(ctx))


def _state(
    *,
    scrape_id: int = 1,
    job_content: str = "",
    css_selectors_blob: dict | None = None,
    page=None,
) -> ScrapeGraphState:
    state = ScrapeGraphState(scrape_id=scrape_id, submitted_url="https://example.com/jobs/1")
    state.job_content = job_content
    if css_selectors_blob is not None:
        state.profile = {
            "id": 42,
            "hostname": "example.com",
            "css_selectors": css_selectors_blob,
        }
    if page is not None:
        state._browser_page = page  # type: ignore[attr-defined]
    return state


def _patch_httpx():
    """Mock the scrape-profile PATCH (promotion side-effect) so tests
    don't need a live api."""
    return patch("scrape_graph.nodes_scrape.httpx")


# ---------------------------------------------------------------------------
# Always routes to PersistScrape
# ---------------------------------------------------------------------------

def test_always_routes_to_persist_scrape_no_config():
    """No host config + no live page → still routes to PersistScrape."""
    state = _state(job_content="hello")
    with _patch_httpx():
        next_node = _run(DetectClosedState(), state)
    assert isinstance(next_node, PersistScrape)
    assert state.detected_posting_status is None  # no signal


# ---------------------------------------------------------------------------
# CSS path
# ---------------------------------------------------------------------------

def test_css_hit_sets_verdict_closed_and_method_css():
    """Live CSS selector hit → verdict=closed, method=css, evidence is
    the matched element's inner_text snippet."""
    page = MagicMock()
    handle = MagicMock()
    handle.inner_text = AsyncMock(return_value="No longer accepting applications")

    async def query_selector(sel):
        return handle if sel == ".job-closed-banner" else None

    page.query_selector = AsyncMock(side_effect=query_selector)

    blob = {"closed_state_config": {"css_selectors": [".job-closed-banner"]}}
    state = _state(job_content="(any)", css_selectors_blob=blob, page=page)
    with _patch_httpx():
        _run(DetectClosedState(), state)
    assert state.detected_posting_status == "closed"
    assert state.closed_detection_method == "css"
    assert "no longer accepting" in (state.detected_closed_evidence or "").lower()


# ---------------------------------------------------------------------------
# Phrase path
# ---------------------------------------------------------------------------

def test_phrase_hit_when_no_css_configured():
    blob = {"closed_state_config": {"text_phrases": ["no longer accepting applications"]}}
    state = _state(
        job_content="Senior Eng. We are no longer accepting applications. Thanks.",
        css_selectors_blob=blob,
    )
    with _patch_httpx():
        _run(DetectClosedState(), state)
    assert state.detected_posting_status == "closed"
    assert state.closed_detection_method == "phrase"


def test_css_runs_alongside_phrase_logs_both_in_trace():
    """Per the design: when both CSS and phrase configured, both run
    so the trace has telemetry on each. Verdict tie-break: CSS wins
    (deterministic on DOM, more specific)."""
    page = MagicMock()
    handle = MagicMock()
    handle.inner_text = AsyncMock(return_value="closed")

    async def query_selector(sel):
        return handle if sel == ".x" else None
    page.query_selector = AsyncMock(side_effect=query_selector)

    blob = {"closed_state_config": {
        "css_selectors": [".x"],
        "text_phrases": ["no longer accepting applications"],
    }}
    state = _state(
        job_content="We are no longer accepting applications.",
        css_selectors_blob=blob,
        page=page,
    )
    with _patch_httpx():
        _run(DetectClosedState(), state)
    # CSS won the verdict
    assert state.closed_detection_method == "css"
    # The trace payload (last node_trace entry) should still record the
    # phrase attempt so a future regret-analysis can ask "would the
    # phrase path also have caught it?"
    last = state.node_trace[-1]
    assert last.payload.get("css", {}).get("ran") is True
    assert last.payload.get("phrase", {}).get("ran") is True


def test_no_signal_when_config_present_but_nothing_matches():
    """Per-host curator decided what to look for; absence is a real
    'open' signal — do NOT fall back to LLM."""
    blob = {"closed_state_config": {"text_phrases": ["never appears"]}}
    state = _state(
        job_content="Active job. Apply now. " * 100,  # plenty of chars
        css_selectors_blob=blob,
    )
    with _patch_httpx():
        _run(DetectClosedState(), state)
    assert state.detected_posting_status is None
    assert state.closed_detection_method == "no_signal"
    last = state.node_trace[-1]
    # LLM block records that it didn't run + why
    assert last.payload.get("llm") == {"ran": False, "reason": "config_present"}


# ---------------------------------------------------------------------------
# LLM path + cost guard (jp 1532 regression)
# ---------------------------------------------------------------------------

def test_llm_skipped_when_capture_below_min_chars_jp_1532_regression():
    """JP 1532 regression. 756 chars of degraded LinkedIn chrome with
    no host config previously caused the LLM-emitted closed_evidence
    path inside the extractor to false-positive. The graph-side cost
    guard prevents Haiku from even being called on captures this thin —
    the verdict stays None so no flip happens.
    """
    blob = {}  # no closed_state_config at all
    thin_chrome = (
        "0 notifications\nSkip to main content\nHome\nMy Network\nJobs\n"
        "GitHub\nSoftware Engineer II, Security\n"
        "United States · Reposted 1 week ago · Over 100 people clicked apply\n"
        "Promoted by hirer · Responses managed off LinkedIn\n"
        "Remote\nFull-time\nApply\nSave\n"
    )
    assert len(thin_chrome) < 1000
    state = _state(job_content=thin_chrome, css_selectors_blob=blob)
    with _patch_httpx():
        # If the LLM were called, this patch would fail — but we don't
        # patch detect_via_llm so a live invocation would either error
        # (no API key) or take real time. Assert it wasn't called by
        # checking the trace.
        _run(DetectClosedState(), state)
    assert state.detected_posting_status is None
    assert state.closed_detection_method == "skipped_thin_capture"
    last = state.node_trace[-1]
    llm = last.payload.get("llm") or {}
    assert llm.get("ran") is False
    assert llm.get("reason") == "below_min_chars"
    assert llm.get("captured_chars") == len(thin_chrome)


def test_llm_called_when_no_config_and_capture_substantive():
    """No host config + substantive capture → Haiku gets invoked.
    Mock the LLM path so the test doesn't make a real call."""
    blob = {}
    text = "We are no longer accepting applications for this position. " * 30
    state = _state(job_content=text, css_selectors_blob=blob)

    async def fake_detect_llm(text_arg, model="anthropic:claude-haiku-4-5"):
        return {
            "model": model,
            "is_closed": True,
            "evidence_quote": "no longer accepting applications for this position",
            "duration_ms": 100,
            "error": None,
        }

    with _patch_httpx() as mock_httpx, patch(
        "scrape_graph.closed_state_detector.detect_via_llm",
        side_effect=fake_detect_llm,
    ):
        # The promotion path PATCHes ScrapeProfile — return a 200.
        mock_httpx.patch = MagicMock(return_value=MagicMock(status_code=200))
        _run(DetectClosedState(), state)

    assert state.detected_posting_status == "closed"
    assert state.closed_detection_method == "llm"
    last = state.node_trace[-1]
    llm = last.payload.get("llm") or {}
    assert llm.get("ran") is True
    assert llm.get("verdict") == "closed"
    assert llm.get("quote_validated") is True


def test_llm_unvalidated_quote_does_not_flip_verdict():
    """LLM emits a quote but it's NOT verbatim in the captured text →
    verdict stays None (anti-hallucination)."""
    blob = {}
    text = "Active posting. " * 100  # no closure phrase
    state = _state(job_content=text, css_selectors_blob=blob)

    async def fake_detect_llm(text_arg, model="anthropic:claude-haiku-4-5"):
        return {
            "model": model,
            "is_closed": True,
            "evidence_quote": "fabricated banner that doesn't exist",
            "duration_ms": 80,
            "error": None,
        }

    with _patch_httpx(), patch(
        "scrape_graph.closed_state_detector.detect_via_llm",
        side_effect=fake_detect_llm,
    ):
        _run(DetectClosedState(), state)
    assert state.detected_posting_status is None
    last = state.node_trace[-1]
    llm = last.payload.get("llm") or {}
    assert llm.get("quote_validated") is False


def test_llm_hit_promotes_phrase_to_scrape_profile():
    """Validated LLM closure → PATCH ScrapeProfile to append the
    learned phrase. Trace records the promotion."""
    blob = {}
    quote = "this position has been filled"
    text = f"Senior Engineer. About the role. Update: {quote}. Thanks."
    state = _state(job_content=text + " padding " * 200, css_selectors_blob=blob)

    async def fake_detect_llm(text_arg, model="anthropic:claude-haiku-4-5"):
        return {"model": model, "is_closed": True, "evidence_quote": quote,
                "duration_ms": 50, "error": None}

    with _patch_httpx() as mock_httpx, patch(
        "scrape_graph.closed_state_detector.detect_via_llm",
        side_effect=fake_detect_llm,
    ):
        mock_httpx.patch = MagicMock(return_value=MagicMock(status_code=200))
        mock_httpx.post = MagicMock(return_value=MagicMock(status_code=201))
        _run(DetectClosedState(), state)

    # PATCH against /scrape-profiles/42/ should have been called once
    patch_calls = mock_httpx.patch.call_args_list
    profile_patches = [c for c in patch_calls if "/scrape-profiles/42/" in c.args[0]]
    assert profile_patches, f"expected ScrapeProfile patch, got: {patch_calls}"
    body = profile_patches[0].kwargs["json"]
    new_blob = body["data"]["attributes"]["css_selectors"]
    learned = new_blob["closed_state_config"]["learned_phrases"]
    assert any(e["phrase"] == quote for e in learned)
    assert all("promoted_at" in e and "from_scrape_id" in e for e in learned)
