"""CC-226 — fast-fail on search-landing / interstitial pages.

Before this guard, a URL that landed somewhere the profile's
ready_selector could never match still walked the whole graph:
DetectClosedState's LLM leg, a full-DOM PATCH, and the extraction ladder
whose Tier1 + Tier2 HTTP calls carry a 120s timeout EACH — enough to
reach the runner's 240s `GRAPH_RUN_TIMEOUT_S` cap on their own. ~170 such
timeouts fired over 2026-07-20..29 across linkedin, adzuna, jobright and
ziprecruiter.

The tests split the way the code does: the detector is pure text/URL
logic and is tested directly; the node tests only prove the routing and
the terminal's contract.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from scrape_graph.landing_page_detector import detect_landing_page
from scrape_graph.nodes_scrape import (
    Capture,
    DetectClosedState,
    LandingPageFail,
)
from scrape_graph.state import ScrapeGraphState


def _run(node, state: ScrapeGraphState):
    ctx = SimpleNamespace(state=state)
    return asyncio.run(node.run(ctx))


# ── Fixtures: what the captured visible text actually looks like ────────

# adzuna, scrape erQeLhRkfu — a /land/ad/ tracker that resolved onto the
# generic search page. `h1[data-testid*='title']` never appeared and the
# graph waited out the cap having captured nothing usable.
ADZUNA_SEARCH_LANDING = """
Adzuna
What?
Where?
Search
1,247,883 jobs found
Sort by relevance
Date posted
Refine your search
Create job alert
Browse jobs by category
Salary
Contract type
Next page
"""

# The shape that must NOT be refused: a real posting rendered inside a
# results layout, so the listing furniture is present alongside the body.
# This is the LinkedIn detail page, and it is why a ready_selector miss
# alone can never be the trigger.
LINKEDIN_DETAIL_WITH_RESULTS_RAIL = """
Jobs
Sort by: Most relevant
Date posted
120 results
Staff Software Engineer
Acme Corp · San Francisco, CA (Hybrid)
About the job
Responsibilities: own the ingestion pipeline end to end, from the
scheduler through to the warehouse.
Requirements: 8+ years of experience building distributed systems.
Acme Corp is an equal opportunity employer.
"""

# A minimal posting on a host with no listing chrome at all — the common
# case, and the regression the guard must never touch.
PLAIN_DETAIL = """
Backend Engineer
Acme Corp
About the role
You will build and operate our payments platform.
Requirements: 5+ years of Python.
"""


# ── Detector ───────────────────────────────────────────────────────────

def test_detects_adzuna_search_landing():
    verdict = detect_landing_page(
        ADZUNA_SEARCH_LANDING,
        url="https://www.adzuna.com/search?q=engineer&loc=98101",
        ready_selector_configured=True,
        ready_selector_matched=False,
    )
    assert verdict is not None
    assert verdict["verdict"] == "landing_page"
    # All four signals are present on this page; the trace names them so a
    # false positive is diagnosable without re-running the scrape.
    assert set(verdict["landing_signals"]) == {
        "result_count", "listing_controls", "search_form", "search_url",
    }
    assert verdict["detail_signals"] == []


def test_real_posting_inside_a_results_layout_is_not_a_landing_page():
    """Two landing signals fire (result count + listing controls) and the
    ready_selector still missed — a stale profile selector, say. The
    job-detail vocabulary must veto the refusal anyway.
    """
    assert detect_landing_page(
        LINKEDIN_DETAIL_WITH_RESULTS_RAIL,
        url="https://www.linkedin.com/jobs/view/4407097463",
        ready_selector_configured=True,
        ready_selector_matched=False,
    ) is None


def test_a_matched_ready_selector_vetoes_the_verdict():
    """The detail anchor appeared at some point — in WaitReadySelector or
    late in ScrollToLoad. Whatever furniture surrounds it, this is a
    detail page.
    """
    assert detect_landing_page(
        ADZUNA_SEARCH_LANDING,
        url="https://www.adzuna.com/search?q=engineer",
        ready_selector_configured=True,
        ready_selector_matched=True,
    ) is None


def test_unprofiled_host_never_fast_fails():
    """With no configured ready_selector there is no expectation to have
    missed, so the guard has no standing to refuse.
    """
    assert detect_landing_page(
        ADZUNA_SEARCH_LANDING,
        url="https://www.adzuna.com/search?q=engineer",
        ready_selector_configured=False,
        ready_selector_matched=False,
    ) is None


def test_one_signal_is_not_enough():
    """A lone result-count phrase is exactly the innocent case — a JD
    saying 'we have 3 openings on this team'. Corroboration is required.
    """
    text = "Senior Engineer\nWe have 3 openings on this team right now."
    assert detect_landing_page(
        text,
        url="https://boards.example.com/jobs/1234",
        ready_selector_configured=True,
        ready_selector_matched=False,
    ) is None


@pytest.mark.parametrize(
    "url",
    [
        "https://jobs.example.com/search?q=python",
        "https://jobs.example.com/jobs/search",
        "https://jobs.example.com/browse",
        "https://jobs.example.com/results?keywords=engineer",
    ],
)
def test_search_shaped_urls_contribute_a_signal(url):
    """The URL alone is one signal; paired with listing controls it makes
    the pair. Interstitials that bounce to a generic search URL are the
    case this covers.
    """
    text = "Sort by relevance\nRefine your search\nNothing to see here."
    verdict = detect_landing_page(
        text, url=url,
        ready_selector_configured=True,
        ready_selector_matched=False,
    )
    assert verdict is not None
    assert "search_url" in verdict["landing_signals"]
    assert "listing_controls" in verdict["landing_signals"]


def test_detail_url_contributes_no_signal():
    text = "Sort by relevance\nRefine your search\nNothing to see here."
    verdict = detect_landing_page(
        text, url="https://jobs.example.com/careers/12345/backend-engineer",
        ready_selector_configured=True,
        ready_selector_matched=False,
    )
    assert verdict is None


# ── Node routing ───────────────────────────────────────────────────────

@pytest.fixture
def quiet_capture(monkeypatch):
    """Silence Capture's three network/DOM side effects. None of them is
    under test here; all three would otherwise reach for httpx.
    """
    from scrape_graph import nodes_scrape

    async def _noop_canonical(page, state):
        return {}

    async def _noop(page, state):
        return None

    monkeypatch.setattr(nodes_scrape, "_adopt_declared_canonical", _noop_canonical)
    monkeypatch.setattr(nodes_scrape, "_screenshot_and_upload", _noop)
    monkeypatch.setattr(nodes_scrape, "_discover_selectors", _noop)


class _FakePage:
    def __init__(self, text: str) -> None:
        self._text = text

    async def inner_text(self, _selector: str) -> str:
        return self._text

    async def content(self) -> str:
        return f"<html><body>{self._text}</body></html>"


def _capture_state(text: str, url: str) -> ScrapeGraphState:
    state = ScrapeGraphState(scrape_id="erQeLhRkfu", submitted_url=url)
    state.profile = {"ready_selector": ["h1[data-testid*='title']"]}
    state.final_url = url
    state._browser_page = _FakePage(text)  # type: ignore[attr-defined]
    return state


def test_capture_routes_a_landing_page_to_the_fast_fail(quiet_capture):
    """The whole point: the graph leaves at Capture instead of paying for
    DetectClosedState's LLM leg, the DOM PATCH and the tier ladder.
    """
    state = _capture_state(
        ADZUNA_SEARCH_LANDING,
        "https://www.adzuna.com/search?q=engineer&loc=98101",
    )
    next_node = _run(Capture(), state)
    assert isinstance(next_node, LandingPageFail)
    assert not isinstance(next_node, DetectClosedState)
    assert state.failure_reason == "landing_page_not_detail"
    # The verdict is on the trace so these are diagnosable in bulk.
    last = state.node_trace[-1]
    assert last.routed_to == "LandingPageFail"
    assert last.payload["landing_page"]["verdict"] == "landing_page"


def test_capture_still_routes_a_real_posting_onward(quiet_capture):
    state = _capture_state(
        PLAIN_DETAIL, "https://boards.example.com/jobs/1234",
    )
    next_node = _run(Capture(), state)
    assert isinstance(next_node, DetectClosedState)
    assert state.failure_reason is None
    assert state.job_content == PLAIN_DETAIL


def test_capture_does_not_fast_fail_when_the_selector_matched(quiet_capture):
    """A late ScrollToLoad hit is recorded on state; Capture must honour
    it even on a page that otherwise reads as a listing.
    """
    state = _capture_state(
        ADZUNA_SEARCH_LANDING,
        "https://www.adzuna.com/search?q=engineer",
    )
    state.matched_ready_selector = "h1[data-testid*='title']"
    next_node = _run(Capture(), state)
    assert isinstance(next_node, DetectClosedState)


# ── Terminal contract ──────────────────────────────────────────────────

def test_landing_page_fail_terminal(monkeypatch):
    """Distinct status note, distinct failure_reason, and a debug artifact
    — the note is what makes these queryable apart from the runner's
    generic 'graph run exceeded 240s cap'.
    """
    from pydantic_graph import End

    from scrape_graph import _artifacts, nodes_scrape

    patched: dict = {}

    def _fake_patch(scrape_id, status, note=None):
        patched.update(scrape_id=scrape_id, status=status, note=note)

    async def _fake_artifact(page, state, *, reason):
        patched["artifact_reason"] = reason
        return {"screenshot_uploaded": True, "dom_saved": True, "reason": reason}

    monkeypatch.setattr(nodes_scrape, "_patch_scrape_status", _fake_patch)
    monkeypatch.setattr(_artifacts, "capture_debug_artifact", _fake_artifact)

    state = ScrapeGraphState(
        scrape_id="erQeLhRkfu", submitted_url="https://www.adzuna.com/search?q=x",
    )
    state.failure_reason = "landing_page_not_detail"
    result = _run(LandingPageFail(), state)

    assert isinstance(result, End)
    assert result.data["outcome"] == "failure"
    assert result.data["failure_reason"] == "landing_page_not_detail"
    assert state.outcome == "failure"
    assert patched["status"] == "failed"
    assert patched["note"] == "landing_page_not_detail"
    assert patched["artifact_reason"] == "landing_page"


def test_landing_page_fail_defaults_its_own_reason(monkeypatch):
    """Reached without Capture having set one (defensive), the terminal
    still names itself rather than inheriting a blank."""
    from scrape_graph import _artifacts, nodes_scrape

    async def _fake_artifact(page, state, *, reason):
        return {}

    monkeypatch.setattr(nodes_scrape, "_patch_scrape_status", lambda *a, **k: None)
    monkeypatch.setattr(_artifacts, "capture_debug_artifact", _fake_artifact)

    state = ScrapeGraphState(scrape_id="x1", submitted_url="https://x.com/search?q=y")
    _run(LandingPageFail(), state)
    assert state.failure_reason == "landing_page_not_detail"
