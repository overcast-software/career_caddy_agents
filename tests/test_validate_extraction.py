"""Tests for the ValidateExtraction node.

Covers the content-quality gate that guards PersistJobPost from
accepting hallucinated extractions off of loading shells and too-thin
source text. Scrape 172 (Salesforce Lightning bootstrap) is the
motivating case.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from scrape_graph.nodes_extract import (
    ExtractFail,
    PersistJobPost,
    ValidateExtraction,
)
from scrape_graph.state import ScrapeGraphState


def _run(node, state: ScrapeGraphState):
    ctx = SimpleNamespace(state=state)
    return asyncio.run(node.run(ctx))


@pytest.fixture
def good_state() -> ScrapeGraphState:
    state = ScrapeGraphState(scrape_id=1, submitted_url="https://x.com/job/1")
    state.parsed = {"title": "Backend Engineer", "company_name": "Acme"}
    state.job_content = (
        "Backend Engineer at Acme. We are hiring a senior backend "
        "engineer to join our distributed systems team. You will work "
        "on our core platform handling millions of requests per day. "
        "Requirements include 5+ years of Python experience, strong "
        "knowledge of distributed systems, and a track record of "
        "shipping production software. Nice to haves include Rust, "
        "Kubernetes, and a sense of humor."
    )
    return state


def test_validate_passes_on_real_job_text(good_state):
    next_node = _run(ValidateExtraction(), good_state)
    assert isinstance(next_node, PersistJobPost)
    assert good_state.evaluation["validate_passed"] is True
    assert good_state.evaluation["validate_reasons"] == []


def test_validate_fails_on_salesforce_loading_shell():
    """Scrape 172 repro — LLM hallucinated title/company off of
    `Loading × Sorry to interrupt CSS Error Refresh ... enable cookies
    in your browser`. The gate must fail even though `parsed` looks
    fine, because the source text is a bootstrap shell.
    """
    state = ScrapeGraphState(scrape_id=172, submitted_url="https://ziprecruiter.com/ekm/xyz")
    state.parsed = {"title": "Plausible Sounding Role", "company_name": "Hallucinated Corp"}
    state.job_content = (
        "Loading × Sorry to interrupt CSS Error Refresh "
        "To view this site, enable cookies in your browser. "
        "cookieEnabled Technical Stuff"
    )
    next_node = _run(ValidateExtraction(), state)
    assert isinstance(next_node, ExtractFail)
    assert "loading_shell_fingerprint" in state.evaluation["validate_reasons"]
    assert state.failure_reason.startswith("validate_failed:")


def test_validate_fails_on_thin_source():
    state = ScrapeGraphState(scrape_id=2, submitted_url="https://x.com/job/2")
    state.parsed = {"title": "Role", "company_name": "Co"}
    state.job_content = "Too few words here."
    next_node = _run(ValidateExtraction(), state)
    assert isinstance(next_node, ExtractFail)
    assert "source_too_short" in state.evaluation["validate_reasons"]


def test_validate_fails_on_ui_chrome_only_description():
    """LinkedIn lazy-hydration repro — when the 'About the job' card
    never renders, Tier1 sometimes returns a description that's just
    the visible chrome (pills, salary banner, Apply/Save, the 'Use AI
    to assess how you fit' CTA). EvaluateExtraction passes it because
    title + company are in the page header; ValidateExtraction must
    reject it before it poisons the JobPost.
    """
    state = ScrapeGraphState(scrape_id=1650, submitted_url="https://linkedin.com/jobs/view/4407097463")
    state.parsed = {
        "title": "Staff Engineer",
        "company_name": "Some Co",
        "description": (
            "$215K/yr - $250K/yr\nRemote\nFull-time\nApply\nSave\n"
            "Use AI to assess how you fit"
        ),
    }
    # Source content is long enough — the gate has to fire on the
    # parsed description, not the source word count.
    state.job_content = " ".join(["word"] * 80)
    next_node = _run(ValidateExtraction(), state)
    assert isinstance(next_node, ExtractFail)
    assert "ui_chrome_only" in state.evaluation["validate_reasons"]


def test_validate_passes_on_real_description_short_but_meaningful():
    """A short description that contains real-job vocabulary
    ('responsibilities', 'requirements', etc.) must NOT trip the
    chrome guard, even if it's under the size threshold.
    """
    state = ScrapeGraphState(scrape_id=1, submitted_url="https://x.com/job/1")
    state.parsed = {
        "title": "Backend Engineer",
        "company_name": "Acme",
        "description": (
            "Responsibilities: ship the platform. "
            "Requirements: 5+ years Python."
        ),
    }
    state.job_content = " ".join(["word"] * 80)
    next_node = _run(ValidateExtraction(), state)
    assert isinstance(next_node, PersistJobPost)


# ── CC-118: error / 404 / dead-posting pages ────────────────────────────

# A 404 rendered by a real ATS still ships the host's nav + footer, so it
# clears _SOURCE_MIN_WORDS comfortably. That is exactly why the word-count
# floor never caught these.
_ERROR_PAGE_SOURCE = (
    "Visa Careers Search Jobs Sign In Create Account Help "
    "Job Posting Not Found The page you are looking for doesn't exist. "
    "Return to the job search page to browse current openings. "
    "About Visa Investor Relations Newsroom Careers Contact Us "
    "Privacy Notice Terms of Use Cookie Preferences Accessibility "
    "Copyright 2026 Visa Inc. All Rights Reserved Follow us on social media"
)


def test_validate_rejects_404_error_page():
    """Workday /apply deep-link repro (JobPost 2w4MaSFCra). The capture
    clears the 40-word floor on nav chrome alone and matches no loading-
    shell phrase, so before CC-118 a `title="Not Found"` extraction went
    straight to PersistJobPost — minting a junk post AND banking a false
    success against the host's success_rate.
    """
    state = ScrapeGraphState(
        scrape_id="FeTWuow8AR",
        submitted_url=(
            "https://visa.wd5.myworkdayjobs.com/en-US/Visa/job/"
            "Sr-Consultant-SW-Engineer_REF080924W/apply?isWholefeeds=1"
        ),
    )
    state.parsed = {
        "title": "Not Found",
        "company_name": "Visa",
        "description": "The page you are looking for doesn't exist.",
    }
    state.job_content = _ERROR_PAGE_SOURCE
    next_node = _run(ValidateExtraction(), state)
    assert isinstance(next_node, ExtractFail)
    assert "error_page" in state.evaluation["validate_reasons"]
    assert state.evaluation["validate_passed"] is False
    assert state.failure_reason.startswith("validate_failed:")


def test_validate_rejects_job_posting_not_found_title():
    """Second junk post from the same batch (YGYTFc7iLf)."""
    state = ScrapeGraphState(scrape_id="YGYTFc7iLf", submitted_url="https://x.com/job/9")
    state.parsed = {
        "title": "Job Posting Not Found",
        "company_name": "Visa",
        "description": "The page you are looking for doesn't exist.",
    }
    state.job_content = _ERROR_PAGE_SOURCE
    next_node = _run(ValidateExtraction(), state)
    assert isinstance(next_node, ExtractFail)
    assert "error_page" in state.evaluation["validate_reasons"]


def test_validate_rejects_error_title_with_site_name_suffix():
    """Hosts append their own name to the <title>. A sentinel hiding
    behind ' | Acme Careers' must still be caught.
    """
    state = ScrapeGraphState(scrape_id="a1", submitted_url="https://acme.com/jobs/1")
    state.parsed = {
        "title": "404 | Acme Careers",
        "company_name": "Acme",
        "description": "Sorry, we can't find that.",
    }
    state.job_content = _ERROR_PAGE_SOURCE
    next_node = _run(ValidateExtraction(), state)
    assert isinstance(next_node, ExtractFail)
    assert "error_page" in state.evaluation["validate_reasons"]


def test_validate_rejects_dead_posting_body_behind_an_honest_stub():
    """The body-sentinel arm. When a tier invents prose off a dead
    posting, EvaluateExtraction's grounding check demotes it to an honest
    stub — so the only surviving evidence is in the captured source. A
    stub is worth persisting when the posting is real; on a dead page it
    is a junk row with a plausible title, so refuse.
    """
    state = ScrapeGraphState(scrape_id="b2", submitted_url="https://boards.x.com/job/2")
    state.parsed = {
        "title": "Senior Platform Engineer",
        "company_name": "Acme",
        "description": (
            "[DESCRIPTION NOT CAPTURED — the scrape reached this posting "
            "but could not read its description (the page returned no "
            "description body). Re-scrape the link, or send the page from "
            "the browser extension while it is open.]"
        ),
    }
    state.job_content = (
        "Acme Careers Search Openings Departments Locations Sign In "
        "Create an account to track your applications. "
        "This job posting no longer exists. It may have been filled or "
        "withdrawn. Browse our current openings instead. "
        "About Acme Newsroom Investors Diversity Benefits Life at Acme "
        "Privacy Terms Cookie Preferences Accessibility Statement "
        "Copyright 2026 Acme Inc. All Rights Reserved"
    )
    next_node = _run(ValidateExtraction(), state)
    assert isinstance(next_node, ExtractFail)
    assert "error_page" in state.evaluation["validate_reasons"]


@pytest.mark.parametrize(
    "title",
    [
        "Error Budget Engineer",
        "Site Reliability Engineer",
        "Senior Engineer - Remote",
        "Lost and Found Operations Coordinator",
        "Oops Media — Content Designer",
    ],
)
def test_validate_passes_titles_that_merely_contain_error_words(good_state, title):
    """False-positive guard. The title check is EXACT after
    normalization; a substring test would eat every one of these.
    """
    good_state.parsed = {**good_state.parsed, "title": title}
    next_node = _run(ValidateExtraction(), good_state)
    assert isinstance(next_node, PersistJobPost)
    assert good_state.evaluation["validate_reasons"] == []


def test_validate_passes_real_posting_that_quotes_a_dead_posting_phrase():
    """A real body long enough to be prose is not an error page even when
    it contains one of the body sentinels verbatim.
    """
    state = ScrapeGraphState(scrape_id="c3", submitted_url="https://x.com/job/3")
    state.parsed = {
        "title": "Backend Engineer",
        "company_name": "Acme",
        "description": (
            "Responsibilities: keep our job board honest. You will build "
            "the sweeper that marks a listing dead the moment the position "
            "is no longer available, and the notifier that tells applicants "
            "before they waste a submission. Requirements: 5+ years of "
            "Python, comfort with distributed systems, and the patience to "
            "chase down every last stale record in a very large table. "
            "Experience with Django and Postgres is a strong plus."
        ),
    }
    state.job_content = " ".join(["word"] * 80)
    next_node = _run(ValidateExtraction(), state)
    assert isinstance(next_node, PersistJobPost)


def test_validate_passes_real_workday_posting(good_state):
    """The non-404 Workday page the guard must never touch."""
    good_state.parsed = {
        "title": "Sr. Consultant, SW Engineer",
        "company_name": "Visa",
        "description": (
            "Requirements: 8+ years of software engineering experience. "
            "You will design and build payment infrastructure at scale."
        ),
    }
    next_node = _run(ValidateExtraction(), good_state)
    assert isinstance(next_node, PersistJobPost)


def test_validate_preserves_prior_evaluation(good_state):
    good_state.evaluation = {"passed": True, "reasons": []}
    _run(ValidateExtraction(), good_state)
    # Merged, not replaced.
    assert good_state.evaluation["passed"] is True
    assert good_state.evaluation["validate_passed"] is True
