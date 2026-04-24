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

from lib.scrape_graph.nodes_extract import (
    ExtractFail,
    PersistJobPost,
    ValidateExtraction,
)
from lib.scrape_graph.state import ScrapeGraphState


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


def test_validate_preserves_prior_evaluation(good_state):
    good_state.evaluation = {"passed": True, "reasons": []}
    _run(ValidateExtraction(), good_state)
    # Merged, not replaced.
    assert good_state.evaluation["passed"] is True
    assert good_state.evaluation["validate_passed"] is True
