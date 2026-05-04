"""Tests for the EvaluateExtraction node.

Covers the partial-render escape hatch landed for LinkedIn SDUI:
when the LLM emits the documented placeholder description ("[DESCRIPTION
NOT CAPTURED — ...]") because the page only rendered the top card,
EvaluateExtraction must let it through with title/company instead of
tripping `thin_description` and escalating to higher tiers.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from scrape_graph.nodes_extract import (
    EvaluateExtraction,
    ExtractFail,
    Tier2Haiku,
    ValidateExtraction,
)
from scrape_graph.state import ScrapeGraphState, TierAttempt


def _run(node, state: ScrapeGraphState):
    ctx = SimpleNamespace(state=state)
    return asyncio.run(node.run(ctx))


def _state_with(parsed: dict, last_tier: str = "tier1") -> ScrapeGraphState:
    state = ScrapeGraphState(scrape_id=1, submitted_url="https://x.com/job/1")
    state.parsed = parsed
    state.tier_attempts.append(
        TierAttempt(tier=last_tier, model="x", produced_output=True)
    )
    return state


_LONG_DESC = " ".join(["word"] * 80)
_PLACEHOLDER = (
    "[DESCRIPTION NOT CAPTURED — LinkedIn page rendered only the top "
    "card; the job description body did not hydrate within the scrape "
    "window. Visit the link directly to read the full posting.]"
)


def test_evaluate_passes_normal_extraction():
    state = _state_with({
        "title": "Backend Engineer",
        "company_name": "Acme",
        "description": _LONG_DESC,
    })
    next_node = _run(EvaluateExtraction(), state)
    assert isinstance(next_node, ValidateExtraction)
    assert state.evaluation["passed"] is True
    assert state.evaluation["partial_render"] is False


def test_evaluate_rejects_thin_description():
    state = _state_with({
        "title": "Backend Engineer",
        "company_name": "Acme",
        "description": "We need a great engineer. Apply now.",
    })
    next_node = _run(EvaluateExtraction(), state)
    # last_tier=tier1 → escalates to Tier2Haiku.
    assert isinstance(next_node, Tier2Haiku)
    assert "thin_description" in state.evaluation["reasons"]
    assert state.evaluation["partial_render"] is False


def test_evaluate_accepts_partial_render_placeholder():
    """LinkedIn SDUI partial-render: the LLM was instructed by the
    profile's extraction_hints to emit the placeholder. The placeholder
    is short — well under _STUB_MIN_WORDS — but is intentional, not a
    thin extraction. Must pass through to ValidateExtraction."""
    state = _state_with({
        "title": "Forward Deployed Engineer",
        "company_name": "Magnitude Consulting",
        "description": _PLACEHOLDER,
    })
    next_node = _run(EvaluateExtraction(), state)
    assert isinstance(next_node, ValidateExtraction)
    assert state.evaluation["passed"] is True
    assert state.evaluation["partial_render"] is True
    assert state.evaluation["reasons"] == []


def test_evaluate_partial_render_still_requires_title_and_company():
    """The placeholder waiver only covers the thin-description gate.
    Title and company must still be present."""
    state = _state_with({
        "title": "",
        "company_name": "Magnitude Consulting",
        "description": _PLACEHOLDER,
    })
    next_node = _run(EvaluateExtraction(), state)
    assert not isinstance(next_node, ValidateExtraction)
    assert "missing_title" in state.evaluation["reasons"]
    assert state.evaluation["partial_render"] is True


def test_evaluate_extraction_failure_terminates_after_tier2():
    """After tier2 fails (and tier3 disabled by default), route to ExtractFail."""
    state = _state_with(
        {"title": "X", "company_name": "Y", "description": "tiny"},
        last_tier="tier2",
    )
    next_node = _run(EvaluateExtraction(), state)
    assert isinstance(next_node, ExtractFail)
