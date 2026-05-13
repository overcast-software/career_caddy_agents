"""Unit tests for the pure closed-state detector helpers.

Mocks Playwright's page interface for the CSS path so we don't need a
browser. The LLM path is exercised separately via test_detect_closed_state_node
where the agent call is monkey-patched.
"""
from __future__ import annotations

import asyncio
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

from scrape_graph.closed_state_detector import (
    detect_via_css,
    detect_via_phrases,
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_page(
    *,
    visible_selectors: Optional[list[str]] = None,
    inner_text_by_selector: Optional[dict[str, str]] = None,
    raise_on_query: Optional[set[str]] = None,
):
    """Build a MagicMock Playwright page where ``visible_selectors``
    return a handle whose ``inner_text()`` yields the mapped string.
    """
    visible_selectors = visible_selectors or []
    inner_text_by_selector = inner_text_by_selector or {}
    raise_on_query = raise_on_query or set()
    page = MagicMock()

    async def query_selector(sel):
        if sel in raise_on_query:
            raise RuntimeError("DOM detached")
        if sel in visible_selectors:
            handle = MagicMock()
            handle.inner_text = AsyncMock(
                return_value=inner_text_by_selector.get(sel, "")
            )
            return handle
        return None

    page.query_selector = AsyncMock(side_effect=query_selector)
    return page


# ---------------------------------------------------------------------------
# CSS path
# ---------------------------------------------------------------------------

def test_css_no_selectors_returns_empty_attempts_no_hit():
    attempts, hit = _run(detect_via_css(_make_page(), []))
    assert attempts == []
    assert hit is None


def test_css_single_match_records_attempt_and_hit():
    page = _make_page(
        visible_selectors=[".job-closed-banner"],
        inner_text_by_selector={".job-closed-banner": "no longer accepting applications"},
    )
    attempts, hit = _run(detect_via_css(page, [".job-closed-banner"]))
    assert len(attempts) == 1
    assert attempts[0]["selector"] == ".job-closed-banner"
    assert attempts[0]["matched"] is True
    assert hit == {
        "selector": ".job-closed-banner",
        "snippet": "no longer accepting applications",
    }


def test_css_runs_all_selectors_for_diff_analysis():
    """Even after the first hit, subsequent selectors are probed so the
    trace records full attempt history. Verdict still comes from the
    first hit per the contract — just don't short-circuit the loop."""
    page = _make_page(
        visible_selectors=[".banner-a", ".banner-b"],
        inner_text_by_selector={".banner-a": "A", ".banner-b": "B"},
    )
    attempts, hit = _run(detect_via_css(page, [".banner-a", ".missing", ".banner-b"]))
    assert [a["matched"] for a in attempts] == [True, False, True]
    assert hit["selector"] == ".banner-a"  # first match wins verdict
    assert hit["snippet"] == "A"


def test_css_query_exception_records_error_and_continues():
    page = _make_page(
        visible_selectors=[".banner"],
        inner_text_by_selector={".banner": "closed"},
        raise_on_query={".bad"},
    )
    attempts, hit = _run(detect_via_css(page, [".bad", ".banner"]))
    assert attempts[0]["matched"] is False
    assert "error" in attempts[0]
    assert hit["selector"] == ".banner"


# ---------------------------------------------------------------------------
# Phrase path
# ---------------------------------------------------------------------------

def test_phrase_curated_match_case_insensitive():
    text = "We are no longer accepting applications for this role."
    hit = detect_via_phrases(
        text,
        ["no longer accepting applications"],
        learned=[],
    )
    assert hit is not None
    assert hit["matched_pattern"] == "no longer accepting applications"
    assert hit["source"] == "curated"
    assert "no longer accepting applications" in hit["matched_substring"].lower()


def test_phrase_curated_takes_precedence_over_learned():
    text = "Position has been filled. We have closed this requisition."
    hit = detect_via_phrases(
        text,
        ["position has been filled"],
        learned=[{"phrase": "we have closed this requisition"}],
    )
    assert hit is not None
    assert hit["source"] == "curated"


def test_phrase_falls_through_to_learned_when_curated_empty():
    text = "We have closed this requisition. Thanks for your interest."
    hit = detect_via_phrases(
        text,
        [],
        learned=[{"phrase": "we have closed this requisition"}],
    )
    assert hit is not None
    assert hit["source"] == "learned"
    assert "we have closed this requisition" in hit["matched_substring"].lower()


def test_phrase_learned_phrase_with_regex_metacharacters_escaped():
    """A learned phrase like '[CLOSED]' must not be interpreted as a
    character class — it must match the literal substring."""
    text = "Banner: [CLOSED] applications no longer accepted."
    hit = detect_via_phrases(
        text,
        [],
        learned=[{"phrase": "[CLOSED]"}],
    )
    assert hit is not None
    assert hit["matched_substring"] == "[CLOSED]"


def test_phrase_no_match_returns_none():
    assert (
        detect_via_phrases(
            "Active job posting. Apply now. Reposted 1 week ago.",
            ["no longer accepting applications"],
            learned=[],
        )
        is None
    )


def test_phrase_empty_text_returns_none():
    assert detect_via_phrases("", ["foo"], learned=[]) is None


def test_phrase_bad_regex_skipped_not_raised():
    """A malformed curated regex shouldn't crash the scan — just log
    and try the next entry."""
    text = "We are no longer accepting applications."
    hit = detect_via_phrases(
        text,
        ["[unclosed", "no longer accepting applications"],
        learned=[],
    )
    assert hit is not None
    assert hit["matched_pattern"] == "no longer accepting applications"
