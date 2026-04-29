"""Unit tests for the apply-destination resolver pure logic.

Mocks Playwright's page interface so we don't need a browser or network.
"""
from __future__ import annotations

import asyncio
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from scrape_graph.apply_resolver import (
    resolve_apply_url,
    scan_apply_candidates,
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_page(
    *,
    visible_selectors: Optional[list[str]] = None,
    href_by_selector: Optional[dict[str, str]] = None,
    button_lands_at: Optional[str] = None,
    raise_on_click: bool = False,
):
    """Build a MagicMock Playwright page where:

    - ``visible_selectors``: query_selector returns truthy for these.
    - ``href_by_selector``: query_selector returns a handle whose
      get_attribute('href') yields the mapped value.
    - ``button_lands_at``: clicking opens a new page whose ``.url`` is this.
    - ``raise_on_click``: simulate anti-bot / nav timeout.
    """
    visible_selectors = visible_selectors or []
    href_by_selector = href_by_selector or {}
    page = MagicMock()
    page.url = "https://example.com/job/1"

    async def query_selector(sel):
        if sel in visible_selectors or sel in href_by_selector:
            handle = MagicMock()
            handle.get_attribute = AsyncMock(
                return_value=href_by_selector.get(sel)
            )
            handle.click = AsyncMock(
                side_effect=Exception("blocked") if raise_on_click else None
            )
            return handle
        return None

    page.query_selector = AsyncMock(side_effect=query_selector)

    new_page = MagicMock()
    new_page.url = button_lands_at or ""
    new_page.wait_for_load_state = AsyncMock()
    new_page.close = AsyncMock()

    class _PageCtx:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        @property
        def value(self):
            async def _v():
                if button_lands_at is None or raise_on_click:
                    raise TimeoutError("no new page")
                return new_page
            return _v()

    context = MagicMock()
    context.expect_page = MagicMock(return_value=_PageCtx())
    page.context = context
    page.wait_for_url = AsyncMock(side_effect=TimeoutError("no nav"))
    return page


def test_no_browser_page_returns_unknown():
    result = _run(resolve_apply_url(None, {"apply_link_selectors": ["a"]}))
    assert result["apply_url_status"] == "unknown"
    assert result["apply_url"] is None
    assert "no_browser_page" in result["reason"]


def test_no_config_returns_unknown():
    page = _make_page()
    result = _run(resolve_apply_url(page, None))
    assert result["apply_url_status"] == "unknown"
    assert "no_config" in result["reason"]


def test_empty_config_returns_unknown():
    page = _make_page()
    result = _run(resolve_apply_url(page, {}))
    assert result["apply_url_status"] == "unknown"


def test_internal_marker_short_circuits():
    page = _make_page(visible_selectors=[".easy-apply"])
    config = {
        "internal_apply_markers": [".easy-apply"],
        "apply_link_selectors": ["a.never-tried"],
    }
    result = _run(resolve_apply_url(page, config))
    assert result["apply_url_status"] == "internal"
    assert result["apply_url"] is None
    assert ".easy-apply" in result["reason"]


def test_link_selector_returns_resolved():
    page = _make_page(href_by_selector={"a.apply": "https://ats.example/p/1"})
    result = _run(resolve_apply_url(page, {"apply_link_selectors": ["a.apply"]}))
    assert result["apply_url_status"] == "resolved"
    assert result["apply_url"] == "https://ats.example/p/1"


def test_link_selector_skips_non_http():
    page = _make_page(href_by_selector={"a.apply": "javascript:void(0)"})
    result = _run(resolve_apply_url(page, {"apply_link_selectors": ["a.apply"]}))
    # No usable link, no buttons; selector list was non-empty so failed.
    assert result["apply_url_status"] == "failed"


def test_link_selector_tried_in_order():
    page = _make_page(href_by_selector={"a.second": "https://ats.example/2"})
    result = _run(resolve_apply_url(
        page,
        {"apply_link_selectors": ["a.first-misses", "a.second"]},
    ))
    assert result["apply_url"] == "https://ats.example/2"


def test_button_click_captures_landing_url():
    page = _make_page(
        visible_selectors=["button.apply"],
        button_lands_at="https://ats.example/applied",
    )
    result = _run(resolve_apply_url(
        page, {"apply_button_selectors": ["button.apply"]},
    ))
    assert result["apply_url_status"] == "resolved"
    assert result["apply_url"] == "https://ats.example/applied"


def test_button_click_anti_bot_returns_failed():
    page = _make_page(
        visible_selectors=["button.apply"], raise_on_click=True,
    )
    result = _run(resolve_apply_url(
        page, {"apply_button_selectors": ["button.apply"]},
    ))
    assert result["apply_url_status"] == "failed"


def test_link_takes_priority_over_button():
    page = _make_page(
        href_by_selector={"a.link": "https://ats.example/from-link"},
        visible_selectors=["button.btn"],
        button_lands_at="https://ats.example/from-button",
    )
    config = {
        "apply_link_selectors": ["a.link"],
        "apply_button_selectors": ["button.btn"],
    }
    result = _run(resolve_apply_url(page, config))
    assert result["apply_url"] == "https://ats.example/from-link"


@pytest.mark.parametrize(
    "config", [
        {"apply_link_selectors": ["a.miss"]},
        {"apply_button_selectors": ["button.miss"]},
    ],
)
def test_no_match_with_non_empty_config_is_failed(config):
    page = _make_page()
    result = _run(resolve_apply_url(page, config))
    assert result["apply_url_status"] == "failed"


# ----- Phase 3: scan_apply_candidates ----------------------------------


def _scan_page(returned_results):
    """Page mock whose page.evaluate(JS) returns the given list."""
    page = MagicMock()
    page.evaluate = AsyncMock(return_value=returned_results)
    return page


def test_scan_returns_empty_when_no_page():
    assert _run(scan_apply_candidates(None)) == []


def test_scan_returns_empty_on_evaluate_crash():
    page = MagicMock()
    page.evaluate = AsyncMock(side_effect=Exception("page crashed"))
    assert _run(scan_apply_candidates(page)) == []


def test_scan_returns_empty_on_non_list_payload():
    page = _scan_page({"not": "a list"})
    assert _run(scan_apply_candidates(page)) == []


def test_scan_sorts_by_score_descending():
    page = _scan_page([
        {"selector": "a.low", "href": "https://x/a", "text": "Apply",
         "tag": "a", "score": 0.4, "reason": "text"},
        {"selector": "a.high", "href": "https://x/b", "text": "Apply Now",
         "tag": "a", "score": 0.9, "reason": "href + text"},
        {"selector": "a.mid", "href": None, "text": "apply",
         "tag": "a", "score": 0.7, "reason": "text"},
    ])
    result = _run(scan_apply_candidates(page))
    scores = [r["score"] for r in result]
    assert scores == sorted(scores, reverse=True)
    assert result[0]["selector"] == "a.high"


def test_scan_drops_entries_missing_selector():
    page = _scan_page([
        {"selector": "a.keep", "href": "h", "text": "t",
         "tag": "a", "score": 0.5, "reason": "r"},
        {"selector": "", "href": "h", "text": "t",
         "tag": "a", "score": 0.9, "reason": "r"},
        "not a dict",
    ])
    result = _run(scan_apply_candidates(page))
    assert len(result) == 1
    assert result[0]["selector"] == "a.keep"


def test_scan_caps_at_max_candidates():
    payload = [
        {"selector": f"a.c{i}", "href": "h", "text": "Apply",
         "tag": "a", "score": 0.1, "reason": "r"}
        for i in range(120)
    ]
    page = _scan_page(payload)
    result = _run(scan_apply_candidates(page, max_candidates=10))
    assert len(result) == 10
