"""DOM-level tests for ``_CANDIDATE_SCAN_JS`` (CC-285).

``tests/test_apply_resolver.py`` mocks ``page.evaluate`` outright, so it
exercises the Python wrapper and never the scan JS itself. That is why
the div-blindness in CC-285 survived: no test ever ran the query
selectors against a real document.

These tests run the real JS in a real (headless chromium) DOM via
``page.set_content``. Chromium is a hard dependency of this repo
(playwright is pinned in pyproject), but the browser binary is fetched
separately — if it is absent the module skips rather than failing.
"""
from __future__ import annotations

import asyncio

import pytest

from scrape_graph.apply_resolver import scan_apply_candidates


def _scan_html(html: str) -> list[dict]:
    """Render ``html`` in headless chromium and run the real scan on it."""

    async def _go():
        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                page = await browser.new_page()
                await page.set_content(html)
                return await scan_apply_candidates(page)
            finally:
                await browser.close()

    try:
        return asyncio.run(_go())
    except Exception as exc:  # pragma: no cover - environment guard
        msg = str(exc)
        if "Executable doesn't exist" in msg or "playwright install" in msg:
            pytest.skip(f"chromium not installed: {msg.splitlines()[0]}")
        raise


def _by_text(results: list[dict], needle: str) -> dict | None:
    for r in results:
        if needle.lower() in (r.get("text") or "").lower():
            return r
    return None


# The CC-285 fixture: jobright.ai renders the control that actually leads
# to the employer ATS as a plain div, next to a button that is its own
# browser-plugin CTA. Before the fix only the button came back.
JOBRIGHT_SHAPED = """
<html><head><style>
  .apply-external { cursor: pointer; }
</style></head><body>
  <div class="apply-external">Apply on Employer Site</div>
  <button class="ant-btn">Apply With Autofill</button>
</body></html>
"""


def test_div_apply_control_is_found_alongside_button():
    results = _scan_html(JOBRIGHT_SHAPED)

    div = _by_text(results, "Apply on Employer Site")
    button = _by_text(results, "Apply With Autofill")

    assert div is not None, f"div-rendered apply control missed: {results}"
    assert button is not None, f"button apply control missed: {results}"

    assert div["tag"] == "div"
    assert div["selector"] == "div.apply-external"
    assert div["href"] is None
    assert "cursor: pointer" in div["reason"]

    assert button["tag"] == "button"


def test_span_with_onclick_is_found():
    results = _scan_html(
        '<span class="apply-cta" onclick="go()">Apply now</span>'
    )
    hit = _by_text(results, "Apply now")
    assert hit is not None
    assert hit["tag"] == "span"
    assert "onclick handler" in hit["reason"]


def test_anchor_without_href_is_found_via_tabindex():
    results = _scan_html(
        '<a class="apply-link" tabindex="0">Apply on Employer Site</a>'
    )
    hit = _by_text(results, "Apply on Employer Site")
    assert hit is not None
    assert hit["tag"] == "a"
    assert hit["href"] is None
    assert "tabindex" in hit["reason"]


def test_applyish_prose_without_click_affordance_is_ignored():
    """Precision guard — the affordance gate is what keeps pass 3 honest."""
    # A leaf div, so the affordance gate is the only thing that can
    # reject it — the wrapper guard has nothing to bite on.
    results = _scan_html(
        "<div>To apply, send your resume to jobs@example.com.</div>"
    )
    assert results == [], results


def test_wrapper_div_does_not_shadow_the_real_control():
    """Only the innermost applyish node is a candidate, not its wrappers."""
    results = _scan_html(
        '<html><head><style>* { cursor: pointer; }</style></head><body>'
        '<div class="outer"><div class="inner">Apply on Employer Site</div>'
        "</div></body></html>"
    )
    assert len(results) == 1, results
    assert results[0]["selector"] == "div.inner"


def test_button_is_not_double_reported_by_pass_three():
    results = _scan_html(
        '<html><head><style>* { cursor: pointer; }</style></head><body>'
        '<a role="button" class="apply">Apply Now</a></body></html>'
    )
    assert len(results) == 1, results
    assert results[0]["tag"] == "a"


def test_pass_three_is_capped_so_a_text_heavy_page_cannot_flood():
    rows = "".join(
        f'<div class="apply-row-{i}">Apply to role {i}</div>' for i in range(60)
    )
    html = (
        "<html><head><style>div { cursor: pointer; }</style></head><body>"
        f"{rows}</body></html>"
    )
    results = _scan_html(html)
    assert len(results) == 15, len(results)
