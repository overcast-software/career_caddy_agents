"""Unit tests for lib/scrape_inspector primitives.

These tests pin the contract the MCP layer (`inspect_scrape_html`,
`find_selectors_for_text`) depends on. The MCP wrappers themselves
are thin enough that exercising them through FastMCP machinery
would be ceremony — every interesting decision lives in the
primitives this file covers.
"""
from __future__ import annotations

from lib.scrape_inspector import (
    derive_hostname,
    extract_skeleton,
    find_selectors_for_text,
    query_selector,
    trim_html,
)


# A trimmed-down LinkedIn-shaped page that exercises the noise-strip
# rules, the keep-rules for aria-*/data-testid/role, and a realistic
# enough structure that selector-finding has multiple plausible
# anchors. Keep this fixture small enough that the assertions can
# point at exact substrings.
_LINKEDIN_LIKE_HTML = """
<html>
<head>
  <title>Senior Engineer | LinkedIn</title>
  <script>window.tracker.push({event:'page_view'});</script>
  <style>.j-cls-9af23b { color: blue }</style>
</head>
<body>
  <main>
    <article class="jobs-description__container" data-testid="job-details">
      <h2 aria-label="About the job" class="jobs-h2">About the job</h2>
      <div class="show-more-less-html" role="region">
        <p>Build the future. Apply now if you love TypeScript.</p>
      </div>
      <a class="jobs-apply-link" data-tracking-control-name="apply"
         href="https://example.com/apply/42" onclick="track('apply')">
        Apply on company website
      </a>
      <img src="/p.gif" width="1" height="1" alt="">
      <!-- tracking comment -->
    </article>
  </main>
</body>
</html>
"""


# --- trim_html ----------------------------------------------------------------


def test_trim_strips_scripts_styles_comments_pixels():
    out = trim_html(_LINKEDIN_LIKE_HTML)
    assert "<script>" not in out
    assert "<style>" not in out
    assert "tracking comment" not in out
    # 1x1 tracking pixel removed.
    assert 'width="1"' not in out


def test_trim_keeps_load_bearing_attrs():
    """aria-*, data-testid, role, semantic classes, id all preserved —
    these are what the enhancer anchors selectors on."""
    out = trim_html(_LINKEDIN_LIKE_HTML)
    assert 'aria-label="About the job"' in out
    assert 'data-testid="job-details"' in out
    assert 'role="region"' in out
    assert 'class="jobs-description__container"' in out


def test_trim_drops_inline_event_handlers_and_tracking_data_attrs():
    out = trim_html(_LINKEDIN_LIKE_HTML)
    assert "onclick" not in out
    assert "data-tracking-control-name" not in out


def test_trim_respects_char_limit_with_truncation_sentinel():
    huge = "<div>" + ("x" * 100_000) + "</div>"
    out = trim_html(huge, limit_chars=2_000)
    assert len(out) <= 2_100  # 2k cap + sentinel
    assert "truncated at" in out


# --- extract_skeleton ---------------------------------------------------------


def test_skeleton_strips_text_keeps_structure():
    """Visible body text is stripped; aria-label / data-testid values
    intentionally survive because they're load-bearing for selector
    orientation even on a structure-only view."""
    skel = extract_skeleton(_LINKEDIN_LIKE_HTML)
    # Body text gone — "<h2>About the job</h2>" becomes "<h2></h2>".
    assert ">About the job<" not in skel
    assert "Build the future" not in skel
    # Tag + class structure survives.
    assert "<article" in skel and "jobs-description__container" in skel
    assert "<h2" in skel
    # aria-label / data-testid intentionally preserved — they anchor
    # selectors even without text content.
    assert 'aria-label="About the job"' in skel


# --- query_selector -----------------------------------------------------------


def test_query_selector_returns_match_count_and_outline_path():
    result = query_selector(_LINKEDIN_LIKE_HTML, "h2.jobs-h2")
    assert result["selector"] == "h2.jobs-h2"
    assert result["match_count"] == 1
    match = result["matches"][0]
    assert match["text_snippet"].startswith("About the job")
    # Outline walks up the parent chain — should include the article wrapper.
    assert "article" in match["outline"]


def test_query_selector_broad_match_caps_at_max_matches():
    html = "<div>" + "<p class='x'>hi</p>" * 50 + "</div>"
    result = query_selector(html, "p.x", max_matches=5)
    assert result["match_count"] == 50  # total, not capped
    assert len(result["matches"]) == 5  # returned set is capped


def test_query_selector_returns_attrs_on_match():
    result = query_selector(
        _LINKEDIN_LIKE_HTML, "article[data-testid='job-details']",
    )
    assert result["match_count"] == 1
    assert result["matches"][0]["attrs"]["data-testid"] == "job-details"


# --- find_selectors_for_text --------------------------------------------------


def test_find_selectors_prefers_data_testid_then_aria_then_role():
    """Stability ordering is the load-bearing piece — the enhancer
    picks the top candidate by default, so the score must reflect
    real selector durability."""
    result = find_selectors_for_text(_LINKEDIN_LIKE_HTML, "About the job")
    assert result["match_count"] >= 1
    top = result["candidates"][0]
    # data-testid scores highest (stability 90) and the article above
    # the h2 carries one; aria-label scores 80 on the h2 itself.
    assert top["stability"] >= 80
    assert "About the job" in top["text_snippet"]
    selectors = {c["selector"] for c in result["candidates"]}
    assert any('aria-label="About the job"' in s for s in selectors)


def test_find_selectors_is_case_insensitive_by_default():
    result = find_selectors_for_text(_LINKEDIN_LIKE_HTML, "ABOUT THE JOB")
    assert result["match_count"] >= 1


def test_find_selectors_filters_hashed_classes():
    """A class that looks like a build-time hash (Tailwind JIT, CSS-in-JS
    artifacts) churns on every deploy and should not be proposed as an
    anchor."""
    html = (
        '<div class="abc_8d3f2a">'
        '  <span class="header__title">Apply Now</span>'
        '</div>'
    )
    result = find_selectors_for_text(html, "Apply Now")
    selectors = " ".join(c["selector"] for c in result["candidates"])
    assert "abc_8d3f2a" not in selectors
    assert "header__title" in selectors  # semantic class preserved


def test_find_selectors_empty_when_text_not_present():
    result = find_selectors_for_text(_LINKEDIN_LIKE_HTML, "this string is not here")
    assert result["match_count"] == 0
    assert result["candidates"] == []


def test_find_selectors_dedupes_across_multiple_hits():
    """If the same text appears under two anchors, repeated selectors
    must collapse so the enhancer doesn't see noise."""
    html = (
        '<div role="region" aria-label="About">'
        '  <h2>About the job</h2>'
        '</div>'
        '<div role="region" aria-label="About">'
        '  <h2>About the job</h2>'
        '</div>'
    )
    result = find_selectors_for_text(html, "About the job")
    selectors = [c["selector"] for c in result["candidates"]]
    assert len(selectors) == len(set(selectors)), "selectors must be unique"


# --- derive_hostname ----------------------------------------------------------


def test_derive_hostname_strips_www_lowercases():
    assert derive_hostname("https://www.linkedin.com/jobs/view/42") == "linkedin.com"
    assert derive_hostname("HTTPS://Indeed.COM/path") == "indeed.com"


def test_derive_hostname_handles_none_and_garbage():
    assert derive_hostname(None) is None
    assert derive_hostname("") is None
    assert derive_hostname("not a url") is None
