"""Unit tests for lib/scrape_inspector primitives.

These tests pin the contract the MCP layer (`inspect_scrape_html`,
`find_selectors_for_text`) depends on. The MCP wrappers themselves
are thin enough that exercising them through FastMCP machinery
would be ceremony — every interesting decision lives in the
primitives this file covers.
"""
from __future__ import annotations

import yaml

from lib.scrape_inspector import (
    css_extract_job_data,
    derive_hostname,
    extract_skeleton,
    find_selectors_for_text,
    jsonld_extract_job_data,
    query_selector,
    trim_html,
)


_JOB_PAGE_HTML = """
<html><body>
  <h1 class="job-title">Senior Backend Engineer</h1>
  <div class="company-name">Acme Corp</div>
  <section class="job-description">
    We are looking for a backend engineer with 5+ years of experience
    building distributed systems. Responsibilities include API design.
  </section>
  <span class="job-location">Remote — US</span>
  <script>var x = 1;</script>
</body></html>
"""


def test_css_extract_job_data_full_parse():
    out = css_extract_job_data(
        _JOB_PAGE_HTML,
        {
            "title": "h1.job-title",
            "company_name": ".company-name",
            "description": ".job-description",
            "location": ".job-location",
        },
    )
    assert out["title"] == "Senior Backend Engineer"
    assert out["company_name"] == "Acme Corp"
    assert out["description"].startswith("We are looking for a backend engineer")
    assert out["location"] == "Remote — US"


def test_css_extract_job_data_company_alias():
    """job_data authored with the extension's 'company' key still maps to
    the ParsedJobData 'company_name' field."""
    out = css_extract_job_data(
        _JOB_PAGE_HTML, {"company": ".company-name"},
    )
    assert out["company_name"] == "Acme Corp"


def test_css_extract_job_data_missing_selector_yields_empty():
    out = css_extract_job_data(
        _JOB_PAGE_HTML,
        {"title": "h1.job-title", "description": ".does-not-exist"},
    )
    assert out["title"] == "Senior Backend Engineer"
    assert out["description"] == ""
    assert out["company_name"] == ""  # key absent → empty, no crash


def test_css_extract_job_data_bad_selector_does_not_crash():
    # Playwright-only pseudo-class is invalid CSS3 → field stays empty.
    out = css_extract_job_data(_JOB_PAGE_HTML, {"title": "h1:has-text('x')"})
    assert out["title"] == ""


def test_css_extract_job_data_empty_inputs():
    assert css_extract_job_data("", {"title": "h1"})["title"] == ""
    assert css_extract_job_data(_JOB_PAGE_HTML, None)["title"] == ""


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


def test_query_selector_multi_class_attr_is_space_joined_and_yaml_safe():
    """Regression: BS4 4.14 returns multi-valued attrs (class, rel, …) as an
    AttributeValueList (list subclass). The previous impl put that raw list in
    the attrs dict, and api_tools._respond's yaml.safe_dump had no representer
    for it → RepresenterError ('cannot represent an object', [<class list>]) —
    crashing mode=selector on every Tailwind / styled-components page.

    The attr must be coerced to its canonical space-joined string, and the
    whole result must survive the same serializer the MCP layer uses.
    """
    html = '<h2 class="font-extrabold text-3xl" data-testid="t">About</h2>'
    result = query_selector(html, "h2")
    match = result["matches"][0]
    # Space-joined string, not a list.
    assert match["attrs"]["class"] == "font-extrabold text-3xl"
    assert isinstance(match["attrs"]["class"], str)
    # The assertion that actually reproduced the original crash: safe_dump
    # (exactly what api_tools._respond uses) must not raise.
    yaml.safe_dump(result)


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


# --- jsonld_extract_job_data --------------------------------------------------


def _jsonld_page(block: str) -> str:
    """Wrap a JSON-LD body in a minimal HTML page with the ld+json script."""
    return (
        "<html><head>"
        f'<script type="application/ld+json">{block}</script>'
        "</head><body><h1>noise</h1></body></html>"
    )


_CLEAN_JOBPOSTING = """
{
  "@context": "https://schema.org",
  "@type": "JobPosting",
  "title": "Senior Backend Engineer",
  "description": "<p>We are looking for a backend engineer with 5+ years building distributed systems.</p><ul><li>Python</li></ul>",
  "hiringOrganization": {"@type": "Organization", "name": "Acme Corp"},
  "jobLocation": {"@type": "Place", "address": {"@type": "PostalAddress", "addressLocality": "Austin", "addressRegion": "TX", "addressCountry": "US"}},
  "baseSalary": {"@type": "MonetaryAmount", "currency": "USD", "value": {"@type": "QuantitativeValue", "minValue": 120000, "maxValue": 160000, "unitText": "YEAR"}},
  "datePosted": "2026-06-01",
  "employmentType": "FULL_TIME",
  "validThrough": "2026-07-01",
  "identifier": "REQ-42"
}
"""


def test_jsonld_clean_jobposting_full_parse():
    out = jsonld_extract_job_data(_jsonld_page(_CLEAN_JOBPOSTING))
    assert out["title"] == "Senior Backend Engineer"
    assert out["company_name"] == "Acme Corp"
    # Description is collapsed PLAIN TEXT — HTML markup stripped.
    assert out["description"].startswith("We are looking for a backend engineer")
    assert "<p>" not in out["description"]
    assert "Python" in out["description"]
    assert out["location"] == "Austin, TX"
    assert out["salary_min"] == 120000
    assert out["salary_max"] == 160000
    assert out["posted_date"] == "2026-06-01"
    assert out["employment_type"] == "FULL_TIME"
    # No model home for validThrough / identifier → not emitted.
    assert "valid_through" not in out
    assert "identifier" not in out


def test_jsonld_graph_array_wrapper():
    """A JobPosting nested in a top-level @graph array is found."""
    block = """
    {
      "@context": "https://schema.org",
      "@graph": [
        {"@type": "Organization", "name": "Board Inc"},
        {"@type": "JobPosting", "title": "Data Scientist",
         "description": "Build models and pipelines for the analytics team.",
         "hiringOrganization": {"name": "DataCo"}}
      ]
    }
    """
    out = jsonld_extract_job_data(_jsonld_page(block))
    assert out["title"] == "Data Scientist"
    assert out["company_name"] == "DataCo"


def test_jsonld_multiple_blocks_one_non_jobposting():
    """Scan every ld+json block; pick the JobPosting-typed one."""
    html = (
        "<html><head>"
        '<script type="application/ld+json">'
        '{"@type": "WebSite", "name": "Careers Portal"}'
        "</script>"
        '<script type="application/ld+json">'
        '{"@type": "JobPosting", "title": "QA Engineer",'
        ' "description": "Own the test suite and CI quality gates here.",'
        ' "hiringOrganization": {"name": "TestCo"}}'
        "</script>"
        "</head><body></body></html>"
    )
    out = jsonld_extract_job_data(html)
    assert out["title"] == "QA Engineer"
    assert out["company_name"] == "TestCo"


def test_jsonld_type_as_list():
    block = """
    {"@type": ["JobPosting", "Thing"], "title": "Platform Engineer",
     "description": "Run the platform and own reliability for the org.",
     "hiringOrganization": {"name": "PlatCo"}}
    """
    out = jsonld_extract_job_data(_jsonld_page(block))
    assert out["title"] == "Platform Engineer"
    assert out["company_name"] == "PlatCo"


def test_jsonld_type_as_schema_url():
    block = """
    {"@type": "https://schema.org/JobPosting", "title": "SRE",
     "description": "Keep the lights on and the pagers quiet for the team.",
     "hiringOrganization": {"name": "UrlCo"}}
    """
    out = jsonld_extract_job_data(_jsonld_page(block))
    assert out["title"] == "SRE"
    assert out["company_name"] == "UrlCo"


def test_jsonld_missing_optional_fields():
    """Only the core trio present → core keys filled, extras omitted."""
    block = """
    {"@type": "JobPosting", "title": "Designer",
     "description": "Craft delightful interfaces across the product surface.",
     "hiringOrganization": {"name": "PixelCo"}}
    """
    out = jsonld_extract_job_data(_jsonld_page(block))
    assert out["title"] == "Designer"
    assert out["company_name"] == "PixelCo"
    assert out["location"] == ""
    assert "salary_min" not in out
    assert "salary_max" not in out
    assert "posted_date" not in out
    assert "employment_type" not in out


def test_jsonld_invalid_block_is_skipped_valid_block_wins():
    """A malformed ld+json block is skipped fail-soft; a valid sibling
    block is still parsed."""
    html = (
        "<html><head>"
        '<script type="application/ld+json">{ not: valid json,,, }</script>'
        '<script type="application/ld+json">'
        '{"@type": "JobPosting", "title": "Recovered Role",'
        ' "description": "This block is valid and should be extracted fine.",'
        ' "hiringOrganization": {"name": "RecoverCo"}}'
        "</script>"
        "</head><body></body></html>"
    )
    out = jsonld_extract_job_data(html)
    assert out["title"] == "Recovered Role"
    assert out["company_name"] == "RecoverCo"


def test_jsonld_no_jobposting_returns_empty_core():
    block = '{"@type": "WebSite", "name": "Just a site"}'
    out = jsonld_extract_job_data(_jsonld_page(block))
    assert out == {
        "title": "", "company_name": "", "description": "", "location": "",
    }


def test_jsonld_no_script_blocks_returns_empty_core():
    out = jsonld_extract_job_data("<html><body><h1>No structured data</h1></body></html>")
    assert out["title"] == ""
    assert out["company_name"] == ""


def test_jsonld_empty_html_returns_empty_core():
    out = jsonld_extract_job_data("")
    assert out == {
        "title": "", "company_name": "", "description": "", "location": "",
    }


def test_jsonld_company_as_bare_string():
    block = """
    {"@type": "JobPosting", "title": "Analyst",
     "description": "Analyze the numbers and brief the leadership team weekly.",
     "hiringOrganization": "Acme String Co"}
    """
    out = jsonld_extract_job_data(_jsonld_page(block))
    assert out["company_name"] == "Acme String Co"


def test_jsonld_salary_single_annual_value_fills_both():
    block = """
    {"@type": "JobPosting", "title": "PM",
     "description": "Own the roadmap and ship outcomes with the product squad.",
     "hiringOrganization": {"name": "RoadCo"},
     "baseSalary": {"@type": "MonetaryAmount",
       "value": {"value": 150000, "unitText": "YEAR"}}}
    """
    out = jsonld_extract_job_data(_jsonld_page(block))
    assert out["salary_min"] == 150000
    assert out["salary_max"] == 150000


def test_jsonld_salary_hourly_is_dropped():
    """JobPostData has no pay-period field, so non-annual salary is
    dropped rather than stored as a misleading bare int."""
    block = """
    {"@type": "JobPosting", "title": "Barista",
     "description": "Pull shots and keep the morning rush moving smoothly.",
     "hiringOrganization": {"name": "CafeCo"},
     "baseSalary": {"@type": "MonetaryAmount",
       "value": {"minValue": 18, "maxValue": 24, "unitText": "HOUR"}}}
    """
    out = jsonld_extract_job_data(_jsonld_page(block))
    assert "salary_min" not in out
    assert "salary_max" not in out


def test_jsonld_salary_string_amount_with_commas():
    block = """
    {"@type": "JobPosting", "title": "Lead",
     "description": "Lead the team and grow the people on it deliberately.",
     "hiringOrganization": {"name": "LeadCo"},
     "baseSalary": {"@type": "MonetaryAmount",
       "value": {"minValue": "$120,000", "maxValue": "$180,000", "unitText": "YEAR"}}}
    """
    out = jsonld_extract_job_data(_jsonld_page(block))
    assert out["salary_min"] == 120000
    assert out["salary_max"] == 180000


def test_jsonld_html_entity_escaped_description():
    """Entity-escaped markup in the description is decoded then stripped
    to plain text."""
    block = """
    {"@type": "JobPosting", "title": "Writer",
     "description": "&lt;p&gt;Great &amp; bold role for a strong communicator&lt;/p&gt;",
     "hiringOrganization": {"name": "WordCo"}}
    """
    out = jsonld_extract_job_data(_jsonld_page(block))
    assert out["description"] == "Great & bold role for a strong communicator"
    assert "<p>" not in out["description"]


def test_jsonld_location_array_takes_first_site():
    block = """
    {"@type": "JobPosting", "title": "Engineer",
     "description": "Work across multiple offices and ship features fast.",
     "hiringOrganization": {"name": "MultiCo"},
     "jobLocation": [
       {"@type": "Place", "address": {"addressLocality": "NYC", "addressRegion": "NY"}},
       {"@type": "Place", "address": {"addressLocality": "SF", "addressRegion": "CA"}}
     ]}
    """
    out = jsonld_extract_job_data(_jsonld_page(block))
    assert out["location"] == "NYC, NY"


def test_jsonld_location_country_fallback():
    block = """
    {"@type": "JobPosting", "title": "Remote Engineer",
     "description": "Fully remote role building services for a global team.",
     "hiringOrganization": {"name": "RemoteCo"},
     "jobLocation": {"address": {"addressCountry": {"@type": "Country", "name": "USA"}}}}
    """
    out = jsonld_extract_job_data(_jsonld_page(block))
    assert out["location"] == "USA"


def test_jsonld_employment_type_list_joined():
    block = """
    {"@type": "JobPosting", "title": "Contractor",
     "description": "Short-term engagement helping ship a focused deliverable.",
     "hiringOrganization": {"name": "GigCo"},
     "employmentType": ["CONTRACTOR", "PART_TIME"]}
    """
    out = jsonld_extract_job_data(_jsonld_page(block))
    assert out["employment_type"] == "CONTRACTOR, PART_TIME"
