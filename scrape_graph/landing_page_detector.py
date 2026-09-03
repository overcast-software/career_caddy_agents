"""Search-landing / interstitial detection — pure logic split out of the
Capture node so each signal is testable without pydantic-graph plumbing
(same shape as ``closed_state_detector``).

WHY THIS EXISTS (CC-226). When a URL lands somewhere that will never
satisfy the profile's ``ready_selector`` — a tracker that bounces to the
site's job *search* page, an interstitial, a generic "browse openings"
shell — the graph used to march the whole way down anyway: SettleWait,
ScrollToLoad, ExpandTruncations, a screenshot upload, DetectClosedState's
LLM leg, a PATCH of the full DOM, and then the entire extraction ladder.
Tier1 and Tier2 each carry a 120s HTTP timeout (``_call_llm_extract``),
so two rungs alone can reach the runner's 240s ``GRAPH_RUN_TIMEOUT_S``
cap. ~170 such timeouts fired over 2026-07-20..29, across linkedin,
adzuna, jobright and ziprecruiter — cross-domain, not one bad profile.

The exit is deliberately taken at Capture rather than at WaitReadySelector
even though the ticket names both. Two reasons:

1. A ready_selector miss is NOT by itself evidence that the page is a
   landing page. Profiles go stale (see CC-26 — readiness() does not
   verify live matches), so a stale selector on a perfectly good detail
   page misses too. Killing on the miss alone would convert a profile
   bug into a refused scrape. Positive landing evidence is required.
2. Exiting before ScrollToLoad would throw away the lazy-hydration
   rescue that fixes real LinkedIn detail pages. ScrollToLoad costs ~5s;
   the tail we skip costs up to ~200s plus two LLM calls. Waiting for
   Capture buys the safety for ~5% of the savings.

DETECTION CONTRACT. Two independent landing signals must fire AND the
page must NOT look like a job detail page. Requiring corroboration is
what keeps a single unlucky phrase ("we have 3 openings on this team")
from refusing a real posting. All signals are computed off the captured
visible text plus the landed URL — no DOM, no LLM, no per-host config.
"""
from __future__ import annotations

import re
from typing import Optional
from urllib.parse import parse_qs, urlparse

# Result-count phrasing: "1,234 jobs", "Showing 1-25 of 300", "20 results
# found". Alone it is weak — a JD can say "3 openings on this team" — so
# it only ever contributes one of the two required signals.
_RESULT_COUNT_RE = re.compile(
    r"\b\d[\d,]*\s*\+?\s*(?:jobs?|results?|vacanc(?:y|ies)|positions?|openings?|matches)\b"
    r"|\bshowing\s+\d[\d,]*\s*(?:-|–|to)\s*\d[\d,]*\b",
    re.IGNORECASE,
)

# Listing furniture. Any ONE of these can appear beside a real posting
# (LinkedIn renders a results rail next to the detail pane), so the
# signal requires _LISTING_CONTROL_MIN_HITS distinct phrases.
_LISTING_CONTROL_PHRASES = (
    "sort by",
    "most relevant",
    "date posted",
    "refine your search",
    "refine search",
    "search results",
    "results for",
    "next page",
    "previous page",
    "page 1 of",
    "jobs per page",
    "create job alert",
    "create a job alert",
    "save this search",
    "browse jobs",
    "browse all jobs",
    "similar searches",
    "no results found",
    "did you mean",
    "broaden your search",
)
_LISTING_CONTROL_MIN_HITS = 2

# The search form itself. Adzuna's landing page is literally a "What?" /
# "Where?" pair; other boards spell it out. Both halves must be present —
# a lone "where" is a location label on a real posting.
_SEARCH_WHAT_PROMPTS = (
    "what?",
    "job title, keywords",
    "job title or keyword",
    "job title, skills",
    "keywords or job title",
    "search jobs by keyword",
    "what job are you looking for",
)
_SEARCH_WHERE_PROMPTS = (
    "where?",
    "city, state or zip",
    "city, state, or zip",
    "city or postcode",
    "town or postcode",
    "city, state or postcode",
    "location or remote",
)

# Query params and path shapes that mean "this is a search results URL".
_SEARCH_QUERY_PARAMS = frozenset({"q", "query", "keywords", "kw", "what", "search", "searchterm"})
_SEARCH_PATH_SEGMENTS = frozenset({"search", "searchjobs", "results", "browse", "browse-jobs"})

# Job-detail vocabulary. Two or more of these mean the page is carrying a
# real posting body and must NOT be refused, whatever the listing
# furniture around it says. "equal opportunity employer" and the
# what-you'll-do variants are the strongest members.
_DETAIL_PHRASES = (
    "responsibilities",
    "qualifications",
    "preferred qualifications",
    "minimum qualifications",
    "requirements",
    "what you'll do",
    "what you will do",
    "about the role",
    "about the job",
    "about this role",
    "job description",
    "years of experience",
    "equal opportunity employer",
    "who you are",
    "what we offer",
    "apply for this job",
)
_DETAIL_MIN_HITS = 2

# Two corroborating landing signals. One is not enough: every individual
# signal has a plausible innocent explanation on a real posting page.
_LANDING_MIN_SIGNALS = 2


def _looks_like_search_url(url: str) -> bool:
    """Does the landed URL itself say 'search results'?

    Checks a closed set of query params and whole path segments — the
    same uniform-web-convention approach as ``_AUTH_PATH_SEGMENTS`` in
    nodes_scrape. No hostname appears here; a host-specific rule would be
    the signal to reach for ScrapeProfile.url_rewrites instead.
    """
    if not url:
        return False
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    params = {k.lower() for k in parse_qs(parsed.query or "")}
    if params & _SEARCH_QUERY_PARAMS:
        return True
    segments = {seg.lower() for seg in (parsed.path or "").split("/") if seg}
    return bool(segments & _SEARCH_PATH_SEGMENTS)


def landing_signals(text: str, url: str = "") -> list[str]:
    """Names of the landing signals that fire for this page. Order is
    stable so the trace payload diffs cleanly across runs."""
    lowered = (text or "").lower()
    found: list[str] = []

    if _RESULT_COUNT_RE.search(lowered):
        found.append("result_count")

    hits = [p for p in _LISTING_CONTROL_PHRASES if p in lowered]
    if len(hits) >= _LISTING_CONTROL_MIN_HITS:
        found.append("listing_controls")

    if any(p in lowered for p in _SEARCH_WHAT_PROMPTS) and any(
        p in lowered for p in _SEARCH_WHERE_PROMPTS
    ):
        found.append("search_form")

    if _looks_like_search_url(url):
        found.append("search_url")

    return found


def detail_signals(text: str) -> list[str]:
    """Names of the job-detail phrases present. Two or more veto a
    landing verdict outright."""
    lowered = (text or "").lower()
    return [p for p in _DETAIL_PHRASES if p in lowered]


def detect_landing_page(
    text: str,
    *,
    url: str = "",
    ready_selector_configured: bool,
    ready_selector_matched: bool,
) -> Optional[dict]:
    """Verdict dict when this page is a search-landing / interstitial,
    else ``None``.

    Preconditions, both owned by the caller and both required:

    - ``ready_selector_configured`` — with no configured expectation of
      what "ready" looks like we have no business declaring the page
      wrong. Unprofiled hosts always fall through.
    - ``not ready_selector_matched`` — if the detail anchor DID appear at
      any point (WaitReadySelector or ScrollToLoad), the page is a detail
      page with listing furniture around it, full stop.

    The returned dict is the trace payload: it names every signal on both
    sides so a false positive is diagnosable from the trace alone,
    without re-running the scrape.
    """
    if not ready_selector_configured or ready_selector_matched:
        return None

    landing = landing_signals(text, url)
    detail = detail_signals(text)
    if len(landing) < _LANDING_MIN_SIGNALS:
        return None
    if len(detail) >= _DETAIL_MIN_HITS:
        return None

    return {
        "verdict": "landing_page",
        "landing_signals": landing,
        "detail_signals": detail,
    }
