"""CC-248 — the page's own declared canonical, read at scrape time.

Before this, NOTHING in the submodule read `link[rel="canonical"]` or
`meta[property="og:url"]`: `grep -rn canonical scrape_graph/ browser/`
returned only redirect-chain work (`ResolveFinalUrl`) and tracker-param
stripping (`url_canonicalize.py`). `state.canonical_url` was therefore
always a TRANSPORT fact — where the browser ended up — never an IDENTITY
fact the site published about itself.

THE PRECEDENCE THESE TESTS PIN
==============================

A page-declared canonical WINS over the ResolveFinalUrl-computed value,
but only behind three host-agnostic gates (absolute http(s); same host as
the LANDED url; a path that is neither empty, nor `/`, nor an auth
segment). The declaration is the site's own statement of identity and is
usually right; a declaration that fails a gate is the well-known job-board
anti-pattern (SPA shell declaring the host root, auth-wall declaring its
own login URL, syndicated listing declaring an aggregator on another
host) and must NOT clobber a good resolved URL.

Two of these are load-bearing and easy to "simplify" into bugs:

- The host gate compares against LANDED (`final_url or submitted_url`),
  never SUBMITTED. Comparing to the submitted tracker host would reject
  every legitimate declaration on exactly the redirect scrapes this work
  exists for — `test_declared_canonical_host_gate_uses_landed_not_submitted`.
- The persistence hop fires ONLY for a declaration, never for a merely
  resolved value: the resolved path already has its own propagation
  (`_propagate_canonical_to_parent_jp`, the redirect branch) and a second
  writer on `JobPost.canonical_link` buys nothing —
  `test_declared_canonical_persist_hop_skipped_for_resolved_value`.

CONVENTIONS (copied from test_resolve_final_url_js_redirect.py)
==============================================================
SimpleNamespace ctx, `asyncio.run(node.run(ctx))`, a local `_FakePage`,
and REAL NanoID-shaped 10-char ids (CC-77). Numeric-string id fixtures
have false-greened a broken `int()` cast in this graph before.

Every test name contains "canonical" — the `-k canonical` filter the
ticket prescribes matches nothing otherwise.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from scrape_graph.nodes_extract import ReviewCompleteness, UpdateProfile
from scrape_graph.nodes_scrape import Capture, DetectClosedState
from scrape_graph.state import ScrapeGraphState

SCRAPE_ID = "Kb7nQ2sVdL"   # NanoID shape (CC-77), not an int
JOB_POST_ID = "rHeRo6qWCG"


def _run(node, state: ScrapeGraphState):
    ctx = SimpleNamespace(state=state)
    return asyncio.run(node.run(ctx))


class _FakeElement:
    def __init__(self, attrs: dict):
        self._attrs = attrs

    async def get_attribute(self, name: str):
        return self._attrs.get(name)


class _FakePage:
    """Minimal Playwright Page stand-in for Capture.

    `selectors` maps a CSS selector to the attribute dict the matched
    element exposes; anything not in the map returns None (no match).
    `raise_on_query` makes every query_selector throw, standing in for a
    page that dies or a DOM that rejects the query.
    """

    def __init__(self, selectors: dict | None = None, raise_on_query: bool = False):
        self._selectors = selectors or {}
        self.raise_on_query = raise_on_query
        self.queried: list[str] = []

    async def inner_text(self, selector: str) -> str:
        return "Senior Platform Engineer\nAcme Corp\nWe are hiring."

    async def content(self) -> str:
        return "<html><body>Senior Platform Engineer</body></html>"

    async def query_selector(self, selector: str):
        self.queried.append(selector)
        if self.raise_on_query:
            raise RuntimeError("Protocol error: Node with given id not found")
        attrs = self._selectors.get(selector)
        return _FakeElement(attrs) if attrs is not None else None


def _state(
    submitted: str,
    landed: str | None = None,
    resolved_canonical: str | None = None,
    profile: dict | None = None,
) -> ScrapeGraphState:
    """A state as it looks on entry to Capture: ResolveFinalUrl has
    already run and set `canonical_url` from the landed URL.
    """
    state = ScrapeGraphState(scrape_id=SCRAPE_ID, submitted_url=submitted)
    state.final_url = landed
    state.canonical_url = resolved_canonical or landed or submitted
    state.profile = profile
    return state


async def _noop(*args, **kwargs):
    return None


def _capture(state: ScrapeGraphState, page: _FakePage):
    """Run Capture with the two unrelated side-effect helpers stubbed —
    the screenshot uploader and the legacy selector discovery both want a
    real browser and are not under test here.
    """
    # `_browser_page` is not a declared field — the runner attaches it as a
    # runtime attribute and every browser-tier node reads it via getattr.
    state._browser_page = page  # type: ignore[attr-defined]
    with patch("scrape_graph.nodes_scrape._screenshot_and_upload", _noop), \
            patch("scrape_graph.nodes_scrape._discover_selectors", _noop), \
            patch("scrape_graph.tracing._post_transition"):
        ctx = SimpleNamespace(state=state)
        return asyncio.run(Capture().run(ctx))


# ----------------------------------------------------------------------
# Rung 1 + 2 — the standard tags, one rule for every host
# ----------------------------------------------------------------------


def test_declared_canonical_link_rel_wins_over_resolved_url():
    """The acceptance case: a host emitting link[rel=canonical] has its
    declaration stored, with NO ScrapeProfile entry for that host.

    Resolved gives us `?position=2&pageNum=0` — params that survive the
    tracker strip because they are not trackers, but that are not
    identity either. The site says the posting is `/jobs/view/4453904340`.
    The site is right.
    """
    state = _state(
        submitted="https://www.example-boards.com/jobs/view/4453904340?position=2",
        landed="https://www.example-boards.com/jobs/view/4453904340?position=2&pageNum=0",
    )
    page = _FakePage({
        'link[rel="canonical"]': {
            "href": "https://www.example-boards.com/jobs/view/4453904340"
        },
    })

    nxt = _capture(state, page)

    assert isinstance(nxt, DetectClosedState)
    assert state.canonical_url == "https://www.example-boards.com/jobs/view/4453904340"
    assert state.canonical_source == "link_rel"
    # profile stayed None the whole way — zero per-host config.
    assert state.profile is None


def test_declared_canonical_og_url_used_when_link_rel_absent():
    state = _state(
        submitted="https://jobs.example.org/p/998?src=feed",
        landed="https://jobs.example.org/p/998?src=feed",
    )
    page = _FakePage({
        'meta[property="og:url"]': {"content": "https://jobs.example.org/p/998"},
    })

    _capture(state, page)

    assert state.canonical_url == "https://jobs.example.org/p/998"
    assert state.canonical_source == "og_url"


def test_declared_canonical_link_rel_preferred_over_og_url():
    """Cheapest rung first: when both tags are present, the standard
    canonical link wins and og:url is never consulted."""
    state = _state(
        submitted="https://jobs.example.org/p/998",
        landed="https://jobs.example.org/p/998",
    )
    page = _FakePage({
        'link[rel="canonical"]': {"href": "https://jobs.example.org/p/998-real"},
        'meta[property="og:url"]': {"content": "https://jobs.example.org/p/other"},
    })

    _capture(state, page)

    assert state.canonical_url == "https://jobs.example.org/p/998-real"
    assert state.canonical_source == "link_rel"
    assert 'meta[property="og:url"]' not in page.queried


def test_declared_canonical_relative_href_resolved_against_landed():
    """A relative href is legal in link[rel=canonical]. Without the
    urljoin it would have no host and fail the same-host gate."""
    state = _state(
        submitted="https://careers.example.net/listing/77?ref=nl",
        landed="https://careers.example.net/listing/77?ref=nl",
    )
    page = _FakePage({'link[rel="canonical"]': {"href": "/listing/77"}})

    _capture(state, page)

    assert state.canonical_url == "https://careers.example.net/listing/77"
    assert state.canonical_source == "link_rel"


def test_declared_canonical_tracker_params_stripped_on_adoption():
    """The adopted declaration still goes through canonicalize_url — a
    site that publishes its own canonical with utm_ on it does not get to
    poison the identity we store."""
    state = _state(
        submitted="https://jobs.example.org/p/12",
        landed="https://jobs.example.org/p/12",
    )
    page = _FakePage({
        'link[rel="canonical"]': {
            "href": "https://jobs.example.org/p/12?utm_source=rss&id=12"
        },
    })

    _capture(state, page)

    assert state.canonical_url == "https://jobs.example.org/p/12?id=12"


def test_declared_canonical_host_gate_uses_landed_not_submitted():
    """LOAD-BEARING. After a tracker redirect the LANDED host is the real
    one. Gating on the submitted host would reject the declaration on
    precisely the redirect scrapes this work exists for.
    """
    state = _state(
        submitted="https://u5250.ct.sendgrid.net/ls/click?upn=OPAQUE",
        landed="https://hiring.example.cafe/job/abc123?utm_medium=email",
        resolved_canonical="https://hiring.example.cafe/job/abc123",
    )
    page = _FakePage({
        'link[rel="canonical"]': {
            "href": "https://hiring.example.cafe/job/abc123"
        },
    })

    _capture(state, page)

    assert state.canonical_url == "https://hiring.example.cafe/job/abc123"
    assert state.canonical_source == "link_rel"


def test_declared_canonical_www_variant_accepted():
    """`www.` is stripped on both sides before the host compare, matching
    the repo-wide profile-lookup convention."""
    state = _state(
        submitted="https://example-boards.com/jobs/5",
        landed="https://example-boards.com/jobs/5",
    )
    page = _FakePage({
        'link[rel="canonical"]': {"href": "https://www.example-boards.com/jobs/5"},
    })

    _capture(state, page)

    assert state.canonical_url == "https://www.example-boards.com/jobs/5"
    assert state.canonical_source == "link_rel"


# ----------------------------------------------------------------------
# The junk filter — a bad declaration must NOT clobber a good resolve
# ----------------------------------------------------------------------


def test_declared_canonical_host_root_rejected():
    """The SPA-shell anti-pattern: a hard-coded `<link rel=canonical
    href="https://site/">` on every page. It identifies nothing."""
    resolved = "https://spa.example.io/careers/openings/9912"
    state = _state(submitted=resolved, landed=resolved)
    page = _FakePage({'link[rel="canonical"]': {"href": "https://spa.example.io/"}})

    _capture(state, page)

    assert state.canonical_url == resolved
    assert state.canonical_source == "resolved"


def test_declared_canonical_cross_host_rejected():
    """A syndication pointer at another host is not this document's
    identity — we never fetched it and cannot verify it names this job."""
    resolved = "https://boards.example-ats.com/acme/jobs/42"
    state = _state(submitted=resolved, landed=resolved)
    page = _FakePage({
        'link[rel="canonical"]': {
            "href": "https://aggregator.example.com/listing/42"
        },
    })

    _capture(state, page)

    assert state.canonical_url == resolved
    assert state.canonical_source == "resolved"


def test_declared_canonical_login_path_rejected():
    """An auth-wall declares the WALL, not what is behind it. The
    rejection is by whole path segment against a closed set of generic
    web words — no hostname is involved."""
    resolved = "https://walled.example.com/jobs/view/4404587081"
    state = _state(submitted=resolved, landed=resolved)
    page = _FakePage({
        'link[rel="canonical"]': {
            "href": "https://walled.example.com/login?session_redirect=%2Fjobs"
        },
    })

    _capture(state, page)

    assert state.canonical_url == resolved
    assert state.canonical_source == "resolved"


def test_declared_canonical_job_slug_containing_auth_word_still_adopted():
    """The auth gate matches WHOLE segments, so an ordinary posting slug
    that merely contains one of the words survives. Guards against the
    obvious over-correction (a substring check would eat this)."""
    state = _state(
        submitted="https://jobs.example.org/careers/account-manager-1187?ref=x",
        landed="https://jobs.example.org/careers/account-manager-1187?ref=x",
    )
    page = _FakePage({
        'link[rel="canonical"]': {
            "href": "https://jobs.example.org/careers/account-manager-1187"
        },
    })

    _capture(state, page)

    assert (
        state.canonical_url
        == "https://jobs.example.org/careers/account-manager-1187"
    )
    assert state.canonical_source == "link_rel"


def test_declared_canonical_non_http_scheme_rejected():
    resolved = "https://jobs.example.org/p/3"
    state = _state(submitted=resolved, landed=resolved)
    page = _FakePage({
        'link[rel="canonical"]': {"href": "javascript:void(0)"},
    })

    _capture(state, page)

    assert state.canonical_url == resolved
    assert state.canonical_source == "resolved"


# ----------------------------------------------------------------------
# Degrade, never raise
# ----------------------------------------------------------------------


def test_declared_canonical_absent_falls_through_to_resolved():
    """The normal case for most of the web: no declaration, today's
    behaviour unchanged."""
    resolved = "https://plain.example.com/jobs/1"
    state = _state(submitted=resolved, landed=resolved)
    page = _FakePage({})

    nxt = _capture(state, page)

    assert isinstance(nxt, DetectClosedState)
    assert state.canonical_url == resolved
    assert state.canonical_source == "resolved"


def test_declared_canonical_query_selector_exception_is_swallowed():
    """A DOM that throws must not fail the scrape — the capture still
    lands and the graph still routes to DetectClosedState."""
    resolved = "https://brittle.example.com/jobs/1"
    state = _state(submitted=resolved, landed=resolved)
    page = _FakePage({}, raise_on_query=True)

    nxt = _capture(state, page)

    assert isinstance(nxt, DetectClosedState)
    assert state.canonical_url == resolved
    assert state.canonical_source == "resolved"
    assert state.job_content  # the capture itself still happened


# ----------------------------------------------------------------------
# Rung 3 — the ScrapeProfile escape hatch, and ONLY as a last resort
# ----------------------------------------------------------------------


def test_declared_canonical_profile_selector_used_when_standard_tags_absent():
    """`extension_selectors` arrives NESTED on state.profile:
    `_flatten_profile_attrs` lifts the keys of `css_selectors` only.
    """
    state = _state(
        submitted="https://odd.example.com/req/551",
        landed="https://odd.example.com/req/551",
        profile={
            "job_data": {"title": "h1"},
            "extension_selectors": {
                "canonical_link_selectors": ["a.permalink"],
            },
        },
    )
    page = _FakePage({
        "a.permalink": {"href": "https://odd.example.com/req/551-canonical"},
    })

    _capture(state, page)

    assert state.canonical_url == "https://odd.example.com/req/551-canonical"
    assert state.canonical_source == "profile_selector"


def test_declared_canonical_profile_selector_not_consulted_when_link_rel_present():
    """Rung 3 is an escape hatch, not a co-equal. A host with a working
    standard tag must never reach its profile entry."""
    state = _state(
        submitted="https://odd.example.com/req/551",
        landed="https://odd.example.com/req/551",
        profile={
            "extension_selectors": {
                "canonical_link_selectors": ["a.permalink"],
            },
        },
    )
    page = _FakePage({
        'link[rel="canonical"]': {"href": "https://odd.example.com/req/551-std"},
        "a.permalink": {"href": "https://odd.example.com/req/551-profile"},
    })

    _capture(state, page)

    assert state.canonical_url == "https://odd.example.com/req/551-std"
    assert state.canonical_source == "link_rel"
    assert "a.permalink" not in page.queried


def test_declared_canonical_profile_selector_junk_still_gated():
    """Rung 3 buys no exemption from the junk filter — a profile entry
    pointing at another host is rejected exactly like a page tag."""
    resolved = "https://odd.example.com/req/551"
    state = _state(
        submitted=resolved,
        landed=resolved,
        profile={
            "extension_selectors": {
                "canonical_link_selectors": ["a.permalink"],
            },
        },
    )
    page = _FakePage({"a.permalink": {"href": "https://elsewhere.example/req/551"}})

    _capture(state, page)

    assert state.canonical_url == resolved
    assert state.canonical_source == "resolved"


# ----------------------------------------------------------------------
# The persistence hop — without it the whole ladder dies in memory
# ----------------------------------------------------------------------


def _extract_state(canonical_source: str) -> ScrapeGraphState:
    state = ScrapeGraphState(
        scrape_id=SCRAPE_ID,
        submitted_url="https://jobs.example.org/p/12?position=2",
    )
    state.job_post_id = JOB_POST_ID
    state.canonical_url = "https://jobs.example.org/p/12"
    state.canonical_source = canonical_source
    return state


def test_declared_canonical_persist_hop_patches_job_post():
    state = _extract_state("link_rel")

    with patch("scrape_graph.nodes_extract.httpx.patch") as mock_patch, \
            patch("scrape_graph.tracing._post_transition"):
        mock_patch.return_value.status_code = 200
        nxt = _run(ReviewCompleteness(), state)

    assert isinstance(nxt, UpdateProfile)
    assert mock_patch.call_count == 1
    _, kwargs = mock_patch.call_args
    attrs = kwargs["json"]["data"]["attributes"]
    assert attrs == {"canonical_link": "https://jobs.example.org/p/12"}
    # `link` is never touched — the stored original link is preserved.
    assert "link" not in attrs
    assert JOB_POST_ID in mock_patch.call_args[0][0]


def test_declared_canonical_persist_hop_skipped_for_resolved_value():
    """LOAD-BEARING. A merely-resolved canonical already has its own
    propagation path in the ResolveFinalUrl redirect branch; writing it
    from here too would put a second writer on the same column."""
    state = _extract_state("resolved")

    with patch("scrape_graph.nodes_extract.httpx.patch") as mock_patch, \
            patch("scrape_graph.tracing._post_transition"):
        mock_patch.return_value.status_code = 200
        nxt = _run(ReviewCompleteness(), state)

    assert isinstance(nxt, UpdateProfile)
    assert mock_patch.call_count == 0


def test_declared_canonical_persist_hop_failure_does_not_break_the_graph():
    state = _extract_state("og_url")

    with patch("scrape_graph.nodes_extract.httpx.patch") as mock_patch, \
            patch("scrape_graph.tracing._post_transition"):
        mock_patch.side_effect = RuntimeError("connection reset")
        nxt = _run(ReviewCompleteness(), state)

    assert isinstance(nxt, UpdateProfile)


def test_declared_canonical_source_recorded_on_capture_trace():
    """Observability: which rung won is visible on the per-scrape trace
    without a re-scrape."""
    state = _state(
        submitted="https://jobs.example.org/p/998",
        landed="https://jobs.example.org/p/998",
    )
    page = _FakePage({
        'link[rel="canonical"]': {"href": "https://jobs.example.org/p/998"},
    })

    _capture(state, page)

    entry = [e for e in state.node_trace if e.node == "Capture"][-1]
    assert entry.payload["canonical_source"] == "link_rel"
    assert entry.payload["canonical_url"] == "https://jobs.example.org/p/998"
