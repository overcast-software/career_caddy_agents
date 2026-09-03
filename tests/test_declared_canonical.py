"""CC-248 — the page's own declared canonical, read at scrape time.

Before this, NOTHING in the submodule read `link[rel~="canonical"]` or
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

Five of these are load-bearing and easy to "simplify" into bugs:

- The host gate compares against LANDED (`final_url or submitted_url`),
  never SUBMITTED. Comparing to the submitted tracker host would reject
  every legitimate declaration on exactly the redirect scrapes this work
  exists for — `test_declared_canonical_host_gate_uses_landed_not_submitted`.
- The junk filter runs PER RUNG, INSIDE the ladder, with fall-through.
  Gate the ladder's output once at the caller instead and a junk rung 1
  short-circuits rungs 2 and 3 — the SPA shell that hard-codes
  `<link rel=canonical href="https://site/">` on every page while still
  emitting a correct og:url is the motivating case, and the operator's
  profile escape hatch could never rescue it —
  `test_declared_canonical_junk_link_rel_falls_through_to_og_url`,
  `test_declared_canonical_profile_rung_reached_when_both_standard_tags_junk`.
- Each rung reads inside its OWN try/except, so a throwing selector on
  one rung cannot blind the rungs below it —
  `test_declared_canonical_throwing_rung_does_not_blind_the_rungs_below`.
- `_adopt_declared_canonical` runs BEFORE Capture's content reads. It
  re-raises driver-closed errors (CC-160); below the reads that raise
  would throw away a complete capture and buy a full re-scrape —
  `test_declared_canonical_driver_death_raises_before_any_bytes_are_banked`.
- The persistence hop fires ONLY for a declaration, never for a merely
  resolved value (the resolved path already has its own propagation,
  `_propagate_canonical_to_parent_jp`, and a second writer on
  `JobPost.canonical_link` buys nothing), and NEVER on the duplicate
  path, where `job_post_id` is a pre-existing row the api matched us
  onto — often another user's — and `canonical_link` is the dedupe key
  we would be rewriting —
  `test_declared_canonical_persist_hop_skipped_for_resolved_value`,
  `test_declared_canonical_persist_hop_skipped_on_duplicate_job_post`.

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

import pytest

from scrape_graph.nodes_extract import (
    PersistJobPost,
    ReviewCompleteness,
    UpdateProfile,
)
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


# The exact wire selectors Capture queries. `rel~=` (not `rel=`) because
# `rel` is a space-separated TOKEN LIST: `<link rel="canonical alternate">`
# is legal and appears on syndicated / i18n pages, and an exact-value
# selector silently misses it. Named here so a "simplification" back to
# `rel=` breaks every rung-1 test at once instead of quietly halving
# rung-1 coverage in production.
LINK_REL = 'link[rel~="canonical"]'
OG_URL = 'meta[property="og:url"]'
DRIVER_DEAD = "Connection closed while reading from the driver"


class _FakePage:
    """Minimal Playwright Page stand-in for Capture.

    `selectors` maps a CSS selector to the attribute dict the matched
    element exposes; anything not in the map returns None (no match).
    `raise_on_query` makes every query_selector throw, standing in for a
    page that dies or a DOM that rejects the query; `query_error` sets
    the message (the driver-closed marker phrase is load-bearing —
    `is_driver_closed` substring-matches it). `raise_on` throws for
    named selectors only, so a single bad rung can be tested against
    healthy ones.

    `read_body` records whether the content reads ever happened, which is
    how the tests below pin Capture's ORDERING.
    """

    def __init__(
        self,
        selectors: dict | None = None,
        raise_on_query: bool = False,
        query_error: str = "Protocol error: Node with given id not found",
        raise_on: dict | None = None,
    ):
        self._selectors = selectors or {}
        self.raise_on_query = raise_on_query
        self.query_error = query_error
        self._raise_on = raise_on or {}
        self.queried: list[str] = []
        self.read_body = False

    async def inner_text(self, selector: str) -> str:
        self.read_body = True
        return "Senior Platform Engineer\nAcme Corp\nWe are hiring."

    async def content(self) -> str:
        return "<html><body>Senior Platform Engineer</body></html>"

    async def query_selector(self, selector: str):
        self.queried.append(selector)
        if self.raise_on_query:
            raise RuntimeError(self.query_error)
        if selector in self._raise_on:
            raise RuntimeError(self._raise_on[selector])
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
        'link[rel~="canonical"]': {
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
        'link[rel~="canonical"]': {"href": "https://jobs.example.org/p/998-real"},
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
    page = _FakePage({'link[rel~="canonical"]': {"href": "/listing/77"}})

    _capture(state, page)

    assert state.canonical_url == "https://careers.example.net/listing/77"
    assert state.canonical_source == "link_rel"


def test_declared_canonical_adopted_verbatim_api_owns_the_stripping():
    """The declaration is adopted AS PUBLISHED; stripping is the api's job.

    REPLACES test_declared_canonical_tracker_params_stripped_on_adoption,
    which asserted that adoption ran the declaration through this module's
    canonicalize_url. That was removed deliberately, and the concern it
    protected is unchanged — a site publishing its own canonical with utm_ on
    it still must not poison the stored identity. What moved is WHO enforces
    it.

    Why it moved: whatever lands in `state.canonical_url` is PATCHed onto
    JobPost.canonical_link by ReviewCompleteness, and that column is the api's
    primary dedupe key. Canonicalizing here wrote an agents-shaped value into
    it — this module's param set is disjoint from the api's (not a superset),
    it applies no ScrapeProfile url_rewrites, and it strips `src`, which the
    api deliberately KEEPS because worksourcewa encodes part of the job id
    there. So the api's own matcher could not reproduce the stored key for the
    same input URL.

    The guarantee now lives in the api, which canonicalizes an inbound
    canonical_link at write (api PR #271) — one owner of the rules, and
    callers send raw. Same end state on the row, enforced by the side that
    defines what canonical means.
    """
    state = _state(
        submitted="https://jobs.example.org/p/12",
        landed="https://jobs.example.org/p/12",
    )
    declared = "https://jobs.example.org/p/12?utm_source=rss&id=12"
    page = _FakePage({'link[rel~="canonical"]': {"href": declared}})

    _capture(state, page)

    # Verbatim — including the utm the api will strip on write.
    assert state.canonical_url == declared
    assert state.canonical_source == "link_rel"


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
        'link[rel~="canonical"]': {
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
        'link[rel~="canonical"]': {"href": "https://www.example-boards.com/jobs/5"},
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
    page = _FakePage({'link[rel~="canonical"]': {"href": "https://spa.example.io/"}})

    _capture(state, page)

    assert state.canonical_url == resolved
    assert state.canonical_source == "resolved"


def test_declared_canonical_cross_host_rejected():
    """A syndication pointer at another host is not this document's
    identity — we never fetched it and cannot verify it names this job."""
    resolved = "https://boards.example-ats.com/acme/jobs/42"
    state = _state(submitted=resolved, landed=resolved)
    page = _FakePage({
        'link[rel~="canonical"]': {
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
        'link[rel~="canonical"]': {
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
        'link[rel~="canonical"]': {
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
        'link[rel~="canonical"]': {"href": "javascript:void(0)"},
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
    lands and the graph still routes to DetectClosedState.

    Also pins the ordering from the other side: the canonical ladder runs
    ahead of the content reads, so a NON-driver throw there must not stop
    them.
    """
    resolved = "https://brittle.example.com/jobs/1"
    state = _state(submitted=resolved, landed=resolved)
    page = _FakePage({}, raise_on_query=True)

    nxt = _capture(state, page)

    assert isinstance(nxt, DetectClosedState)
    assert state.canonical_url == resolved
    assert state.canonical_source == "resolved"
    assert state.job_content  # the capture itself still happened
    assert page.read_body is True


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
        'link[rel~="canonical"]': {"href": "https://odd.example.com/req/551-std"},
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
        'link[rel~="canonical"]': {"href": "https://jobs.example.org/p/998"},
    })

    _capture(state, page)

    entry = [e for e in state.node_trace if e.node == "Capture"][-1]
    assert entry.payload["canonical_source"] == "link_rel"
    assert entry.payload["canonical_url"] == "https://jobs.example.org/p/998"


# ----------------------------------------------------------------------
# The gate runs PER RUNG — a junk rung must not shadow a good one
# ----------------------------------------------------------------------


def test_declared_canonical_junk_link_rel_falls_through_to_og_url():
    """LOAD-BEARING. The SPA shell hard-codes `<link rel=canonical
    href="https://site/">` on every page while still emitting a correct
    og:url — the exact anti-pattern this ticket exists for.

    With the junk filter applied once to the ladder's OUTPUT, the
    present-but-useless rung 1 short-circuits rungs 2 and 3 and the
    page's own good declaration is thrown away. The gate has to run
    INSIDE the ladder, per rung, and fall through on rejection.
    """
    resolved = "https://spa.example.io/careers/openings/9912?position=2"
    state = _state(submitted=resolved, landed=resolved)
    page = _FakePage({
        LINK_REL: {"href": "https://spa.example.io/"},                    # host_root
        OG_URL: {"content": "https://spa.example.io/careers/openings/9912"},
    })

    _capture(state, page)

    assert state.canonical_url == "https://spa.example.io/careers/openings/9912"
    assert state.canonical_source == "og_url"
    assert OG_URL in page.queried


def test_declared_canonical_profile_rung_reached_when_both_standard_tags_junk():
    """The escape hatch's OWN stated purpose. Rung 3 is documented as the
    rescue for a host that emits neither standard tag *or emits a wrong
    one* — and the wrong-one half is unreachable unless a rejected rung
    falls through. An operator configuring `canonical_link_selectors` for
    precisely this host would otherwise watch it never be consulted.
    """
    resolved = "https://walled.example.com/jobs/view/4404587081?trk=feed"
    state = _state(
        submitted=resolved,
        landed=resolved,
        profile={
            "extension_selectors": {"canonical_link_selectors": ["a.permalink"]},
        },
    )
    page = _FakePage({
        LINK_REL: {"href": "https://walled.example.com/login"},        # auth_path
        OG_URL: {"content": "https://aggregator.example/listing/1"},   # cross_host
        "a.permalink": {
            "href": "https://walled.example.com/jobs/view/4404587081"
        },
    })

    _capture(state, page)

    assert (
        state.canonical_url == "https://walled.example.com/jobs/view/4404587081"
    )
    assert state.canonical_source == "profile_selector"


def test_declared_canonical_every_rung_junk_keeps_resolved_and_traces_reasons():
    """All three rungs declare garbage: the resolved URL survives
    untouched, and the trace records WHY each candidate was thrown away.

    `canonical_source` alone reads "resolved" identically for "the page
    declared nothing" and "we rejected three declarations", so a host
    being systematically eaten by the `_AUTH_PATH_SEGMENTS` word list
    would otherwise be invisible in production — the one case you would
    actually go to the trace for.
    """
    resolved = "https://junk.example.com/jobs/view/77"
    state = _state(
        submitted=resolved,
        landed=resolved,
        profile={"extension_selectors": {"canonical_link_selectors": ["a.perma"]}},
    )
    page = _FakePage({
        LINK_REL: {"href": "https://junk.example.com/"},
        OG_URL: {"content": "https://elsewhere.example/jobs/view/77"},
        "a.perma": {"href": "javascript:void(0)"},
    })

    _capture(state, page)

    assert state.canonical_url == resolved
    assert state.canonical_source == "resolved"
    entry = [e for e in state.node_trace if e.node == "Capture"][-1]
    assert [
        (r["source"], r["reason"]) for r in entry.payload["canonical_rejected"]
    ] == [
        ("link_rel", "host_root"),
        ("og_url", "cross_host"),
        ("profile_selector", "bad_scheme"),
    ]


def test_declared_canonical_throwing_rung_does_not_blind_the_rungs_below():
    """One selector that throws is a bad selector, not a dead browser. A
    single try/except around the whole ladder makes rung 1 able to abort
    rungs 2 and 3; each rung reads inside its own.
    """
    resolved = "https://brittle.example.com/jobs/1?utm_source=x"
    state = _state(submitted=resolved, landed=resolved)
    page = _FakePage(
        {OG_URL: {"content": "https://brittle.example.com/jobs/1"}},
        raise_on={LINK_REL: "Protocol error: Node with given id not found"},
    )

    _capture(state, page)

    assert state.canonical_url == "https://brittle.example.com/jobs/1"
    assert state.canonical_source == "og_url"


def test_declared_canonical_link_rel_selector_is_a_token_match():
    """`<link rel="canonical alternate">` is legal and does appear on
    syndicated / i18n pages. `rel~=` matches one token of a
    space-separated list; `rel=` is an exact whole-value match that
    silently misses the multi-token form. Playwright's selector parser
    accepts `~=` (driver `utils/isomorphic/selectorParser.js` allows
    `=, *=, ^=, $=, |=, ~=`).

    Pinned on the WIRE selector: `_FakePage` is a dict lookup and cannot
    model CSS token matching itself.
    """
    resolved = "https://plain.example.com/jobs/1"
    state = _state(submitted=resolved, landed=resolved)
    page = _FakePage({})

    _capture(state, page)

    assert page.queried[0] == 'link[rel~="canonical"]'


# ----------------------------------------------------------------------
# Ordering — the ladder runs BEFORE the content reads (CC-160)
# ----------------------------------------------------------------------


def test_declared_canonical_driver_death_raises_before_any_bytes_are_banked():
    """LOAD-BEARING. `_adopt_declared_canonical` re-raises a
    driver-closed error, per the CC-160 convention for every except that
    wraps a Playwright call — so it must run BEFORE the content reads.

    Below the reads, that same raise discards a complete, usable
    `job_content` + `html` and buys a full re-scrape (the runner treats
    a raise out of the graph as infra-death: relaunch + re-queue hold).
    Capture had no such raise site before CC-248 —
    `_screenshot_and_upload` never raises by contract
    (`_artifacts.upload_page_screenshot`) and `_discover_selectors`
    swallows everything. Above the reads it costs nothing: the driver
    death that kills this query_selector kills `page.inner_text` on the
    next line anyway.
    """
    resolved = "https://dead.example.com/jobs/1"
    state = _state(submitted=resolved, landed=resolved)
    page = _FakePage({}, raise_on_query=True, query_error=DRIVER_DEAD)

    with pytest.raises(RuntimeError, match="Connection closed"):
        _capture(state, page)

    # Nothing was captured, so nothing was thrown away.
    assert page.read_body is False
    assert not state.job_content
    assert state.canonical_source == "resolved"


# ----------------------------------------------------------------------
# The persistence hop must not re-key a row we merely matched
# ----------------------------------------------------------------------


def test_declared_canonical_persist_hop_skipped_on_duplicate_job_post():
    """LOAD-BEARING. `outcome="duplicate"` from /persist-extraction/ means
    `state.job_post_id` is a PRE-EXISTING JobPost the api matched us onto.
    `JobPostExtractor` finds it with
    `filter(Q(link=link) | Q(canonical_link=canonical))` — no owner
    filter — so it is frequently another user's row, and the runner's
    staff `CC_API_TOKEN` sails straight through the owner-or-staff check
    on `PATCH /api/v1/job-posts/<id>/`.

    `canonical_link` is the PRIMARY DEDUPE KEY. Rewriting it on a row we
    merely matched makes that post unfindable at its own identity by
    `find_duplicate`'s canonical leg, by federation ingest's
    `filter(canonical_link=canonical)`, and by the serializer's sibling
    lookup — and starts federating a new value to instances that ingested
    the old one. The api's own duplicate branch merges empty fields only,
    and its dedup-verb field allowlist explicitly excludes
    "dedupe-pipeline columns (canonical_link, fingerprints)". This hop
    honours the same policy.
    """
    state = _extract_state("link_rel")
    state.was_duplicate = True

    with patch("scrape_graph.nodes_extract.httpx.patch") as mock_patch, \
            patch("scrape_graph.tracing._post_transition"):
        mock_patch.return_value.status_code = 200
        nxt = _run(ReviewCompleteness(), state)

    assert isinstance(nxt, UpdateProfile)
    assert mock_patch.call_count == 0
    entry = [
        e for e in state.node_trace if e.node == "ReviewCompleteness"
    ][-1]
    assert entry.payload["canonical_skipped"] == "duplicate_target"
    assert entry.payload["canonical_written"] is False


def test_declared_canonical_persist_hop_still_fires_on_a_non_duplicate():
    """The duplicate guard must narrow, not disable. A run that CREATED
    or upgraded the post still writes its declaration — that is the whole
    delivery mechanism for the ladder.
    """
    state = _extract_state("og_url")
    assert state.was_duplicate is False   # the dataclass default

    with patch("scrape_graph.nodes_extract.httpx.patch") as mock_patch, \
            patch("scrape_graph.tracing._post_transition"):
        mock_patch.return_value.status_code = 200
        _run(ReviewCompleteness(), state)

    assert mock_patch.call_count == 1
    entry = [
        e for e in state.node_trace if e.node == "ReviewCompleteness"
    ][-1]
    assert entry.payload["canonical_written"] is True
    assert "canonical_skipped" not in entry.payload


# ---------------------------------------------------------------------------
# CC-248 review follow-up: the duplicate guard is only as wide as the flag
# that feeds it.
#
# The two tests above set `state.was_duplicate` BY HAND, so they pin the guard
# in ReviewCompleteness but say nothing about what actually sets that flag.
# PersistJobPost does, from the api's `meta.outcome` string — and the api
# reaches a duplicate two ways under two names:
#
#   "duplicate"                 lib/parsers/job_post_extractor.py:937
#                               link / canonical_link hit
#   "duplicate_via_fingerprint" lib/parsers/job_post_extractor.py:1009
#                               title+company hit, from a bare
#                               JobPost.objects.get_or_create(title=, company=)
#                               with NO owner filter, merging empty fields only
#
# An `== "duplicate"` test passes the fingerprint match straight through the
# guard. That is not a corner: the fingerprint branch is reached precisely
# when the link leg MISSES, i.e. when the stored link differs from the scraped
# one — which is the exact situation a declared canonical exists to describe.
# So the one outcome the guard most needed to catch was the one it dropped,
# and the canonical_link of a pre-existing, possibly another user's, row would
# be rewritten by the runner's staff token.
# ---------------------------------------------------------------------------

def _persist_state() -> ScrapeGraphState:
    state = ScrapeGraphState(
        scrape_id=SCRAPE_ID,
        submitted_url="https://jobs.example.org/p/12?position=2",
    )
    state.parsed = {"title": "Staff Engineer"}
    return state


@pytest.mark.parametrize(
    "outcome, expected",
    [
        ("duplicate", True),
        ("duplicate_via_fingerprint", True),   # the branch the guard dropped
        ("created", False),
        ("updated_stub", False),
        ("force_updated", False),
    ],
)
def test_declared_canonical_duplicate_flag_covers_both_api_outcomes(
    outcome, expected,
):
    """`was_duplicate` must be true for EVERY outcome that means "the api
    pointed us at a row it did not create for us".
    """
    state = _persist_state()

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"meta": {"job_post_id": JOB_POST_ID, "outcome": outcome}}

    with patch("scrape_graph.nodes_extract.httpx.post", return_value=_Resp()), \
            patch("scrape_graph.tracing._post_transition"):
        _run(PersistJobPost(), state)

    assert state.job_post_id == JOB_POST_ID
    assert state.was_duplicate is expected


def test_declared_canonical_not_rekeyed_on_a_fingerprint_duplicate():
    """End to end for the branch the first fix missed: a fingerprint duplicate
    must reach ReviewCompleteness with the guard armed, so no PATCH is sent.

    This is the test that bites. With `== "duplicate"` the flag comes back
    False here and the hop PATCHes canonical_link onto a row the api merely
    matched us onto.
    """
    state = _persist_state()
    state.canonical_url = "https://jobs.example.org/p/12"
    state.canonical_source = "link_rel"

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {
                "meta": {
                    "job_post_id": JOB_POST_ID,
                    "outcome": "duplicate_via_fingerprint",
                },
            }

    with patch("scrape_graph.nodes_extract.httpx.post", return_value=_Resp()), \
            patch("scrape_graph.tracing._post_transition"):
        _run(PersistJobPost(), state)

    assert state.was_duplicate is True

    with patch("scrape_graph.nodes_extract.httpx.patch") as mock_patch, \
            patch("scrape_graph.tracing._post_transition"):
        mock_patch.return_value.status_code = 200
        _run(ReviewCompleteness(), state)

    assert mock_patch.call_count == 0
    entry = [
        e for e in state.node_trace if e.node == "ReviewCompleteness"
    ][-1]
    assert entry.payload["canonical_skipped"] == "duplicate_target"
    assert entry.payload["canonical_written"] is False
