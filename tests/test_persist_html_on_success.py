"""Tests for PersistScrape persisting the captured DOM (`html`).

PACA #30 — raw scrape HTML must stay inspectable on successful browser
scrapes so `inspect_scrape_html` / `find_selectors_for_text` (and #26's
readiness live-match) have something to read.

Two guarantees this node now owns:

1. **Never clobber.** When `state.html` is empty, `html` is OMITTED from
   the PATCH rather than sent as null. A null would overwrite a DOM
   persisted on an earlier run (lease-sweep re-dispatch) or by
   `_artifacts.capture_debug_artifact`'s Fail-path backfill. Composes
   with the api anti-clobber guard (api PR #190 drops a falsy `html`)
   without relying on it.
2. **Write it at least once.** Capture sets `state.html` from
   `page.content()`, but that call can fail (frame detached, execution
   context destroyed mid-navigation) while the `inner_text("body")` read
   just before it succeeds. The Fail path re-grabs the DOM; the success
   path did not, so a hiccuped Capture meant the DOM was lost forever.
   PersistScrape now re-grabs from the still-live page when `state.html`
   is empty, giving the success path the same recovery.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from scrape_graph._artifacts import _DOM_TRUNCATION_TRAILER, _MAX_DOM_BYTES
from scrape_graph.nodes_extract import StartExtract
from scrape_graph.nodes_scrape import PersistScrape
from scrape_graph.state import ScrapeGraphState


def _run(node, state: ScrapeGraphState):
    ctx = SimpleNamespace(state=state)
    return asyncio.run(node.run(ctx))


class _FakePage:
    """Minimal stand-in for the live Playwright page on `state`."""

    def __init__(self, content_html: str = "<html><body>dom</body></html>",
                 fail_content: bool = False):
        self._html = content_html
        self._fail_content = fail_content
        self.content_calls = 0

    async def content(self) -> str:
        self.content_calls += 1
        if self._fail_content:
            raise RuntimeError("frame detached")
        return self._html


def _state(*, html=None, page=None) -> ScrapeGraphState:
    state = ScrapeGraphState(scrape_id=7, submitted_url="https://example.com/jobs/7")
    state.job_content = "Senior Engineer — full job description body text"
    state.html = html
    if page is not None:
        # Same attribute the runner attaches via run_scrape_graph.
        state._browser_page = page  # type: ignore[attr-defined]
    return state


def _patched_attributes(mock_patch) -> dict:
    assert mock_patch.called, "PersistScrape must PATCH the scrape row"
    return mock_patch.call_args.kwargs["json"]["data"]["attributes"]


def test_captured_html_reaches_persist_and_is_sent():
    """A successful browser scrape: the DOM captured into state.html by
    Capture is threaded to PersistScrape and PATCHed as a non-empty html."""
    state = _state(html="<html><body>greenhouse SSR</body></html>")

    with patch("scrape_graph.nodes_scrape.httpx.patch") as mock_patch:
        next_node = _run(PersistScrape(), state)

    attrs = _patched_attributes(mock_patch)
    assert attrs["html"] == "<html><body>greenhouse SSR</body></html>"
    assert attrs["job_content"]  # job_content still delivered alongside
    assert isinstance(next_node, StartExtract)


def test_empty_html_omitted_not_sent_as_null():
    """No captured DOM and no live page → html is OMITTED from the PATCH,
    never sent as null, so a previously stored DOM survives."""
    state = _state(html="")  # falsy, no _browser_page attached

    with patch("scrape_graph.nodes_scrape.httpx.patch") as mock_patch:
        next_node = _run(PersistScrape(), state)

    attrs = _patched_attributes(mock_patch)
    assert "html" not in attrs, "empty html must be omitted, not sent as null"
    assert attrs["job_content"]  # job_content is still PATCHed
    assert isinstance(next_node, StartExtract)


def test_html_regrabbed_from_live_page_when_empty():
    """Capture's content() hiccuped (state.html empty) but the page is
    still live → PersistScrape re-grabs the DOM so the success path
    persists a non-empty html at least once."""
    page = _FakePage(content_html="<html><body>re-grabbed DOM</body></html>")
    state = _state(html=None, page=page)

    with patch("scrape_graph.nodes_scrape.httpx.patch") as mock_patch:
        next_node = _run(PersistScrape(), state)

    attrs = _patched_attributes(mock_patch)
    assert page.content_calls == 1
    assert attrs["html"] == "<html><body>re-grabbed DOM</body></html>"
    assert state.html == "<html><body>re-grabbed DOM</body></html>"
    assert isinstance(next_node, StartExtract)


def test_no_regrab_when_html_already_present():
    """When Capture already populated state.html, PersistScrape sends it
    as-is and does NOT make a redundant page.content() call."""
    page = _FakePage(content_html="<should-not-be-used/>")
    state = _state(html="<html>captured-in-capture</html>", page=page)

    with patch("scrape_graph.nodes_scrape.httpx.patch") as mock_patch:
        _run(PersistScrape(), state)

    attrs = _patched_attributes(mock_patch)
    assert page.content_calls == 0, "must not re-grab when html already captured"
    assert attrs["html"] == "<html>captured-in-capture</html>"


def test_regrab_failure_omits_html_and_does_not_raise():
    """If the re-grab itself fails (detached page), html is omitted rather
    than sent as null; the node never raises and still routes onward."""
    page = _FakePage(fail_content=True)
    state = _state(html=None, page=page)

    with patch("scrape_graph.nodes_scrape.httpx.patch") as mock_patch:
        next_node = _run(PersistScrape(), state)

    attrs = _patched_attributes(mock_patch)
    assert page.content_calls == 1
    assert "html" not in attrs
    assert isinstance(next_node, StartExtract)


def test_oversized_html_capped_at_max_dom_bytes():
    """A captured DOM larger than the cap is truncated before the PATCH —
    a multi-MB LinkedIn DOM must not be sent uncapped on the success path.
    The PATCHed html is at most the cap plus the truncation trailer, and
    the trailer is present so consumers can tell it was cut."""
    big = "<html><body>" + ("x" * (_MAX_DOM_BYTES * 2)) + "</body></html>"
    assert len(big) > _MAX_DOM_BYTES  # guard: the input really is oversized
    state = _state(html=big)

    with patch("scrape_graph.nodes_scrape.httpx.patch") as mock_patch:
        next_node = _run(PersistScrape(), state)

    attrs = _patched_attributes(mock_patch)
    sent = attrs["html"]
    assert len(sent) == _MAX_DOM_BYTES + len(_DOM_TRUNCATION_TRAILER)
    assert sent.endswith(_DOM_TRUNCATION_TRAILER)
    assert sent[:_MAX_DOM_BYTES] == big[:_MAX_DOM_BYTES]
    assert isinstance(next_node, StartExtract)


def test_small_html_passes_through_uncapped():
    """A DOM at or under the cap is sent verbatim — no trailer, no
    truncation. The cap only touches oversized DOMs."""
    small = "<html><body>greenhouse SSR</body></html>"
    assert len(small) <= _MAX_DOM_BYTES
    state = _state(html=small)

    with patch("scrape_graph.nodes_scrape.httpx.patch") as mock_patch:
        _run(PersistScrape(), state)

    attrs = _patched_attributes(mock_patch)
    assert attrs["html"] == small
    assert _DOM_TRUNCATION_TRAILER not in attrs["html"]
