"""Tests for ResolveFinalUrl's networkidle re-read of page.url.

Repro: scrape 394 = jp 1908 (2026-05-08, prod attended Camoufox).
Submitted URL was a ZipRecruiter /km/<opaque-token> email-tracker URL.
Graph trace showed ResolveFinalUrl with `duration_ms=0
did_redirect=false canonical_url=/km/<opaque>` — the browser navigated
but the canonical_url stayed pinned to the tracker form, so dedupe
against the canonical /jobs/altus-llc/... rows could never fire.

Root cause: Navigate uses `page.goto(wait_until="domcontentloaded")`
and captures `state.final_url = page.url` immediately. Server-side
301/302 redirects are followed by Playwright before domcontentloaded,
so those work. But meta-refresh + JS `window.location` redirects
execute AFTER domcontentloaded — Navigate already returned, and
state.final_url is frozen at the tracker URL. ResolveFinalUrl then
reads state.final_url without re-checking page.url and short-circuits
when submitted == landed.

The fix (nodes_scrape.py ResolveFinalUrl): before _resolve_final_url_body,
await page.wait_for_load_state("networkidle", timeout=5_000) and
re-read page.url into state.final_url. Best-effort try/except so a
slow tracker can't park ResolveFinalUrl past the 15s outer budget.
Navigate's domcontentloaded choice stays unchanged — the comment there
documents that wait_until="load" can deadlock on LinkedIn /comm/
auth-interstitials.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from scrape_graph.nodes_scrape import ResolveFinalUrl, CheckLinkDedup
from scrape_graph.state import ScrapeGraphState

# Real NanoID shapes (CC-77). Ids are 10-char strings, not ints — the swap
# is `state.scrape_id = new_id` (passthrough). int(new_id) would raise on a
# real NanoID and the swap would silently fail. The prior numeric-string
# fixtures ("394"/"395"/"476") + `== 395` int assertions false-greened the
# broken int() cast.
PARENT_A = "Zc3p_QeR9k"   # parent scrape id (was 394)
CHILD_A = "Mn4Lp2Qr_T"    # child scrape id from redirect (was "395")
PARENT_B = "Wq8sR3tUvX"   # parent scrape id (was 475)
CHILD_B = "Yh1Zk5LmNp"    # child scrape id from redirect (was "476")


def _run(node, state: ScrapeGraphState):
    ctx = SimpleNamespace(state=state)
    return asyncio.run(node.run(ctx))


class _FakeResp:
    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = ""

    def json(self):
        return self._payload


class _FakePage:
    """Minimal Playwright Page stand-in.

    `url` returns the page URL — by mutating `_url` between
    wait_for_load_state and the read, we simulate a JS / meta-refresh
    redirect that resolves only after networkidle.
    """

    def __init__(self, url_at_domcontentloaded: str, url_at_networkidle: str):
        self._url = url_at_domcontentloaded
        self._post_idle_url = url_at_networkidle
        self.wait_calls: list[tuple[str, int | None]] = []

    @property
    def url(self) -> str:
        return self._url

    async def wait_for_load_state(self, state: str, timeout: int | None = None):
        self.wait_calls.append((state, timeout))
        # The redirect happens during the wait — flip _url so the next
        # `.url` read reflects post-redirect canonical.
        self._url = self._post_idle_url


def _state_for_js_redirect(submitted: str, landed_after_idle: str) -> ScrapeGraphState:
    """Build a state where Navigate captured the tracker URL but a
    later networkidle wait will see the redirected canonical URL.
    """
    state = ScrapeGraphState(scrape_id=PARENT_A, submitted_url=submitted)
    state.final_url = submitted  # Navigate's capture, frozen pre-redirect
    page = _FakePage(
        url_at_domcontentloaded=submitted,
        url_at_networkidle=landed_after_idle,
    )
    # _browser_page is not a declared field — runtime attribute.
    state._browser_page = page  # type: ignore[attr-defined]
    return state


def test_networkidle_reread_picks_up_js_redirect():
    """ResolveFinalUrl must wait for networkidle, re-read page.url, and
    treat the post-redirect URL as the landed URL — driving did_redirect=True
    and the parent terminal-close + child-scrape handoff."""
    submitted = (
        "https://www.ziprecruiter.com/km/AAHQDn_YYvnvafLIF4GPjtD94u5MyqYcZ2T"
    )
    landed = "https://www.ziprecruiter.com/jobs/altus-llc/software-developer-c-remote"
    state = _state_for_js_redirect(submitted, landed)

    patches: list[dict] = []

    def fake_post(url, json=None, headers=None, timeout=None):
        return _FakeResp(201, {"data": {"id": CHILD_A}})

    def fake_patch_status(scrape_id, status, note=None):
        patches.append({"scrape_id": scrape_id, "status": status, "note": note})

    with patch("scrape_graph.nodes_scrape.httpx.post", side_effect=fake_post), \
         patch("scrape_graph.nodes_scrape._patch_scrape_status", side_effect=fake_patch_status):
        next_node = _run(ResolveFinalUrl(), state)

    assert isinstance(next_node, CheckLinkDedup)
    page = state._browser_page  # type: ignore[attr-defined]
    assert page.wait_calls == [("networkidle", 5_000)], (
        "must call wait_for_load_state with networkidle + 5s budget"
    )
    assert state.final_url == landed, (
        "page.url re-read after networkidle must replace Navigate's stale capture"
    )
    assert state.did_redirect is True, (
        "post-redirect URL differs from submitted — should detect the redirect"
    )
    assert state.scrape_id == CHILD_A, "child scrape id should be swapped in"
    assert isinstance(state.scrape_id, str)
    assert len(patches) == 1, "parent scrape should be terminal-closed exactly once"
    assert patches[0]["scrape_id"] == PARENT_A
    assert patches[0]["status"] == "completed"


def test_no_browser_page_falls_back_to_navigate_capture():
    """When state has no _browser_page (e.g. text-only paste flow),
    ResolveFinalUrl must not crash on the networkidle re-read — it
    falls through to whatever final_url Navigate captured."""
    same = "https://example.com/jobs/42"
    state = ScrapeGraphState(scrape_id=PARENT_A, submitted_url=same)
    state.final_url = same
    # No state._browser_page set.

    patches: list[dict] = []

    def fake_patch_status(scrape_id, status, note=None):
        patches.append({"scrape_id": scrape_id, "status": status, "note": note})

    with patch("scrape_graph.nodes_scrape._patch_scrape_status", side_effect=fake_patch_status):
        next_node = _run(ResolveFinalUrl(), state)

    assert isinstance(next_node, CheckLinkDedup)
    assert state.did_redirect is False
    assert patches == []


def test_networkidle_timeout_does_not_break_node():
    """If wait_for_load_state raises (real-world: a tracker host that
    never goes idle), swallow it and continue with whatever final_url
    Navigate captured. The 5s budget keeps this from eating the outer
    15s _RESOLVE_FINAL_URL_BUDGET_S."""
    submitted = "https://example.com/track/abc"
    state = ScrapeGraphState(scrape_id=PARENT_A, submitted_url=submitted)
    state.final_url = submitted

    class _HangingPage:
        url = submitted

        async def wait_for_load_state(self, state_name, timeout=None):
            raise asyncio.TimeoutError()

    state._browser_page = _HangingPage()  # type: ignore[attr-defined]

    patches: list[dict] = []

    def fake_patch_status(scrape_id, status, note=None):
        patches.append({"scrape_id": scrape_id, "status": status, "note": note})

    with patch("scrape_graph.nodes_scrape._patch_scrape_status", side_effect=fake_patch_status):
        next_node = _run(ResolveFinalUrl(), state)

    assert isinstance(next_node, CheckLinkDedup)
    assert state.final_url == submitted, "frozen Navigate capture preserved on timeout"
    assert state.did_redirect is False
    assert patches == []


def test_networkidle_timeout_still_rereads_page_url_when_url_moved_on():
    """page.url is the source of truth for URL after navigation; the
    network-idle signal is a separate concern. Tracker-heavy SPAs keep
    the network perpetually busy via heartbeat / telemetry beacons, so
    `wait_for_load_state("networkidle")` routinely times out even when
    the URL has already settled on the post-redirect destination. The
    old code put `state.final_url = page.url` *inside* the same try
    as the wait, so a timeout silently skipped the URL re-read and
    left state.final_url at whatever Navigate captured at
    domcontentloaded time (i.e. pre-redirect).

    Surfaced in production by scrape 475 = jp 715 (2026-05-28): a
    LinkedIn login wrapper that ObstacleRememberMe successfully
    authenticated through — the post-login screenshot at Capture-time
    proved the browser was on the real /jobs/view/<id> page — but
    state.final_url stayed pinned to the /uas/login wrapper, so
    CheckLinkDedup compared the wrong URL and missed the obvious
    dup against jp 2963. The fix (split the try/except so the URL
    re-read runs *unconditionally*) is generic: every chatty SPA
    benefits, not just LinkedIn.
    """
    submitted = "https://tracker.example.test/wrap?dest=%2Fjobs%2F42"
    landed = "https://content.example.test/jobs/42"
    state = ScrapeGraphState(scrape_id=PARENT_B, submitted_url=submitted)
    state.final_url = submitted  # Navigate's pre-redirect capture

    class _PageUrlMovedNetworkNeverIdle:
        """Browser whose page.url has navigated past the submitted URL
        but whose network never goes idle (trackers / beacons keep
        firing). Mirrors the failure mode on any heartbeat-heavy SPA."""

        def __init__(self):
            self._url = landed

        @property
        def url(self) -> str:
            return self._url

        async def wait_for_load_state(self, state_name, timeout=None):
            raise asyncio.TimeoutError()

    state._browser_page = _PageUrlMovedNetworkNeverIdle()  # type: ignore[attr-defined]

    patches: list[dict] = []

    def fake_post(url, json=None, headers=None, timeout=None):
        return _FakeResp(201, {"data": {"id": CHILD_B}})

    def fake_patch_status(scrape_id, status, note=None):
        patches.append({"scrape_id": scrape_id, "status": status, "note": note})

    with patch("scrape_graph.nodes_scrape.httpx.post", side_effect=fake_post), \
         patch("scrape_graph.nodes_scrape._patch_scrape_status", side_effect=fake_patch_status):
        next_node = _run(ResolveFinalUrl(), state)

    assert isinstance(next_node, CheckLinkDedup)
    assert state.final_url == landed, (
        "page.url must be re-read even when networkidle times out — "
        "the URL is the source of truth, not the network-idle signal"
    )
    assert state.did_redirect is True, (
        "post-login URL differs from submitted login wrapper — should "
        "detect the redirect and hand off to a child scrape"
    )
    assert state.scrape_id == CHILD_B, "child scrape id should be swapped in"
    assert isinstance(state.scrape_id, str)
    assert any(
        p["status"] == "completed" and f"redirected to scrape {CHILD_B}" in (p.get("note") or "")
        for p in patches
    ), "parent must be terminal-closed with the redirect note"
