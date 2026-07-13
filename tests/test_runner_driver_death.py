"""CC-141 regression — a dead resident browser driver must NOT drain the
hold queue into ``failed`` rows.

The resident Camoufox/Playwright browser can die mid-run; the driver
connection then closes. After that every claimed scrape insta-fails at
``open_tab()`` with "Page.evaluate: Connection closed while reading from the
driver", and the pre-fix runner PATCHed each one to ``failed`` in ~0.3s —
alive to the queue, dead to the browser (134 such failures / 24h in prod,
2026-07-13).

The fix (all three legs):
  1. ``ResidentBrowser.open_tab`` detects the driver-closed condition,
     relaunches the context+anchor ONCE, and retries. Only if the retry
     still hits a driver-closed error does it raise ``ResidentDriverDead``.
  2. ``_run_graph`` treats ResidentDriverDead / any driver-closed error as
     infra-death: re-queue the scrape as ``hold`` (tagged ``[driver-death]``),
     never ``failed``, and raise ``DriverDeath``.
  3. ``_run_poll_loop`` counts consecutive DriverDeaths and, past a small
     threshold, backs off hard with a loud ERROR instead of claim-looping.

These tests fail against the pre-fix code (which had no relaunch, no
ResidentDriverDead, and PATCHed driver deaths straight to ``failed``).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

import runners.scrape_runner as runner
from browser.resident import (
    ResidentBrowser,
    ResidentDriverDead,
    is_driver_closed,
)

DRIVER_CLOSED_MSG = (
    "Page.evaluate: Connection closed while reading from the driver"
)


def _driver_closed_exc() -> Exception:
    return Exception(DRIVER_CLOSED_MSG)


class TestIsDriverClosed:
    def test_matches_evaluate_screenshot_cookies_messages(self):
        for call in ("Page.evaluate", "Page.screenshot", "browser.cookies"):
            exc = Exception(f"{call}: Connection closed while reading from the driver")
            assert is_driver_closed(exc)

    def test_does_not_match_ordinary_failure(self):
        assert not is_driver_closed(Exception("ready_selector timed out after 30s"))
        assert not is_driver_closed(ValueError("bad selector"))


class _FakeContext:
    """Minimal Playwright BrowserContext stand-in for open_tab."""

    def __init__(self, anchor):
        self._anchor = anchor
        self.closed = False

    async def new_page(self):
        return self._anchor

    async def add_cookies(self, cookies):
        return None

    async def close(self):
        self.closed = True

    def expect_page(self):
        # Playwright's expect_page() is an async CM yielding an info object
        # whose `.value` awaitable resolves to the spawned page. The anchor's
        # evaluate() (called inside the `async with`) is what raises when the
        # driver is dead, so this CM just wires up the page hand-back.
        return _CtxProxy(MagicMock(name="spawned_page"))


class _CtxProxy:
    def __init__(self, page):
        self._page = page

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    @property
    def value(self):
        page = self._page

        async def _get():
            return page

        return _get()


class _FakeAnchor:
    """Anchor page whose ``evaluate`` raises driver-closed for the first N
    calls, then succeeds — models a dead driver that a relaunch revives (or
    doesn't, when deaths=999).
    """

    def __init__(self, deaths: int):
        self._deaths = deaths
        self.evaluate_calls = 0

    async def goto(self, *a, **kw):
        return None

    async def evaluate(self, *a, **kw):
        self.evaluate_calls += 1
        if self.evaluate_calls <= self._deaths:
            raise _driver_closed_exc()
        return None


class _FakeBrowser:
    """Browser stand-in that hands out a fresh context per new_context().

    ``contexts_before_recovery`` contexts get a dead anchor; contexts after
    that get a live anchor (models the relaunch rebuilding onto a browser
    that is itself still alive).
    """

    def __init__(self, deaths_per_anchor: int, recover_after: int):
        self._deaths_per_anchor = deaths_per_anchor
        self._recover_after = recover_after
        self.new_context_calls = 0
        self.anchors: list[_FakeAnchor] = []

    async def new_context(self):
        self.new_context_calls += 1
        deaths = 0 if self.new_context_calls > self._recover_after else self._deaths_per_anchor
        anchor = _FakeAnchor(deaths=deaths)
        self.anchors.append(anchor)
        ctx = _FakeContext(anchor)
        return ctx


class TestResidentOpenTabRelaunch:
    @pytest.mark.asyncio
    async def test_relaunch_and_succeed_on_driver_death(self, monkeypatch):
        # First context's anchor dies once (the initial evaluate); relaunch
        # rebuilds onto a live anchor and the retry succeeds.
        browser = _FakeBrowser(deaths_per_anchor=99, recover_after=1)
        rb = ResidentBrowser(browser)

        page = await rb.open_tab(domain="example.com")

        assert page is not None
        # Relaunched exactly once: two contexts built (dead, then live).
        assert browser.new_context_calls == 2

    @pytest.mark.asyncio
    async def test_raises_resident_driver_dead_when_relaunch_also_dies(self):
        # Every context's anchor is dead — the browser process itself is gone.
        browser = _FakeBrowser(deaths_per_anchor=99, recover_after=999)
        rb = ResidentBrowser(browser)

        with pytest.raises(ResidentDriverDead):
            await rb.open_tab(domain="example.com")

        # One relaunch attempted (2 contexts) before giving up — bounded, not
        # an infinite relaunch loop.
        assert browser.new_context_calls == 2

    @pytest.mark.asyncio
    async def test_non_driver_error_is_not_swallowed(self):
        browser = _FakeBrowser(deaths_per_anchor=0, recover_after=0)
        rb = ResidentBrowser(browser)

        async def boom(*a, **kw):
            raise ValueError("not a driver problem")

        # Corrupt the anchor after context build so evaluate raises a
        # non-driver error; it must propagate unchanged (no relaunch).
        await rb._ensure_context()
        rb._anchor.evaluate = boom
        with pytest.raises(ValueError):
            await rb.open_tab(domain="example.com")


class TestRunGraphDriverDeath:
    @pytest.mark.asyncio
    async def test_driver_death_requeues_hold_not_failed(self, monkeypatch):
        patched: list[dict] = []

        async def fake_update(api, scrape_id, **kw):
            patched.append({"id": scrape_id, **kw})
            return "data:\n  ok: true"

        monkeypatch.setattr(runner, "update_scrape", fake_update)

        # Resident whose open_tab reports the driver is dead-after-relaunch.
        resident = MagicMock()
        resident.open_tab = AsyncMock(side_effect=ResidentDriverDead("driver dead"))
        resident.close_tab = AsyncMock()
        resident.save_sessions = AsyncMock(return_value=0)
        monkeypatch.setattr(runner, "_RESIDENT", resident)

        with pytest.raises(runner.DriverDeath):
            await runner._run_graph(
                MagicMock(), "V1StGXR8_Z", "https://example.com/job/1",
                "example.com", None,
            )

        # The scrape was re-queued as hold, NEVER failed. This is the crux of
        # CC-141: a dead driver must not drain the queue into failed rows.
        statuses = [p.get("status") for p in patched]
        assert "hold" in statuses
        assert "failed" not in statuses
        hold_patch = next(p for p in patched if p.get("status") == "hold")
        assert "[driver-death]" in hold_patch.get("note", "")

    @pytest.mark.asyncio
    async def test_content_failure_still_marks_failed(self, monkeypatch):
        patched: list[dict] = []

        async def fake_update(api, scrape_id, **kw):
            patched.append({"id": scrape_id, **kw})
            return "data:\n  ok: true"

        monkeypatch.setattr(runner, "update_scrape", fake_update)

        # A real (non-driver) exception must still fail the scrape.
        resident = MagicMock()
        resident.open_tab = AsyncMock(side_effect=RuntimeError("selector blew up"))
        resident.close_tab = AsyncMock()
        resident.save_sessions = AsyncMock(return_value=0)
        monkeypatch.setattr(runner, "_RESIDENT", resident)

        result = await runner._run_graph(
            MagicMock(), "V1StGXR8_Z", "https://example.com/job/1",
            "example.com", None,
        )

        assert result is False
        assert "failed" in [p.get("status") for p in patched]


class TestPollLoopBackoff:
    @pytest.mark.asyncio
    async def test_backs_off_after_consecutive_driver_deaths(self, monkeypatch):
        sleeps: list[float] = []

        async def fake_sleep(secs):
            sleeps.append(secs)

        # poll_once always signals driver death; loop should escalate to a
        # long backoff sleep once past the threshold, then we stop it.
        calls = {"n": 0}

        async def fake_poll_once(api):
            calls["n"] += 1
            raise runner.DriverDeath("browser dead")

        def running_flag():
            # Run just past the threshold so the backoff branch fires once.
            return calls["n"] < runner._DRIVER_DEATH_BACKOFF_THRESHOLD

        monkeypatch.setattr(runner.asyncio, "sleep", fake_sleep)
        monkeypatch.setattr(runner, "poll_once", fake_poll_once)

        errors: list[str] = []
        monkeypatch.setattr(
            runner.logger, "error",
            lambda msg, *a, **kw: errors.append(msg % a if a else msg),
        )

        await runner._run_poll_loop(MagicMock(), running_flag)

        # The final cycle crossed the threshold → a long backoff sleep was
        # scheduled (POLL_INTERVAL + backoff), and a loud ERROR was logged.
        assert any(s > runner.POLL_INTERVAL for s in sleeps)
        assert any("backing off" in e.lower() for e in errors)
