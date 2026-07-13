"""CC-160 regression — a driver death that happens MID-scrape (after
``open_tab`` succeeded) must NOT be swallowed by the browser-tier nodes as a
per-selector "not matched" and marched over a dead page to ``failed``.

CC-141 hardened the driver-death seam at CLAIM time (``open_tab``). But a
Camoufox/Playwright driver that dies ~7s later (live prod scrape ddwm38K5a0,
2026-07-13: driver alive through Navigate + ResolveFinalUrl, then every one of
19 selectors errored ``Locator.wait_for: Connection closed while reading from
the driver`` in ~1ms each) slipped past it: ``WaitReadySelector`` caught each
per-selector exception into ``attempts[]`` and continued, ``Capture`` grabbed 0
bytes, and the run terminated at ``ExtractFail`` → the row was marked
``failed`` and the ExtractFail screenshot fired against a corpse
(``screenshot_uploaded: false``).

The fix (CC-160): the browser-tier nodes re-raise on ``is_driver_closed(exc)``
instead of swallowing it. The exception propagates out of ``run_scrape_graph``;
the runner's ``_run_graph`` already treats any driver-closed error as
infra-death — relaunch (CC-141 ``open_tab``) + re-queue the scrape as ``hold``
(never ``failed``) — so the eventual real failure happens on a live,
re-navigated page where the screenshot invariant can actually fire.

These tests fail against the pre-CC-160 code (which recorded the driver-closed
error as one more "not matched" and routed onward to SettleWait).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

import runners.scrape_runner as runner
from scrape_graph.nodes_scrape import (
    Capture,
    SettleWait,
    WaitReadySelector,
    _reraise_if_driver_closed,
)
from scrape_graph.state import ScrapeGraphState

DRIVER_CLOSED_MSG = (
    "Locator.wait_for: Connection closed while reading from the driver"
)


def _driver_closed_exc() -> Exception:
    return Exception(DRIVER_CLOSED_MSG)


class _FakeLocator:
    """``page.locator(sel).first`` stand-in. ``wait_for`` either raises the
    driver-closed error (mid-scrape death) or a plain timeout (normal miss)."""

    def __init__(self, exc: Exception):
        self._exc = exc

    @property
    def first(self):
        return self

    async def wait_for(self, *a, **kw):
        raise self._exc


class _FakePage:
    """Persistent work-page whose selector waits raise ``exc``. Models a page
    the graph has been handed after a successful open_tab."""

    def __init__(self, exc: Exception):
        self._exc = exc

    def locator(self, sel):
        return _FakeLocator(self._exc)

    async def inner_text(self, sel):
        raise self._exc

    async def content(self):
        raise self._exc


def _state_with_page(page) -> ScrapeGraphState:
    state = ScrapeGraphState(
        scrape_id="V1StGXR8_Z",
        submitted_url="https://example.com/job/1",
        original_scrape_id="V1StGXR8_Z",
        profile={"ready_selector": [".job-title", ".job-desc", "h1"]},
        source="poller",
    )
    state._browser_page = page  # type: ignore[attr-defined]
    return state


class _Ctx:
    def __init__(self, state):
        self.state = state


class TestReraiseHelper:
    def test_reraises_driver_closed(self):
        with pytest.raises(Exception, match="Connection closed"):
            _reraise_if_driver_closed(_driver_closed_exc())

    def test_swallows_ordinary_error(self):
        # No raise — a normal miss / timeout stays a benign, swallowed error.
        _reraise_if_driver_closed(Exception("ready_selector timed out after 30s"))
        _reraise_if_driver_closed(ValueError("bad selector"))


class TestWaitReadySelectorDriverDeath:
    @pytest.mark.asyncio
    async def test_driver_closed_propagates_not_swallowed(self, monkeypatch):
        # trace_node would post to the api; a driver death re-raises BEFORE
        # any trace, but patch it so the test never touches the network.
        monkeypatch.setattr(
            "scrape_graph.nodes_scrape.trace_node", lambda *a, **kw: None
        )
        state = _state_with_page(_FakePage(_driver_closed_exc()))

        # The crux: the node must RAISE the driver-closed error, not return
        # SettleWait after recording it as "not matched".
        with pytest.raises(Exception, match="Connection closed"):
            await WaitReadySelector().run(_Ctx(state))

    @pytest.mark.asyncio
    async def test_normal_timeout_still_routes_to_settlewait(self, monkeypatch):
        monkeypatch.setattr(
            "scrape_graph.nodes_scrape.trace_node", lambda *a, **kw: None
        )
        # A genuine "no selector matched" (plain timeout, no driver marker)
        # must NOT be treated as driver-death — it routes onward exactly as
        # today so Capture/tiers/ExtractFail can screenshot a live page.
        timeout_exc = Exception("Locator.wait_for: Timeout 500ms exceeded")
        state = _state_with_page(_FakePage(timeout_exc))

        result = await WaitReadySelector().run(_Ctx(state))

        assert isinstance(result, SettleWait)


class TestCaptureDriverDeath:
    @pytest.mark.asyncio
    async def test_driver_closed_during_capture_propagates(self, monkeypatch):
        # Capture reads inner_text + content; a mid-scrape death there must
        # propagate rather than persist a 0-byte capture and march on.
        monkeypatch.setattr(
            "scrape_graph.nodes_scrape.trace_node", lambda *a, **kw: None
        )
        state = _state_with_page(_FakePage(_driver_closed_exc()))

        with pytest.raises(Exception, match="Connection closed"):
            await Capture().run(_Ctx(state))


class TestRunGraphMidScrapeDriverDeath:
    @pytest.mark.asyncio
    async def test_mid_scrape_driver_death_requeues_hold_not_failed(
        self, monkeypatch
    ):
        """open_tab SUCCEEDS; the driver dies later INSIDE the graph run. The
        runner's _run_graph must catch the propagated driver-closed error and
        re-queue hold (not failed) — the CC-160 end-to-end path."""
        patched: list[dict] = []

        async def fake_update(api, scrape_id, **kw):
            patched.append({"id": scrape_id, **kw})
            return "data:\n  ok: true"

        monkeypatch.setattr(runner, "update_scrape", fake_update)

        # Resident whose open_tab succeeds (returns a page) — the death is
        # NOT at claim time. The graph run then raises driver-closed.
        resident = MagicMock()
        resident.open_tab = AsyncMock(return_value=MagicMock())
        resident.close_tab = AsyncMock()
        resident.save_sessions = AsyncMock(return_value=0)
        monkeypatch.setattr(runner, "_RESIDENT", resident)

        async def fake_run_scrape_graph(state, **kw):
            # Mid-scrape driver death propagating out of the browser-tier
            # nodes (WaitReadySelector/Capture) after open_tab succeeded.
            raise _driver_closed_exc()

        monkeypatch.setattr(
            "scrape_graph.runner.run_scrape_graph", fake_run_scrape_graph
        )

        with pytest.raises(runner.DriverDeath):
            await runner._run_graph(
                MagicMock(), "V1StGXR8_Z", "https://example.com/job/1",
                "example.com", None,
            )

        # Re-queued as hold, NEVER failed — the whole point of CC-160.
        statuses = [p.get("status") for p in patched]
        assert "hold" in statuses
        assert "failed" not in statuses
        hold_patch = next(p for p in patched if p.get("status") == "hold")
        assert "[driver-death]" in hold_patch.get("note", "")

    @pytest.mark.asyncio
    async def test_mid_scrape_content_error_still_marks_failed(
        self, monkeypatch
    ):
        """A genuine (non-driver) error raised mid-scrape still fails the
        scrape — CC-160 must not turn content failures into hold re-queues."""
        patched: list[dict] = []

        async def fake_update(api, scrape_id, **kw):
            patched.append({"id": scrape_id, **kw})
            return "data:\n  ok: true"

        monkeypatch.setattr(runner, "update_scrape", fake_update)

        resident = MagicMock()
        resident.open_tab = AsyncMock(return_value=MagicMock())
        resident.close_tab = AsyncMock()
        resident.save_sessions = AsyncMock(return_value=0)
        monkeypatch.setattr(runner, "_RESIDENT", resident)

        async def fake_run_scrape_graph(state, **kw):
            raise RuntimeError("real content failure, not a driver problem")

        monkeypatch.setattr(
            "scrape_graph.runner.run_scrape_graph", fake_run_scrape_graph
        )

        result = await runner._run_graph(
            MagicMock(), "V1StGXR8_Z", "https://example.com/job/1",
            "example.com", None,
        )

        assert result is False
        statuses = [p.get("status") for p in patched]
        assert "failed" in statuses
        assert "hold" not in statuses
