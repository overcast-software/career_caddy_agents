"""Scheduled windows are the runner's DEFAULT mode (2026-08-23).

Doug runs the scrape runner from tmuxinator and leaves it up. The old default
polled ``claim-next`` every 30s forever — 8,568 calls over three days, every
one a 204 No Content — which held a Camoufox process open around the clock,
kept the Cloud Run api instance from scaling to zero, and pushed ~5,700
records/day into Logfire for no work done. Email arrives in digests, so there
is nothing to find between batches.

So the scheduling lives in the runner itself rather than in a timer or a cron
entry: the process is already long-lived, and this way there is no
machine-level config to install. It wakes at 06:00 and 18:00, drains, and
sleeps again.

The properties that matter:
  * the default really is scheduled — not something you have to opt into
  * a window fires once, not repeatedly, when the clock sits on it
  * nothing runs between windows, browser included
  * a suspended machine that misses a window fires on resume
  * --continuous still gives back the old daemon unchanged
"""

from datetime import datetime, time as dtime
from unittest.mock import MagicMock

import pytest

import runners.scrape_runner as runner


def _at(iso: str) -> datetime:
    return datetime.fromisoformat(iso).astimezone()


class TestDefaultsAreScheduled:
    """The headline of this change: you get windows without asking for them."""

    def test_bare_invocation_selects_scheduled_mode_at_6_and_18(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["caddy-runner"])
        args = runner._parse_args()

        assert args.drain is False, "not a one-shot"
        assert args.continuous is False, "not the old forever-poll daemon"
        assert args.at == "06:00,18:00"
        # …and that string is the thing the scheduler actually consumes.
        assert runner.parse_run_at(args.at) == [dtime(6, 0), dtime(18, 0)]

    def test_continuous_is_available_as_an_explicit_opt_in(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["caddy-runner", "--continuous"])
        assert runner._parse_args().continuous is True

    def test_windows_are_overridable(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["caddy-runner", "--at", "07:15"])
        assert runner.parse_run_at(runner._parse_args().at) == [dtime(7, 15)]


class TestParseRunAt:
    def test_parses_the_default(self):
        assert runner.parse_run_at("06:00,18:00") == [dtime(6, 0), dtime(18, 0)]

    def test_sorts_dedupes_and_tolerates_whitespace(self):
        assert runner.parse_run_at(" 18:30 , 06:00 ,18:30") == [
            dtime(6, 0), dtime(18, 30)
        ]

    def test_bare_hour_means_top_of_the_hour(self):
        assert runner.parse_run_at("7") == [dtime(7, 0)]

    @pytest.mark.parametrize("spec", ["", " , ", "notatime", "25:00", "06:99"])
    def test_rejects_garbage_loudly(self, spec):
        # A typo'd schedule must fail at startup, not silently become "never".
        with pytest.raises(ValueError):
            runner.parse_run_at(spec)


class TestNextWindow:
    TIMES = [dtime(6, 0), dtime(18, 0)]

    @pytest.mark.parametrize("now,expected", [
        ("2026-08-23T03:00", "2026-08-23 06:00"),  # before both
        ("2026-08-23T09:30", "2026-08-23 18:00"),  # between
        ("2026-08-23T19:00", "2026-08-24 06:00"),  # after both -> tomorrow
        ("2026-08-23T23:59", "2026-08-24 06:00"),  # midnight rollover
    ])
    def test_picks_the_next_occurrence(self, now, expected):
        got = runner.next_window(self.TIMES, _at(now))
        assert got.strftime("%Y-%m-%d %H:%M") == expected

    def test_exactly_on_a_window_moves_to_the_next_one(self):
        # Strictly-after matters: returning "now" would let a just-finished
        # batch immediately re-fire in a tight loop.
        got = runner.next_window(self.TIMES, _at("2026-08-23T06:00"))
        assert got.strftime("%H:%M") == "18:00"


class TestRunScheduled:
    @pytest.mark.asyncio
    async def test_drains_once_per_window_then_sleeps_again(self, monkeypatch):
        slept: list[str] = []
        batches = {"n": 0}

        async def fake_sleep_until(target, running_flag, stop_event=None):
            slept.append(target.strftime("%H:%M"))
            return True

        async def fake_batch(api, running_flag, headed):
            batches["n"] += 1
            return True

        monkeypatch.setattr(runner, "_sleep_until", fake_sleep_until)
        monkeypatch.setattr(runner, "_run_batch", fake_batch)

        # Stop after two windows.
        def running_flag():
            return batches["n"] < 2

        clean = await runner.run_scheduled(
            MagicMock(), running_flag, False, [dtime(6, 0), dtime(18, 0)]
        )

        assert clean is True
        assert batches["n"] == 2
        # It waited before each batch rather than draining on startup.
        assert len(slept) == 2

    @pytest.mark.asyncio
    async def test_nothing_runs_when_stopped_while_waiting(self, monkeypatch):
        # Ctrl-C during the 12-hour idle must not fall through into a batch.
        async def fake_sleep_until(target, running_flag, stop_event=None):
            return False  # interrupted

        ran = {"batch": False}

        async def fake_batch(api, running_flag, headed):
            ran["batch"] = True
            return True

        monkeypatch.setattr(runner, "_sleep_until", fake_sleep_until)
        monkeypatch.setattr(runner, "_run_batch", fake_batch)

        clean = await runner.run_scheduled(
            MagicMock(), lambda: True, False, [dtime(6, 0)]
        )

        assert clean is True
        assert ran["batch"] is False

    @pytest.mark.asyncio
    async def test_a_failed_batch_does_not_stop_the_schedule(self, monkeypatch):
        # A dead browser at 06:00 must not kill the 18:00 run — the holds stay
        # claimable and get retried — but it must still be reported.
        batches = {"n": 0}

        async def fake_sleep_until(target, running_flag, stop_event=None):
            return True

        async def fake_batch(api, running_flag, headed):
            batches["n"] += 1
            return False  # browser died

        monkeypatch.setattr(runner, "_sleep_until", fake_sleep_until)
        monkeypatch.setattr(runner, "_run_batch", fake_batch)

        clean = await runner.run_scheduled(
            MagicMock(), lambda: batches["n"] < 2, False, [dtime(6, 0)]
        )

        assert batches["n"] == 2, "kept its schedule after a failure"
        assert clean is False, "but reported the failure for the exit code"


class TestSleepUntilHandlesSuspend:
    @pytest.mark.asyncio
    async def test_wakes_in_chunks_rather_than_one_long_sleep(self, monkeypatch):
        # One 12-hour asyncio.sleep would overshoot a window by however long
        # the machine was suspended. Chunked waking re-reads the clock.
        sleeps: list[float] = []

        async def fake_sleep(secs):
            sleeps.append(secs)

        monkeypatch.setattr(runner.asyncio, "sleep", fake_sleep)

        # Target far in the future; stop after a few ticks.
        target = _at("2030-01-01T06:00")
        ticks = {"n": 0}

        def running_flag():
            ticks["n"] += 1
            return ticks["n"] <= 3

        await runner._sleep_until(target, running_flag)

        assert sleeps, "it slept"
        assert all(s <= runner._SCHEDULE_TICK_S for s in sleeps), (
            "no single sleep longer than the tick — that is what makes a "
            "resumed machine notice a missed window"
        )

    @pytest.mark.asyncio
    async def test_a_stop_event_interrupts_the_wait_immediately(self):
        # Regression: without this, Ctrl-C in the tmux pane is only noticed
        # when the current chunk elapses — up to 5 minutes of looking hung.
        # Found in a smoke test where the runner outlived its `timeout 30`.
        import asyncio as _asyncio

        stop_event = _asyncio.Event()
        stop_event.set()

        # Target is a decade out; if the event were ignored this would hang.
        fired = await _asyncio.wait_for(
            runner._sleep_until(_at("2036-01-01T06:00"), lambda: True, stop_event),
            timeout=2,
        )
        assert fired is False, "reported as interrupted, not as a due window"

    @pytest.mark.asyncio
    async def test_returns_immediately_when_the_window_already_passed(
        self, monkeypatch
    ):
        # The suspend case: we come back and 06:00 is behind us. Fire now.
        async def fake_sleep(secs):  # pragma: no cover - must not be reached
            raise AssertionError("should not sleep past a due window")

        monkeypatch.setattr(runner.asyncio, "sleep", fake_sleep)

        fired = await runner._sleep_until(_at("2020-01-01T06:00"), lambda: True)
        assert fired is True
