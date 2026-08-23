"""``--drain``: the runner as a BATCH job rather than a daemon.

Doug stopped scraping for other people (2026-08-23) and no longer wants a
resident Camoufox polling ``claim-next`` every 30s around the clock — it was
~2,900 empty 204 polls a day, keeping the Cloud Run api instance awake and
inflating Logfire ingestion for no work done. The replacement is a timer that
fires at 06:00 and 18:00, drains whatever email triage queued, and exits.

"Exits" is the load-bearing word. A ``systemd`` timer whose service never
terminates is a service that runs once and then blocks every later fire,
because the unit is still active. So drain mode needs three properties, and
each has a test here:

  1. It STOPS when the queue is empty — otherwise the timer is just the daemon
     with extra steps.
  2. It does NOT stop on the first empty poll — one unlucky 204 between two
     queued scrapes must not end the batch early.
  3. It reports a dead browser through the EXIT CODE — an unattended run that
     silently does nothing is worse than one that fails loudly.

The daemon path must keep its old behaviour, so the last test pins that
drain=False still loops and backs off rather than returning.
"""

from unittest.mock import MagicMock

import pytest

import runners.scrape_runner as runner


@pytest.fixture
def no_sleep(monkeypatch):
    """Record sleep durations instead of actually waiting."""
    sleeps: list[float] = []

    async def fake_sleep(secs):
        sleeps.append(secs)

    monkeypatch.setattr(runner.asyncio, "sleep", fake_sleep)
    return sleeps


def _always_running():
    return True


class TestDrainStops:
    @pytest.mark.asyncio
    async def test_exits_once_the_queue_is_empty(self, monkeypatch, no_sleep):
        # Two scrapes, then nothing. Drain should process both and return.
        counts = [1, 1, 0, 0]

        async def fake_poll_once(api):
            return counts.pop(0) if counts else 0

        monkeypatch.setattr(runner, "poll_once", fake_poll_once)

        clean = await runner._run_poll_loop(
            MagicMock(), _always_running, drain=True
        )

        assert clean is True
        # It stopped on its own despite running_flag() never going false.
        assert counts == []

    @pytest.mark.asyncio
    async def test_does_not_stop_on_a_single_empty_poll(self, monkeypatch, no_sleep):
        # A 204 between two queued scrapes is a race, not an empty queue.
        # _DRAIN_IDLE_POLLS=2 means the loop must survive the gap and pick
        # the second scrape up.
        seen = {"n": 0}
        counts = [1, 0, 1, 0, 0]

        async def fake_poll_once(api):
            seen["n"] += 1
            return counts.pop(0) if counts else 0

        monkeypatch.setattr(runner, "poll_once", fake_poll_once)

        clean = await runner._run_poll_loop(
            MagicMock(), _always_running, drain=True
        )

        assert clean is True
        # All five polls consumed — it did not bail at the first 0.
        assert seen["n"] == 5

    @pytest.mark.asyncio
    async def test_does_not_idle_between_scrapes_while_draining(
        self, monkeypatch, no_sleep
    ):
        # A backlog should clear at browser speed, not one scrape per
        # POLL_INTERVAL. Productive cycles skip the sleep entirely.
        counts = [1, 1, 1, 0, 0]

        async def fake_poll_once(api):
            return counts.pop(0) if counts else 0

        monkeypatch.setattr(runner, "poll_once", fake_poll_once)

        await runner._run_poll_loop(MagicMock(), _always_running, drain=True)

        # Three productive polls slept zero times; only the two trailing
        # empty polls paced themselves.
        assert len(no_sleep) <= 2


class TestDrainSurfacesBrowserDeath:
    @pytest.mark.asyncio
    async def test_gives_up_and_reports_failure(self, monkeypatch, no_sleep):
        async def fake_poll_once(api):
            raise runner.DriverDeath("browser dead")

        monkeypatch.setattr(runner, "poll_once", fake_poll_once)

        errors: list[str] = []
        monkeypatch.setattr(
            runner.logger, "error",
            lambda msg, *a, **kw: errors.append(msg % a if a else msg),
        )

        clean = await runner._run_poll_loop(
            MagicMock(), _always_running, drain=True
        )

        # False is what main() turns into a non-zero exit, so the failure is
        # visible in `systemctl --user status` rather than silent.
        assert clean is False
        assert any("abandoning this drain" in e.lower() for e in errors)
        # It did NOT sit in the daemon's 120s backoff waiting for a human.
        assert all(s <= runner.POLL_INTERVAL for s in no_sleep)


class TestDaemonPathUnchanged:
    @pytest.mark.asyncio
    async def test_daemon_keeps_polling_an_empty_queue(self, monkeypatch, no_sleep):
        # Without --drain, an empty queue is normal and must not end the loop.
        calls = {"n": 0}

        async def fake_poll_once(api):
            calls["n"] += 1
            return 0

        def running_flag():
            return calls["n"] < 5

        monkeypatch.setattr(runner, "poll_once", fake_poll_once)

        clean = await runner._run_poll_loop(MagicMock(), running_flag)

        assert clean is True
        # Ran until the flag stopped it, not until the queue looked empty.
        assert calls["n"] == 5
        # And it paced every cycle, unlike drain.
        assert len(no_sleep) == 5
