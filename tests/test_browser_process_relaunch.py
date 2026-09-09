"""CC-172 — relaunch the browser PROCESS when the driver process dies.

CC-141 taught ``ResidentBrowser`` to rebuild its BrowserContext and work-page
after a driver-closed error, and taught the runner to re-queue the claimed
scrape as ``hold`` rather than fail it. Both were right, and neither could
recover a dead browser PROCESS: ``relaunch()`` calls ``new_context()`` on the
browser object it already holds, so when the process behind that object is
gone the rebuild raises driver-closed again and ``open_tab`` gives up with
``ResidentDriverDead``.

``relaunch()``'s own docstring said the runner would "back off and let the
process/systemd respawn the whole browser". Nothing did. The runner is started
from tmux, it catches the exception and never exits, so no ``Restart=`` unit
could fire even if one existed. Live on 2026-07-13 that produced a runner
spinning in a 120s backoff cycle holding a corpse, re-queueing every scrape it
claimed, until a human noticed.

So the runner supervises its own browser now. These tests cover the decision,
not the browser: every Playwright object here is a fake and the driver-closed
error is a plain Exception carrying the marker phrase. No browser is launched.

Covered:
  1. ``ResidentSupervisor`` swaps in a genuinely new browser process — a new
     context manager, the corpse torn down, and the old one never reused.
  2. Teardown of a corpse is best-effort — a browser whose pipe is closed
     raises on ``cookies()`` and ``__aexit__``, and that must not abort the
     relaunch.
  3. Cookies survive the swap: the replacement resident re-seeds each domain
     from SessionStore, which is what keeps the attended warm-cookie contract.
  4. The relaunch is BOUNDED — repeated launch failures and repeated launches
     that never complete a scrape both stop it, and completed work restores
     the budget so a merely flaky browser heals indefinitely.
  5. The poll loop escalates to a process relaunch at the driver-death
     threshold and resumes claiming, instead of backing off forever.
  6. With no supervisor (headless mode) the old backoff behaviour is intact.
"""

from unittest.mock import MagicMock

import pytest

import runners.scrape_runner as runner
from browser.resident import ResidentSupervisor, is_driver_closed

DRIVER_CLOSED = "Page.evaluate: Connection closed while reading from the driver"


class FakePage:
    async def goto(self, url):
        return None

    async def evaluate(self, expr):
        return 1


class FakeContext:
    def __init__(self, browser):
        self.browser = browser
        self.closed = False
        self.added_cookies = []

    async def new_page(self):
        return FakePage()

    async def add_cookies(self, cookies):
        self.added_cookies.extend(cookies)

    async def cookies(self):
        return list(self.added_cookies)

    async def close(self):
        self.closed = True


class FakeBrowser:
    """A browser whose process can be 'killed' by setting ``dead``."""

    def __init__(self, name):
        self.name = name
        self.dead = False
        self.contexts = []

    async def new_context(self):
        if self.dead:
            raise Exception(DRIVER_CLOSED)
        ctx = FakeContext(self)
        self.contexts.append(ctx)
        return ctx


class FakeLaunchCM:
    """Stand-in for ``launch_browser(...)`` — an async context manager."""

    def __init__(self, browser, exit_error=None):
        self.browser = browser
        self.entered = False
        self.exited = False
        self.exit_error = exit_error

    async def __aenter__(self):
        self.entered = True
        return self.browser

    async def __aexit__(self, *exc_info):
        self.exited = True
        if self.exit_error is not None:
            raise self.exit_error
        return False


def make_launcher(browsers, exit_error=None):
    """Return (launch_callable, list_of_managers_it_handed_out)."""
    managers = []
    queue = list(browsers)

    def launch():
        cm = FakeLaunchCM(queue.pop(0), exit_error=exit_error)
        managers.append(cm)
        return cm

    return launch, managers


async def no_sleep(_secs):
    return None


class TestDriverClosedMarker:
    def test_the_fixture_error_is_recognised_as_driver_death(self):
        # If this ever stops matching, every test below is testing nothing.
        assert is_driver_closed(Exception(DRIVER_CLOSED))


class TestSupervisorRelaunchesTheProcess:
    @pytest.mark.asyncio
    async def test_relaunch_enters_a_new_manager_and_new_browser(self):
        dead, fresh = FakeBrowser("dead"), FakeBrowser("fresh")
        launch, managers = make_launcher([dead, fresh])
        sup = ResidentSupervisor(launch, sleep=no_sleep)

        first = await sup.start()
        assert first.browser is dead

        # The process dies under us.
        dead.dead = True
        assert await sup.relaunch_process() is True

        # A genuinely new process: second manager entered, first one exited.
        assert len(managers) == 2
        assert managers[0].exited is True
        assert managers[1].entered is True
        assert sup.resident is not None
        assert sup.resident.browser is fresh
        # And a new ResidentBrowser, not the corpse rebuilt.
        assert sup.resident is not first
        assert sup.total_relaunches == 1

    @pytest.mark.asyncio
    async def test_relaunched_resident_has_a_live_page(self):
        launch, _ = make_launcher([FakeBrowser("a"), FakeBrowser("b")])
        sup = ResidentSupervisor(launch, sleep=no_sleep)
        await sup.start()
        await sup.relaunch_process()
        # ensure_ready() ran on the replacement, so the work-page exists.
        assert sup.resident.page is not None

    @pytest.mark.asyncio
    async def test_teardown_of_a_corpse_does_not_abort_the_relaunch(self):
        """A dead browser raises on close AND on manager exit. Both swallowed."""
        dead, fresh = FakeBrowser("dead"), FakeBrowser("fresh")
        launch, managers = make_launcher(
            [dead, fresh], exit_error=Exception(DRIVER_CLOSED)
        )
        sup = ResidentSupervisor(launch, sleep=no_sleep)
        await sup.start()

        # Make the resident's own teardown throw too, the way a closed pipe does.
        async def boom():
            raise Exception(DRIVER_CLOSED)

        sup.resident.close = boom
        dead.dead = True

        assert await sup.relaunch_process() is True
        assert managers[0].exited is True
        assert sup.resident.browser is fresh

    @pytest.mark.asyncio
    async def test_backs_off_before_relaunching(self):
        """A camoufox that just died needs its profile lock to clear."""
        slept: list[float] = []

        async def record(secs):
            slept.append(secs)

        launch, _ = make_launcher([FakeBrowser("a"), FakeBrowser("b")])
        sup = ResidentSupervisor(launch, backoff_seconds=15.0, sleep=record)
        await sup.start()
        await sup.relaunch_process()

        assert slept == [15.0]


class TestCookiesSurviveTheRelaunch:
    @pytest.mark.asyncio
    async def test_replacement_reseeds_the_domain_from_session_store(self):
        """The attended warm-cookie contract: a login solved by hand before the
        crash must still be in effect after the browser is replaced.
        """
        launch, _ = make_launcher([FakeBrowser("a"), FakeBrowser("b")])
        sup = ResidentSupervisor(launch, sleep=no_sleep)
        first = await sup.start()

        stored = [{"name": "li_at", "value": "warm", "domain": ".linkedin.com"}]
        first._session_store.load = lambda domain: list(stored)

        await first.open_tab(domain="linkedin.com")
        assert "linkedin.com" in first._seeded_domains

        await sup.relaunch_process()
        replacement = sup.resident
        replacement._session_store.load = lambda domain: list(stored)

        # The replacement starts unseeded, so the next open_tab reloads from
        # disk rather than assuming the dead browser's in-memory jar.
        assert replacement._seeded_domains == set()
        await replacement.open_tab(domain="linkedin.com")
        ctx = replacement.browser.contexts[0]
        assert ctx.added_cookies == stored


class TestRelaunchIsBounded:
    @pytest.mark.asyncio
    async def test_repeated_launch_failures_exhaust_the_budget(self):
        """A host that cannot start a browser at all must stop being asked."""

        def launch():
            raise Exception("camoufox binary not found")

        sup = ResidentSupervisor(
            lambda: FakeLaunchCM(FakeBrowser("a")),
            max_failed_relaunches=3,
            sleep=no_sleep,
        )
        await sup.start()
        sup._launch = launch

        assert await sup.relaunch_process() is False
        assert await sup.relaunch_process() is False
        assert await sup.relaunch_process() is False
        assert sup.exhausted is True
        # Budget spent — it does not even try a fourth time.
        assert await sup.relaunch_process() is False
        assert sup.failed_relaunches == 3

    @pytest.mark.asyncio
    async def test_a_crash_loop_that_launches_cleanly_also_stops(self):
        """Launching fine and dying instantly must not relaunch forever."""
        launch, _ = make_launcher([FakeBrowser(str(i)) for i in range(10)])
        sup = ResidentSupervisor(
            launch, max_unproductive_relaunches=3, sleep=no_sleep
        )
        await sup.start()

        assert await sup.relaunch_process() is True
        assert await sup.relaunch_process() is True
        assert await sup.relaunch_process() is True
        assert sup.exhausted is True
        assert await sup.relaunch_process() is False

    @pytest.mark.asyncio
    async def test_completed_work_restores_the_budget(self):
        """A flaky-but-useful browser should be healed indefinitely."""
        launch, _ = make_launcher([FakeBrowser(str(i)) for i in range(10)])
        sup = ResidentSupervisor(
            launch, max_unproductive_relaunches=2, sleep=no_sleep
        )
        await sup.start()

        assert await sup.relaunch_process() is True
        assert await sup.relaunch_process() is True
        assert sup.exhausted is True

        sup.note_progress()  # a scrape completed on the current browser

        assert sup.exhausted is False
        assert await sup.relaunch_process() is True

    @pytest.mark.asyncio
    async def test_backoff_escalates_and_is_capped(self):
        slept: list[float] = []

        async def record(secs):
            slept.append(secs)

        launch, _ = make_launcher([FakeBrowser(str(i)) for i in range(10)])
        sup = ResidentSupervisor(
            launch,
            max_unproductive_relaunches=5,
            backoff_seconds=10.0,
            max_backoff_seconds=40.0,
            sleep=record,
        )
        await sup.start()
        for _ in range(4):
            await sup.relaunch_process()

        assert slept == [10.0, 20.0, 40.0, 40.0]


class TestPollLoopEscalatesToProcessRelaunch:
    @pytest.mark.asyncio
    async def test_relaunches_and_resumes_claiming_instead_of_backing_off(
        self, monkeypatch
    ):
        """THE REGRESSION THIS TICKET IS ABOUT.

        Consecutive driver deaths used to end in a 120s backoff that never
        lifted, because the death counter only reset on a successful poll and
        no poll could succeed against a dead browser. Now the loop relaunches
        the process and the very next claim works.
        """
        sleeps: list[float] = []

        async def fake_sleep(secs):
            sleeps.append(secs)

        monkeypatch.setattr(runner.asyncio, "sleep", fake_sleep)

        launch, managers = make_launcher([FakeBrowser("a"), FakeBrowser("b")])
        sup = ResidentSupervisor(launch, sleep=no_sleep)
        await sup.start()

        # Die exactly as many times as it takes to cross the threshold, then
        # work — which is what a healed browser looks like.
        deaths = runner._DRIVER_DEATH_BACKOFF_THRESHOLD
        calls = {"n": 0}

        async def fake_poll_once(api):
            calls["n"] += 1
            if calls["n"] <= deaths:
                raise runner.DriverDeath("browser dead")
            return 1

        monkeypatch.setattr(runner, "poll_once", fake_poll_once)

        def running_flag():
            return calls["n"] <= deaths  # stop right after the post-relaunch poll

        await runner._run_poll_loop(MagicMock(), running_flag, supervisor=sup)

        # The process was actually replaced...
        assert len(managers) == 2
        assert sup.total_relaunches == 1
        # ...the runner's module-global resident points at the NEW browser, so
        # the next scrape does not reach through to the corpse...
        assert runner._RESIDENT is sup.resident
        assert runner._RESIDENT.browser is managers[1].browser
        # ...and it went straight back to claiming rather than serving out the
        # long backoff sleep.
        assert not any(s >= runner._DRIVER_DEATH_BACKOFF_SECONDS for s in sleeps)

    @pytest.mark.asyncio
    async def test_falls_back_to_backoff_when_the_relaunch_fails(
        self, monkeypatch
    ):
        """A genuinely broken host still backs off loudly — that part was right."""
        sleeps: list[float] = []

        async def fake_sleep(secs):
            sleeps.append(secs)

        monkeypatch.setattr(runner.asyncio, "sleep", fake_sleep)

        sup = ResidentSupervisor(
            lambda: FakeLaunchCM(FakeBrowser("a")), sleep=no_sleep
        )
        await sup.start()

        def dead_host():
            raise Exception("cannot start browser: no display")

        sup._launch = dead_host

        calls = {"n": 0}

        async def fake_poll_once(api):
            calls["n"] += 1
            raise runner.DriverDeath("browser dead")

        monkeypatch.setattr(runner, "poll_once", fake_poll_once)

        def running_flag():
            return calls["n"] < runner._DRIVER_DEATH_BACKOFF_THRESHOLD

        errors: list[str] = []
        monkeypatch.setattr(
            runner.logger,
            "error",
            lambda msg, *a, **kw: errors.append(msg % a if a else msg),
        )

        await runner._run_poll_loop(MagicMock(), running_flag, supervisor=sup)

        assert sup.total_relaunches == 0
        assert any(s > runner.POLL_INTERVAL for s in sleeps)
        assert any("backing off" in e.lower() for e in errors)

    @pytest.mark.asyncio
    async def test_drain_gives_up_when_the_relaunch_fails(self, monkeypatch):
        """Batch mode still reports a dead browser through the exit code."""

        async def fake_sleep(secs):
            return None

        monkeypatch.setattr(runner.asyncio, "sleep", fake_sleep)

        sup = ResidentSupervisor(
            lambda: FakeLaunchCM(FakeBrowser("a")), sleep=no_sleep
        )
        await sup.start()
        sup._launch = MagicMock(side_effect=Exception("no display"))

        async def fake_poll_once(api):
            raise runner.DriverDeath("browser dead")

        monkeypatch.setattr(runner, "poll_once", fake_poll_once)

        clean = await runner._run_poll_loop(
            MagicMock(), lambda: True, drain=True, supervisor=sup
        )
        assert clean is False


class TestNoSupervisorKeepsOldBehaviour:
    @pytest.mark.asyncio
    async def test_headless_mode_still_backs_off(self, monkeypatch):
        """Headless _run_graph already launches a browser per scrape, so there
        is no resident process to relaunch and nothing should change.
        """
        sleeps: list[float] = []

        async def fake_sleep(secs):
            sleeps.append(secs)

        monkeypatch.setattr(runner.asyncio, "sleep", fake_sleep)

        calls = {"n": 0}

        async def fake_poll_once(api):
            calls["n"] += 1
            raise runner.DriverDeath("browser dead")

        monkeypatch.setattr(runner, "poll_once", fake_poll_once)

        def running_flag():
            return calls["n"] < runner._DRIVER_DEATH_BACKOFF_THRESHOLD

        await runner._run_poll_loop(MagicMock(), running_flag)

        assert any(s > runner.POLL_INTERVAL for s in sleeps)
