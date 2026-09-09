"""ResidentBrowser — long-lived headed browser with ONE persistent work-page.

One browser, one BrowserContext, one page that stays open for the whole
process lifetime. It is created eagerly when the headed runner starts, so
the resident window is always up (idle = the one tab sits on about:blank or
the last job page). Each scrape NAVIGATES that same page to the job URL and
hands it back — it is never closed between scrapes and no fresh tab is
spawned per scrape.

Why one persistent page via ctx.new_page(): Playwright Firefox / Camoufox
renders ctx.new_page() as a separate OS window. With a SINGLE persistent
page that's exactly what we want — one window == one tab == the workbench.
(The old model kept an idle "anchor" page and spawned an ephemeral tab per
scrape via anchor.evaluate("window.open(...)"), which flashed a window open
and closed it on completion; that per-scrape tab churn is gone.) A separate
omarchy windowrule parks class ``camoufox-default`` into ``special:scratchpad``,
so no agents-side window management is needed — we just guarantee exactly one
top-level window exists.

Doug's rationale for one reused tab: "prefer one so I don't hold the past in
memory" — bounded memory, no tab pile-up, the last job page stays visible.

Cookies persist on the shared context, so login state survives across scrapes
regardless of which navigation solved the auth challenge.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

from browser.session_store import SessionStore

logger = logging.getLogger(__name__)

# Playwright surfaces a dead driver connection as an Exception whose message
# contains this phrase, regardless of which call tripped it (evaluate,
# screenshot, cookies, …). Match on the phrase, not a single call site: the
# resident window can die at any point mid-run and the next Playwright call
# — whichever it is — carries the same marker.
_DRIVER_CLOSED_MARKER = "connection closed while reading from the driver"


def is_driver_closed(exc: BaseException) -> bool:
    """True when ``exc`` is a Playwright driver-connection-dead error.

    The resident browser subprocess can die mid-run; every subsequent
    Playwright call then raises with "Connection closed while reading from
    the driver" in the message. This is infrastructure death, distinct from
    a page/content failure, and the caller must rebuild the resident rather
    than fail the claimed scrape.
    """
    return _DRIVER_CLOSED_MARKER in str(exc).lower()


class ResidentDriverDead(Exception):
    """The resident browser's driver connection is dead and could not be
    rebuilt on the existing browser object. Raised by ``open_tab`` after a
    relaunch attempt still hits a driver-closed error, which means the
    browser PROCESS itself is gone, not just the context.

    The runner treats this as infra-death — re-queue the scrape, do NOT mark
    it ``failed`` — and, past a small threshold of consecutive deaths, asks
    its ``ResidentSupervisor`` to relaunch the browser process (CC-172).
    """


class ResidentBrowser:
    def __init__(self, browser):
        self._browser = browser
        self._context = None
        # The single persistent work-page. Created eagerly by ensure_ready()
        # / _ensure_context() and reused for every scrape — never closed
        # between scrapes, rebuilt only on driver death via relaunch().
        self._page = None
        self._session_store = SessionStore()
        self._seeded_domains: set[str] = set()

    @property
    def browser(self):
        return self._browser

    @property
    def page(self):
        """The single persistent work-page (None until ensure_ready)."""
        return self._page

    async def _ensure_context(self):
        if self._context is None:
            self._context = await self._browser.new_context()
            self._page = await self._context.new_page()
            try:
                await self._page.goto("about:blank")
            except Exception:
                pass
        return self._context

    async def ensure_ready(self) -> None:
        """Eagerly build the context + persistent page so the resident window
        is up before the first scrape.

        Called once when the headed runner starts. Idle state is then the one
        persistent tab sitting on about:blank; each scrape navigates it.
        """
        await self._ensure_context()

    async def relaunch(self) -> None:
        """Tear down and rebuild the context + persistent page on the existing
        browser.

        Called when a driver-closed error is seen mid-run. Rebuilds the
        BrowserContext and the single work-page so the next ``open_tab`` runs
        against a live pipe. Cookies are re-seeded lazily per domain on the
        next ``open_tab`` (``_seeded_domains`` is cleared), so warm login state
        is reloaded from SessionStore rather than lost.

        This rebuilds the context on the EXISTING browser object and cannot
        recover a dead browser PROCESS. If the process itself is gone the
        ``new_context()`` below re-raises the driver-closed error; the caller
        (``open_tab``) converts that into ``ResidentDriverDead``, and the
        runner escalates to ``ResidentSupervisor.relaunch_process()`` — which
        launches a genuinely new browser.

        (This docstring used to say the runner "lets the process/systemd
        respawn the whole browser". Nothing did: the runner is started from
        tmux, catches ``ResidentDriverDead`` and never exits, so no
        ``Restart=`` unit could fire. That is the hole CC-172 closed.)
        """
        logger.warning("Resident: relaunching context+page after driver death")
        old_ctx = self._context
        self._context = None
        self._page = None
        self._seeded_domains.clear()
        if old_ctx is not None:
            try:
                await old_ctx.close()
            except Exception:
                logger.debug("Resident: old context close raised during relaunch", exc_info=True)
        # Rebuild eagerly so a dead browser surfaces here (via driver-closed)
        # rather than on the next open_tab.
        await self._ensure_context()

    async def open_tab(self, domain: str = "", seed_cookies: list[dict] | None = None):
        """Return the single persistent work-page for this scrape.

        This does NOT spawn a new tab — it hands back the same reused page
        every time (bounded memory, no tab pile-up). On first encounter for a
        domain, seed cookies (from arg or SessionStore) into the shared context
        so subsequent navigations run already-authenticated. The caller is
        expected to navigate the page (typically via the graph's Navigate node)
        and must NOT close it between scrapes.

        If the resident driver connection is dead (the browser subprocess died
        on a previous scrape), the first Playwright call here raises a
        driver-closed error. We relaunch the context+page ONCE and retry rather
        than let the scrape fail: a dead driver is infrastructure death, not a
        content failure, and the resident model exists to be rebuilt. If the
        retry still hits a driver-closed error the browser process itself is
        gone — raise ``ResidentDriverDead`` so the runner re-queues the scrape
        and backs off.
        """
        try:
            return await self._open_tab_once(domain, seed_cookies)
        except Exception as exc:
            if not is_driver_closed(exc):
                raise
            logger.warning(
                "Resident: driver closed preparing page for %s; relaunching and retrying",
                domain or "?",
            )
        # One relaunch + retry. Any driver-closed error on this path means
        # the browser process is gone, not just the context.
        try:
            await self.relaunch()
            return await self._open_tab_once(domain, seed_cookies)
        except Exception as exc:
            if is_driver_closed(exc):
                raise ResidentDriverDead(
                    "resident browser driver dead after relaunch"
                ) from exc
            raise

    async def _open_tab_once(self, domain: str, seed_cookies: list[dict] | None):
        ctx = await self._ensure_context()
        assert self._page is not None  # _ensure_context() set it

        if domain and domain not in self._seeded_domains:
            cookies = seed_cookies or self._session_store.load(domain) or []
            if cookies:
                try:
                    await ctx.add_cookies(cookies)
                    logger.info("Resident: seeded %d cookies for %s", len(cookies), domain)
                except Exception as exc:
                    if is_driver_closed(exc):
                        raise
                    logger.warning("Resident: cookie seed failed for %s: %s", domain, exc)
            self._seeded_domains.add(domain)

        # A no-op Playwright call so a dead driver surfaces HERE (and is
        # relaunched by open_tab) rather than later inside the graph's
        # Navigate node, where it would be misread as a content failure.
        await self._page.evaluate("1")
        return self._page

    async def close_tab(self, page: Any) -> None:
        """No-op — the persistent work-page is deliberately NOT closed between
        scrapes. It stays on the last job page (or about:blank) while idle so
        the resident window never flashes shut. Signature kept for callers.
        """
        return None

    async def save_sessions(self) -> int:
        """Write current cookies back to SessionStore, one file per seeded
        domain. Called after each scrape and on shutdown so manually-solved
        logins persist across poller restarts.
        """
        if self._context is None or not self._seeded_domains:
            return 0
        try:
            all_cookies = await self._context.cookies()
        except Exception as exc:
            logger.warning("Resident: cookies() failed, skipping save: %s", exc)
            return 0
        saved = 0
        for domain in self._seeded_domains:
            matches = [
                c for c in all_cookies
                if _cookie_matches_domain(c.get("domain") or "", domain)
            ]
            if not matches:
                continue
            self._session_store.save(domain, matches)
            saved += 1
        return saved

    async def close(self):
        if self._context is not None:
            try:
                await self._context.close()
            except Exception:
                pass
            self._context = None
        self._page = None
        self._seeded_domains.clear()


# --------------------------------------------------------------------------
# Browser-PROCESS supervision (CC-172)
# --------------------------------------------------------------------------

# Consecutive relaunch attempts that FAILED to produce a working browser
# before the supervisor calls the host broken. A relaunch that keeps throwing
# is not a transient crash — it is a missing binary, no display, or an OOM
# host — and retrying it forever hides the outage instead of surfacing it.
DEFAULT_MAX_FAILED_RELAUNCHES = 5

# Consecutive SUCCESSFUL relaunches that produced no completed scrape before
# the supervisor gives up. This is the separate bound that catches a hot crash
# loop: a browser that launches cleanly and dies on first contact would
# otherwise satisfy the failure counter forever. Reset by ``note_progress()``,
# so a merely flaky browser keeps healing indefinitely — which is the point.
DEFAULT_MAX_UNPRODUCTIVE_RELAUNCHES = 5

# First wait before a relaunch, doubled per consecutive attempt up to the cap.
# A Camoufox that just died needs a moment for its profile lock and juggler
# pipe to clear; relaunching instantly tends to fail for that reason alone.
DEFAULT_RELAUNCH_BACKOFF_SECONDS = 15.0
DEFAULT_MAX_RELAUNCH_BACKOFF_SECONDS = 240.0


class ResidentSupervisor:
    """Owns the browser PROCESS behind a :class:`ResidentBrowser`.

    ``ResidentBrowser.relaunch()`` rebuilds the BrowserContext and work-page
    on the browser object it was handed. That recovers a dead *context*, but
    when the Camoufox/Chromium **process** is gone there is nothing left to
    build a context on: the rebuild raises driver-closed again and
    ``open_tab`` turns it into :class:`ResidentDriverDead`.

    Before this class the runner's answer was to back off and wait for "the
    process/systemd" to respawn the browser. Nothing did — the runner is
    started from tmux, holds no browser, and never exits, so no ``Restart=``
    unit could fire. A process death meant an idle runner holding a corpse
    until a human noticed (CC-172).

    The supervisor closes that loop by owning the launch context manager
    itself rather than sitting inside somebody's ``async with``, so it can
    tear the corpse down and enter a *fresh* one in-process. That works
    identically under tmux, systemd and Docker, none of which are consulted.

    **Cookies survive a relaunch.** The runner calls ``save_sessions()`` after
    every scrape, so warm login state is already on disk in SessionStore; the
    replacement ``ResidentBrowser`` starts with an empty ``_seeded_domains``
    and re-seeds each domain from disk on its next ``open_tab``. A captcha or
    login solved by hand in the attended window is therefore not lost when the
    browser dies under it.

    ``launch`` is a zero-argument callable returning a fresh async context
    manager yielding a Playwright-compatible browser — in practice
    ``lambda: launch_browser(get_engine(), headless=False)``. It is called
    once per process launch, so each relaunch gets a genuinely new manager
    rather than a re-entered spent one.
    """

    def __init__(
        self,
        launch: Callable[[], Any],
        *,
        max_failed_relaunches: int = DEFAULT_MAX_FAILED_RELAUNCHES,
        max_unproductive_relaunches: int = DEFAULT_MAX_UNPRODUCTIVE_RELAUNCHES,
        backoff_seconds: float = DEFAULT_RELAUNCH_BACKOFF_SECONDS,
        max_backoff_seconds: float = DEFAULT_MAX_RELAUNCH_BACKOFF_SECONDS,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._launch = launch
        self._cm: Any = None
        self.resident: ResidentBrowser | None = None
        self._max_failed = max_failed_relaunches
        self._max_unproductive = max_unproductive_relaunches
        self._backoff_seconds = backoff_seconds
        self._max_backoff_seconds = max_backoff_seconds
        self._sleep = sleep or asyncio.sleep
        # Consecutive relaunch attempts that raised. Reset by a launch that
        # produced a working browser.
        self.failed_relaunches = 0
        # Consecutive successful relaunches with no completed scrape since.
        # Reset by note_progress().
        self.unproductive_relaunches = 0
        # Lifetime count, for the shutdown line — how sick was this run?
        self.total_relaunches = 0

    @property
    def exhausted(self) -> bool:
        """True when the relaunch budget is spent and the host looks broken.

        Either bound trips it: repeated launch failures (nothing to run) or
        repeated launches that never got a scrape done (a crash loop).
        """
        return (
            self.failed_relaunches >= self._max_failed
            or self.unproductive_relaunches >= self._max_unproductive
        )

    async def start(self) -> ResidentBrowser:
        """Launch the browser process and build the resident window.

        Raises whatever the engine raises — a missing Camoufox binary must
        surface at startup, not be swallowed into an idle runner.
        """
        cm = self._launch()
        browser = await cm.__aenter__()
        self._cm = cm
        resident = ResidentBrowser(browser)
        await resident.ensure_ready()
        self.resident = resident
        return resident

    def note_progress(self) -> None:
        """A scrape completed on the current browser — restore the budget.

        This is what separates "flaky browser we keep healing" from "crash
        loop we should stop feeding". Real work done means the relaunch
        strategy is working, however many times it has fired.
        """
        self.unproductive_relaunches = 0
        self.failed_relaunches = 0

    async def relaunch_process(self) -> bool:
        """Tear down the dead browser and launch a new one.

        Returns True when a live resident is ready and claiming can resume,
        False when the budget is spent or the launch itself failed — the
        caller then falls back to backing off, which is the only honest
        response to a host that cannot run a browser at all.
        """
        if self.exhausted:
            logger.error(
                "Resident: browser relaunch budget spent (%d failed, %d "
                "unproductive) — NOT relaunching again. The host cannot keep a "
                "browser alive; scrapes stay queued as hold until it is fixed.",
                self.failed_relaunches,
                self.unproductive_relaunches,
            )
            return False

        attempt = self.failed_relaunches + self.unproductive_relaunches + 1
        delay = min(
            self._backoff_seconds * (2 ** (attempt - 1)),
            self._max_backoff_seconds,
        )
        logger.error(
            "Resident: browser process is dead — relaunching it in-process "
            "(attempt %d, waiting %.0fs first). CC-172: nothing external "
            "respawns this browser, so the runner does it itself.",
            attempt,
            delay,
        )

        saved = await self._teardown()
        if saved:
            logger.info(
                "Resident: persisted sessions for %d domain(s) before relaunch",
                saved,
            )
        await self._sleep(delay)

        try:
            await self.start()
        except Exception:
            self.failed_relaunches += 1
            logger.exception(
                "Resident: browser relaunch FAILED (%d consecutive failure(s) "
                "of %d allowed)",
                self.failed_relaunches,
                self._max_failed,
            )
            return False

        self.failed_relaunches = 0
        self.unproductive_relaunches += 1
        self.total_relaunches += 1
        logger.warning(
            "Resident: browser process relaunched (relaunch #%d this run) — "
            "resuming claims. Cookies re-seed from SessionStore per domain on "
            "the next tab open, so warm logins carry over.",
            self.total_relaunches,
        )
        return True

    async def _teardown(self) -> int:
        """Best-effort: persist cookies, close the resident, exit the manager.

        Every step is allowed to fail. A browser that died mid-scrape has a
        closed pipe, so ``cookies()`` and ``__aexit__`` both raise — that noise
        is expected here and must not stop the relaunch.

        Returns the number of domains whose sessions were saved (0 on a
        corpse, since its cookies are unreachable).
        """
        resident, cm = self.resident, self._cm
        self.resident = None
        self._cm = None
        saved = 0
        if resident is not None:
            try:
                saved = await resident.save_sessions()
            except Exception:
                logger.debug("Resident: save_sessions raised during teardown", exc_info=True)
            try:
                await resident.close()
            except Exception:
                logger.debug("Resident: close raised during teardown", exc_info=True)
        if cm is not None:
            try:
                await cm.__aexit__(None, None, None)
            except Exception:
                logger.debug(
                    "Resident: browser manager exit raised during teardown "
                    "(expected when the process is already gone)",
                    exc_info=True,
                )
        return saved

    async def close(self) -> int:
        """Shutdown: persist cookies and close the browser process."""
        return await self._teardown()


def _cookie_matches_domain(cookie_domain: str, target: str) -> bool:
    """Match a Playwright cookie's domain attribute against our canonical
    target domain (e.g. 'linkedin.com'). Accepts '.linkedin.com',
    'www.linkedin.com', 'linkedin.com'.
    """
    if not cookie_domain or not target:
        return False
    cd = cookie_domain.lstrip(".").lower()
    tgt = target.lower()
    return cd == tgt or cd.endswith("." + tgt)
