"""ResidentBrowser — long-lived headed browser with ephemeral per-scrape tabs.

One browser, one BrowserContext, one anchor page that stays open for the
process lifetime, plus one ephemeral tab spawned per scrape via
window.open() and closed when the scrape finishes.

Why window.open() instead of ctx.new_page(): Playwright Firefox / Camoufox
renders ctx.new_page() as a separate OS window. Pages spawned via
anchor.evaluate("window.open(...)") and captured via expect_page() honor
the user's browser.link.open_newwindow preference and land as tabs in the
anchor's window — the attended ergonomics we want.

Cookies persist on the shared context, so login state survives across
scrapes regardless of which tab solved the auth challenge.
"""

from __future__ import annotations

import logging
from typing import Any

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
    rebuilt. Raised by ``open_tab`` after a relaunch attempt still hits a
    driver-closed error. The runner treats this as infra-death — re-queue
    the scrape, do NOT mark it ``failed`` — and backs off from claiming.
    """


class ResidentBrowser:
    def __init__(self, browser):
        self._browser = browser
        self._context = None
        self._anchor = None
        self._session_store = SessionStore()
        self._seeded_domains: set[str] = set()

    @property
    def browser(self):
        return self._browser

    async def _ensure_context(self):
        if self._context is None:
            self._context = await self._browser.new_context()
            self._anchor = await self._context.new_page()
            try:
                await self._anchor.goto("about:blank")
            except Exception:
                pass
        return self._context

    async def relaunch(self) -> None:
        """Tear down and rebuild the context + anchor on the existing browser.

        Called when a driver-closed error is seen mid-run. Rebuilds the
        BrowserContext and anchor page so the next ``open_tab`` runs against
        a live pipe. Cookies are re-seeded lazily per domain on the next
        ``open_tab`` (``_seeded_domains`` is cleared), so warm login state is
        reloaded from SessionStore rather than lost.

        If the browser process itself is dead — not just the context — the
        ``new_context()`` below re-raises the driver-closed error; the caller
        (``open_tab``) converts that into ``ResidentDriverDead`` so the runner
        backs off and lets the process/systemd respawn the whole browser.
        """
        logger.warning("Resident: relaunching context+anchor after driver death")
        old_ctx = self._context
        self._context = None
        self._anchor = None
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
        """Spawn a fresh blank tab in the resident window and return it.

        On first encounter for a domain, seed cookies (from arg or
        SessionStore) into the shared context so subsequent navigations
        run already-authenticated. The caller is expected to drive the
        page (typically via the graph's Navigate node).

        If the resident driver connection is dead (the browser subprocess
        died on a previous scrape), the first Playwright call here raises a
        driver-closed error. We relaunch the context+anchor ONCE and retry
        rather than let the scrape fail: a dead driver is infrastructure
        death, not a content failure, and the resident model exists to be
        rebuilt. If the retry still hits a driver-closed error the browser
        process itself is gone — raise ``ResidentDriverDead`` so the runner
        re-queues the scrape and backs off.
        """
        try:
            return await self._open_tab_once(domain, seed_cookies)
        except Exception as exc:
            if not is_driver_closed(exc):
                raise
            logger.warning(
                "Resident: driver closed opening tab for %s; relaunching and retrying",
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
        assert self._anchor is not None  # _ensure_context() set it

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

        async with ctx.expect_page() as new_page_info:
            await self._anchor.evaluate("window.open('about:blank', '_blank')")
        return await new_page_info.value

    async def close_tab(self, page: Any) -> None:
        try:
            await page.close()
        except Exception:
            logger.debug("Resident: tab close raised", exc_info=True)

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
        self._anchor = None
        self._seeded_domains.clear()


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
