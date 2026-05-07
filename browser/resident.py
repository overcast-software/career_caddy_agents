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

    async def open_tab(self, domain: str = "", seed_cookies: list[dict] | None = None):
        """Spawn a fresh blank tab in the resident window and return it.

        On first encounter for a domain, seed cookies (from arg or
        SessionStore) into the shared context so subsequent navigations
        run already-authenticated. The caller is expected to drive the
        page (typically via the graph's Navigate node).
        """
        ctx = await self._ensure_context()
        assert self._anchor is not None  # _ensure_context() set it

        if domain and domain not in self._seeded_domains:
            cookies = seed_cookies or self._session_store.load(domain) or []
            if cookies:
                try:
                    await ctx.add_cookies(cookies)
                    logger.info("Resident: seeded %d cookies for %s", len(cookies), domain)
                except Exception as exc:
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
