"""ResidentBrowser — long-lived headed browser with per-domain contexts.

Used by the attended poller: one browser, one BrowserContext + tab per
domain. Captcha/login state persists for the lifetime of the process,
so challenges solved manually stay warm across scrapes.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from lib.browser.session_store import SessionStore

logger = logging.getLogger(__name__)


class ResidentBrowser:
    """One shared BrowserContext, one page (tab) per domain.

    Firefox renders each BrowserContext as a separate window, so we keep
    ONE context and add tabs into it. Cookies are shared across tabs, but
    each tab only navigates to its own domain — cross-contamination is
    minimal in practice.
    """

    def __init__(self, browser):
        self._browser = browser
        self._context = None  # created lazily on first page_for()
        self._pages: dict[str, Any] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._session_store = SessionStore()
        self._seeded_domains: set[str] = set()

    @property
    def browser(self):
        return self._browser

    def lock_for(self, domain: str) -> asyncio.Lock:
        if domain not in self._locks:
            self._locks[domain] = asyncio.Lock()
        return self._locks[domain]

    async def _ensure_context(self):
        if self._context is None:
            self._context = await self._browser.new_context()
        return self._context

    async def page_for(self, domain: str, seed_cookies: list[dict] | None = None):
        """Return the tab for `domain`, creating it on demand in the shared context."""
        if not domain:
            domain = "_default"
        ctx = await self._ensure_context()

        # Seed cookies once per domain into the shared context.
        if domain not in self._seeded_domains:
            cookies = seed_cookies or self._session_store.load(domain) or []
            if cookies:
                try:
                    await ctx.add_cookies(cookies)
                    logger.info("Resident: seeded %d cookies for %s", len(cookies), domain)
                except Exception as exc:
                    logger.warning("Resident: cookie seed failed for %s: %s", domain, exc)
            self._seeded_domains.add(domain)

        if domain not in self._pages:
            page = await ctx.new_page()
            self._pages[domain] = page
            logger.info("Resident: opened new tab for %s", domain)
        return self._pages[domain]

    async def preseed(self, domains: list[str]):
        """Open a tab per domain up-front and navigate to its homepage so the
        user can solve captchas / log in before scrapes start arriving.
        """
        for d in domains:
            try:
                page = await self.page_for(d)
                homepage = f"https://{d}/"
                try:
                    await page.goto(homepage, wait_until="domcontentloaded", timeout=30_000)
                    logger.info("Resident: preseed navigated to %s", homepage)
                except Exception as exc:
                    logger.info("Resident: preseed nav skipped for %s: %s", d, exc)
            except Exception as exc:
                logger.warning("Resident: preseed failed for %s: %s", d, exc)

    async def save_sessions(self) -> int:
        """Write current cookies back to SessionStore, one file per seeded
        domain. Called on shutdown so manually-solved logins persist across
        poller restarts. Returns the number of domains whose sessions were
        written.
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
                logger.info("Resident: no cookies to save for %s", domain)
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
        self._pages.clear()
        self._seeded_domains.clear()


def _cookie_matches_domain(cookie_domain: str, target: str) -> bool:
    """Match a Playwright cookie's domain attribute against our canonical
    target domain (e.g. 'linkedin.com'). Accepts '.linkedin.com',
    'www.linkedin.com', 'linkedin.com' — all three should save under the
    target.
    """
    if not cookie_domain or not target:
        return False
    cd = cookie_domain.lstrip(".").lower()
    tgt = target.lower()
    return cd == tgt or cd.endswith("." + tgt)
