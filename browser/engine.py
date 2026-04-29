"""Browser engine abstraction — Camoufox (Firefox) or Playwright Chromium + stealth.

Usage:
    from browser.engine import configure, launch_browser, get_engine, get_headless

    # At startup (CLI entry point):
    configure(engine="chrome", headless=True)

    # When you need a browser:
    async with launch_browser(get_engine(), get_headless()) as browser:
        ctx = await browser.new_context()
        page = await ctx.new_page()    # stealth auto-applied for chrome engine
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from playwright.async_api import Browser, BrowserContext, Page

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level overrides (set once at startup via configure())
# ---------------------------------------------------------------------------

_engine_override: Optional[str] = None
_headless_override: Optional[bool] = None


class BrowserEngineError(RuntimeError):
    """Raised when the selected browser engine cannot be launched."""


def configure(engine: str | None = None, headless: bool | None = None) -> None:
    """Set engine/headless overrides. Call once from CLI entry points."""
    global _engine_override, _headless_override
    if engine is not None:
        _engine_override = engine
    if headless is not None:
        _headless_override = headless


def get_engine() -> str:
    """CLI override -> BROWSER_ENGINE env -> 'camoufox'."""
    if _engine_override is not None:
        return _engine_override
    return os.environ.get("BROWSER_ENGINE", "camoufox")


def get_headless() -> bool:
    """CLI override -> BROWSER_HEADLESS env -> True."""
    if _headless_override is not None:
        return _headless_override
    return os.environ.get("BROWSER_HEADLESS", "true").lower() not in ("false", "0", "no")


def _get_proxy_config() -> dict | None:
    """Build a Playwright-compatible proxy dict from env vars.

    Env:
        BROWSER_PROXY_SERVER   — e.g. "socks5://localhost:1080" or "http://host:3128"
        BROWSER_PROXY_USERNAME — optional
        BROWSER_PROXY_PASSWORD — optional
        BROWSER_PROXY_BYPASS   — optional, comma-separated host list

    Caveat: Chromium (playwright) does NOT honor username/password for SOCKS
    proxies — auth only works for HTTP/HTTPS proxies. Firefox (and Camoufox)
    DO support SOCKS5 auth natively. Use --engine camoufox for authed SOCKS5.
    """
    server = os.environ.get("BROWSER_PROXY_SERVER")
    if not server:
        return None
    cfg: dict = {"server": server}
    user = os.environ.get("BROWSER_PROXY_USERNAME")
    password = os.environ.get("BROWSER_PROXY_PASSWORD")
    if user:
        cfg["username"] = user
    if password:
        cfg["password"] = password
    bypass = os.environ.get("BROWSER_PROXY_BYPASS")
    if bypass:
        cfg["bypass"] = bypass
    return cfg


# ---------------------------------------------------------------------------
# Stealth wrapper for Chromium — auto-applies playwright-stealth per page
# ---------------------------------------------------------------------------

class _StealthContext:
    """Wraps a BrowserContext so new_page() auto-applies stealth patches."""

    def __init__(self, ctx: BrowserContext) -> None:
        self._ctx = ctx

    async def new_page(self) -> Page:
        from playwright_stealth import stealth_async
        page = await self._ctx.new_page()
        await stealth_async(page)
        return page

    def __getattr__(self, name: str):
        return getattr(self._ctx, name)


class _StealthBrowser:
    """Wraps a Browser so new_context() returns _StealthContext."""

    def __init__(self, browser: Browser) -> None:
        self._browser = browser

    async def new_context(self, **kwargs) -> _StealthContext:
        ctx = await self._browser.new_context(**kwargs)
        return _StealthContext(ctx)

    def __getattr__(self, name: str):
        return getattr(self._browser, name)


# ---------------------------------------------------------------------------
# launch_browser — the single public async context manager
# ---------------------------------------------------------------------------

@asynccontextmanager
async def launch_browser(engine: str, headless: bool):
    """Yield a Playwright-compatible Browser for the chosen engine.

    For 'chrome', pages created via new_context().new_page() have
    playwright-stealth patches applied automatically.
    """
    if engine == "camoufox":
        async with _launch_camoufox(headless) as browser:
            yield browser
    elif engine == "chrome":
        async with _launch_chrome(headless) as browser:
            yield browser
    else:
        raise BrowserEngineError(f"Unknown browser engine: {engine!r}")


@asynccontextmanager
async def _launch_camoufox(headless: bool):
    """Launch Camoufox Firefox with anti-fingerprinting."""
    try:
        from camoufox.async_api import AsyncCamoufox
    except ImportError:
        raise BrowserEngineError(
            "camoufox package not installed. Install it or use --engine chrome."
        )
    try:
        from camoufox.exceptions import CamoufoxNotInstalled
    except ImportError:
        CamoufoxNotInstalled = Exception

    proxy = _get_proxy_config()
    if proxy:
        logger.info(
            "Starting Camoufox browser (headless=%s) via proxy %s",
            headless, proxy.get("server"),
        )
    else:
        logger.info("Starting Camoufox browser (headless=%s)", headless)
    camoufox_kwargs: dict = {"headless": headless}
    if proxy:
        camoufox_kwargs["proxy"] = proxy
    cm = AsyncCamoufox(**camoufox_kwargs)
    try:
        browser = await cm.__aenter__()
    except CamoufoxNotInstalled:
        raise BrowserEngineError(
            "Camoufox browser binary not found. Run: python -m camoufox fetch"
        )
    try:
        yield browser
    finally:
        await cm.__aexit__(None, None, None)


@asynccontextmanager
async def _launch_chrome(headless: bool):
    """Launch Playwright Chromium with stealth wrapper."""
    from playwright.async_api import async_playwright

    proxy = _get_proxy_config()
    if proxy:
        if proxy.get("server", "").startswith("socks") and (
            proxy.get("username") or proxy.get("password")
        ):
            raise BrowserEngineError(
                "Chromium does not support SOCKS proxy authentication. "
                "Use --engine camoufox (Firefox — default engine, supports "
                "SOCKS5 auth natively) or bridge the authed SOCKS5 through a "
                "local HTTP proxy (e.g. gost -L=http://:8080 "
                "-F=socks5://user:pass@localhost:1080) and set "
                "BROWSER_PROXY_SERVER=http://localhost:8080."
            )
        logger.info(
            "Starting Playwright Chromium (headless=%s) via proxy %s",
            headless, proxy.get("server"),
        )
    else:
        logger.info("Starting Playwright Chromium (headless=%s)", headless)
    pw = await async_playwright().start()
    try:
        launch_kwargs: dict = {"headless": headless}
        if proxy:
            launch_kwargs["proxy"] = proxy
        browser = await pw.chromium.launch(**launch_kwargs)
        yield _StealthBrowser(browser)
    finally:
        await pw.stop()
