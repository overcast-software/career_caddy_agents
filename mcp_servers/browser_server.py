#!/usr/bin/env python3
"""
Browser MCP server with pluggable engine: Camoufox (Firefox, anti-fingerprint)
or Playwright Chromium + stealth patches (ARM-compatible).

Engine selection: --engine camoufox|chrome (or BROWSER_ENGINE env var).

Maintains a single persistent browser instance and a tab registry.
Session cookies are persisted to disk (~/.career_caddy/sessions/) in
Playwright's universal format — sessions are portable across engines.

Tools:
    create_tab              — open a new tab, return tab_id
    navigate                — go to a URL, return title/status (auto-injects saved session)
    navigate_and_snapshot   — navigate + snapshot in one call (auto-injects saved session)
    snapshot                — return visible page text (token-efficient)
    screenshot              — save a PNG, return path
    get_links               — return all hrefs on the page
    click                   — click an element by CSS selector
    fill_form               — fill fields by CSS selector (generic)
    login_to_site           — inject stored credentials directly (never via LLM)
    ensure_authenticated    — high-level: inject session or auto-login, no selectors needed
    clear_session           — delete the saved session for a domain (force re-login)
    list_available_domains  — list domains with stored credentials
    close_tab               — close a tab (auto-saves session cookies)
    scrape_page             — one-shot: create tab → navigate → snapshot → close
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

import os

try:
    import logfire
    _logfire_available = True
except ImportError:
    _logfire_available = False

from playwright.async_api import Browser, BrowserContext, Page

from lib.browser.engine import (
    BrowserEngineError,
    configure as configure_engine,
    get_engine,
    get_headless,
    launch_browser,
)
from fastmcp import FastMCP

from lib.browser.credentials import Credentials
from lib.browser.firefox_cookies import load_cookies_for_domain
from lib.browser.session_store import SessionStore

if _logfire_available and os.environ.get("LOGFIRE_TOKEN"):
    logfire.configure(scrubbing=False, service_name="browser_mcp_server", console=False)
logging.basicConfig(level=logging.INFO)
logging.getLogger("fastmcp").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

from datetime import datetime
from pathlib import Path

SCREENSHOT_DIR = Path(os.environ.get("SCREENSHOT_DIR", "screenshots"))
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

LOGIN_WALL_SIGNALS = [
    "sign in", "log in", "login", "create an account",
    "forgot password", "enter your email", "join now",
    "access denied", "not in our system", "contact support",
    "continue to sign in",
]


def _is_headless() -> bool:
    return get_headless()


def _detect_login_wall(content: str) -> bool:
    stripped = content.strip().lower()
    word_count = len(stripped.split())
    return (
        word_count < 200
        and sum(1 for s in LOGIN_WALL_SIGNALS if s in stripped) >= 2
    )


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

try:
    credentials = Credentials.load()
    logger.info(
        f"Loaded credentials for {len(credentials.domains)} domains, "
        f"site configs for {len(credentials.site_configs)}"
    )
except FileNotFoundError:
    logger.warning("No credentials file found — running without saved credentials")
    credentials = Credentials(domains={})
except Exception as e:
    logger.error(f"Error loading credentials: {e}")
    credentials = Credentials(domains={})

session_store = SessionStore()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _domain_from_url(url: str) -> str:
    from urllib.parse import urlparse
    return Credentials.normalize_domain(urlparse(url).hostname or "")


def _resolve_tab_id(tab_id: str) -> str:
    """Accept either a raw tab_id string or the full create_tab JSON response.

    The LLM sometimes passes the entire JSON blob returned by create_tab
    (e.g. '{"tab_id": "140487689885008"}') instead of just the ID string.
    """
    try:
        parsed = json.loads(tab_id)
        if isinstance(parsed, dict) and "tab_id" in parsed:
            return str(parsed["tab_id"])
    except (json.JSONDecodeError, TypeError):
        pass
    return tab_id


async def _inject_session(ctx: BrowserContext, domain: str) -> int:
    """Inject saved session cookies for a domain into the context. Returns count injected."""
    if not domain:
        return 0
    cookies = session_store.load(domain)
    if cookies:
        await ctx.add_cookies(cookies)
        logger.info(f"Injected {len(cookies)} saved session cookies for {domain}")
        return len(cookies)
    return 0


async def _save_session(ctx: BrowserContext, domain: str) -> int:
    """Capture and persist all cookies for a domain. Returns count saved."""
    if not domain:
        return 0
    try:
        all_cookies = await ctx.cookies()
        domain_cookies = [
            c for c in all_cookies
            if Credentials.normalize_domain(c.get("domain", "")) == domain
        ]
        if domain_cookies:
            session_store.save(domain, domain_cookies)
        return len(domain_cookies)
    except Exception as e:
        logger.warning(f"Could not save session for {domain}: {e}")
        return 0


# ---------------------------------------------------------------------------
# Persistent browser session — lazy init, auto-recover on crash
# ---------------------------------------------------------------------------

_engine_cm = None  # async context manager from launch_browser
_browser: Optional[Browser] = None
_context: Optional[BrowserContext] = None
_tabs: dict[str, Page] = {}


async def _ensure_context() -> BrowserContext:
    global _engine_cm, _browser, _context
    if _context is not None and _browser is not None and _browser.is_connected():
        return _context

    engine = get_engine()
    headless = _is_headless()
    logger.info("Starting %s browser (headless=%s)", engine, headless)
    try:
        _engine_cm = launch_browser(engine, headless)
        _browser = await _engine_cm.__aenter__()
    except BrowserEngineError as exc:
        logging.critical(str(exc))
        raise SystemExit(1)
    _context = await _browser.new_context()
    return _context


async def _shutdown() -> None:
    global _engine_cm, _browser, _context
    if _context:
        await _context.close()
        _context = None
    if _engine_cm:
        await _engine_cm.__aexit__(None, None, None)
        _engine_cm = None
        _browser = None
    logger.info("Browser stopped")


# ---------------------------------------------------------------------------
# FastMCP server
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(server):
    yield
    await _shutdown()


server = FastMCP("browser-server", lifespan=_lifespan)


@server.tool()
async def create_tab() -> str:
    """Open a new browser tab. Returns tab_id used by all other tools."""
    ctx = await _ensure_context()
    page = await ctx.new_page()
    tab_id = str(id(page))
    _tabs[tab_id] = page
    return json.dumps({"tab_id": tab_id})


@server.tool()
async def navigate(tab_id: str, url: str) -> str:
    """Navigate a tab to a URL. Automatically injects saved session cookies for the
    domain so previously authenticated sessions are restored transparently.

    Args:
        tab_id: Tab ID from create_tab.
        url: Full URL including protocol.
    """
    tab_id = _resolve_tab_id(tab_id)
    page = _tabs.get(tab_id)
    if page is None:
        return json.dumps({"error": f"Unknown tab_id: {tab_id}"})
    try:
        ctx = await _ensure_context()
        domain = _domain_from_url(url)
        await _inject_session(ctx, domain)
        with logfire.span("browser.navigate", tab_id=tab_id, url=url):
            resp = await page.goto(url, wait_until="domcontentloaded", timeout=60_000)
            await asyncio.sleep(1)
            try:
                await page.wait_for_load_state("networkidle", timeout=15_000)
            except Exception:
                pass  # networkidle timeout is non-fatal; content may still be usable
        return json.dumps(
            {
                "title": await page.title(),
                "url": page.url,
                "status": resp.status if resp else None,
            }
        )
    except Exception as e:
        return json.dumps({"error": str(e)})


@server.tool()
async def navigate_and_snapshot(tab_id: str, url: str) -> str:
    """Navigate to a URL and return the visible page text in one call.
    Automatically injects saved session cookies for the domain.

    Args:
        tab_id: Tab ID from create_tab.
        url: Full URL including protocol.
    """
    tab_id = _resolve_tab_id(tab_id)
    page = _tabs.get(tab_id)
    if page is None:
        return json.dumps({"error": f"Unknown tab_id: {tab_id}"})
    try:
        ctx = await _ensure_context()
        domain = _domain_from_url(url)
        await _inject_session(ctx, domain)
        with logfire.span("browser.navigate_and_snapshot", tab_id=tab_id, url=url):
            resp = await page.goto(url, wait_until="domcontentloaded", timeout=60_000)
            await asyncio.sleep(1)
            try:
                await page.wait_for_load_state("networkidle", timeout=15_000)
            except Exception:
                pass
            text = await page.inner_text("body")
        return json.dumps(
            {
                "title": await page.title(),
                "url": page.url,
                "status": resp.status if resp else None,
                "content": text,
            }
        )
    except Exception as e:
        return json.dumps({"error": str(e)})


@server.tool()
async def snapshot(tab_id: str) -> str:
    """Return the visible text content of the current page (token-efficient).

    Args:
        tab_id: Tab ID from create_tab.
    """
    tab_id = _resolve_tab_id(tab_id)
    page = _tabs.get(tab_id)
    if page is None:
        return json.dumps({"error": f"Unknown tab_id: {tab_id}"})
    try:
        return json.dumps(
            {
                "title": await page.title(),
                "url": page.url,
                "content": await page.inner_text("body"),
            }
        )
    except Exception as e:
        return json.dumps({"error": str(e)})


@server.tool()
async def get_links(tab_id: str) -> str:
    """Return all hyperlinks on the current page.

    Args:
        tab_id: Tab ID from create_tab.
    """
    tab_id = _resolve_tab_id(tab_id)
    page = _tabs.get(tab_id)
    if page is None:
        return json.dumps({"error": f"Unknown tab_id: {tab_id}"})
    try:
        links = await page.evaluate(
            "() => Array.from(document.querySelectorAll('a[href]'))"
            ".map(a => ({text: a.innerText.trim(), href: a.href}))"
        )
        return json.dumps({"links": links})
    except Exception as e:
        return json.dumps({"error": str(e)})


@server.tool()
async def click(tab_id: str, selector: str) -> str:
    """Click an element by CSS selector.

    Args:
        tab_id: Tab ID from create_tab.
        selector: CSS selector, e.g. 'button.see-more' or 'text=See more'.
    """
    tab_id = _resolve_tab_id(tab_id)
    page = _tabs.get(tab_id)
    if page is None:
        return json.dumps({"error": f"Unknown tab_id: {tab_id}"})
    try:
        await page.click(selector, timeout=5_000)
        return json.dumps({"clicked": selector})
    except Exception as e:
        return json.dumps({"error": str(e)})


@server.tool()
async def fill_form(tab_id: str, fields: list[dict]) -> str:
    """Fill form fields by CSS selector.

    Args:
        tab_id: Tab ID from create_tab.
        fields: List of dicts, each with "selector" (CSS selector string) and
            "value" (text to type). Both keys are required per entry.
            Example: [{"selector": "input[name=email]", "value": "user@example.com"},
                      {"selector": "input[name=password]", "value": "secret"}]
    """
    tab_id = _resolve_tab_id(tab_id)
    page = _tabs.get(tab_id)
    if page is None:
        return json.dumps({"error": f"Unknown tab_id: {tab_id}"})
    try:
        for f in fields:
            if "selector" in f and "value" in f:
                await page.fill(f["selector"], f["value"])
        return json.dumps({"filled": len(fields)})
    except Exception as e:
        return json.dumps({"error": str(e)})


@server.tool()
async def login_to_site(
    tab_id: str,
    domain: str,
    username_selector: str,
    password_selector: str,
    submit_selector: Optional[str] = None,
) -> str:
    """Fill a login form using stored credentials without exposing them to the LLM.

    Credentials are injected directly via Playwright — they never appear in any
    tool result, model input, or response.

    Args:
        tab_id: Tab ID from create_tab.
        domain: Domain key (e.g. 'linkedin.com') to look up stored credentials.
        username_selector: CSS selector for the username/email field.
        password_selector: CSS selector for the password field.
        submit_selector: Optional CSS selector for the submit button.
    """
    tab_id = _resolve_tab_id(tab_id)
    page = _tabs.get(tab_id)
    if page is None:
        return json.dumps({"error": f"Unknown tab_id: {tab_id}"})

    creds = credentials.get_credentials(domain)
    if not creds:
        return json.dumps({"error": f"No credentials configured for {domain}"})

    username = creds.get("username") or creds.get("email", "")
    password = creds.get("password", "")
    if not username or not password:
        return json.dumps({"error": f"Incomplete credentials for {domain}"})

    try:
        await page.fill(username_selector, username)
        await page.fill(password_selector, password)
        if submit_selector:
            await page.click(submit_selector)
            await page.wait_for_load_state("domcontentloaded")
        # Persist session so future navigations skip login
        ctx = await _ensure_context()
        saved = await _save_session(ctx, Credentials.normalize_domain(domain))
        return json.dumps({"status": f"Login form filled for {domain}", "session_cookies_saved": saved})
    except Exception as e:
        return json.dumps({"error": str(e)})


@server.tool()
async def get_form_fields(tab_id: str) -> str:
    """Return all form inputs and submit buttons on the current page with CSS selectors.

    Call this before login_to_site to identify the correct selectors for the
    username and password fields.

    Args:
        tab_id: Tab ID from create_tab.
    """
    tab_id = _resolve_tab_id(tab_id)
    page = _tabs.get(tab_id)
    if page is None:
        return json.dumps({"error": f"Unknown tab_id: {tab_id}"})
    try:
        fields = await page.evaluate(
            """() =>
            Array.from(document.querySelectorAll('input, button[type=submit], [role=button]'))
            .map(el => {
                const sel = el.id ? '#' + el.id
                    : el.name ? '[name="' + el.name + '"]'
                    : el.getAttribute('aria-label') ? '[aria-label="' + el.getAttribute('aria-label') + '"]'
                    : null;
                return {
                    tag: el.tagName.toLowerCase(),
                    type: el.type || null,
                    name: el.name || null,
                    id: el.id || null,
                    placeholder: el.placeholder || null,
                    aria_label: el.getAttribute('aria-label'),
                    selector: sel,
                };
            })
        """
        )
        return json.dumps({"fields": fields})
    except Exception as e:
        return json.dumps({"error": str(e)})


@server.tool()
async def inject_firefox_cookies(domain: str) -> str:
    """Load cookies from the local Firefox profile and inject them into the
    browser context. Works with any engine (Camoufox or Chromium) — the
    cookies are converted to Playwright's universal format.

    Requires Firefox to be installed on the host (reads cookies.sqlite).
    On systems without Firefox (e.g. Raspberry Pi), use manual_login.py
    to seed sessions instead — those are saved to ~/.career_caddy/sessions/
    and auto-injected on navigation regardless of engine.

    Call this before navigate/navigate_and_snapshot to skip login entirely.

    Args:
        domain: Domain to pull cookies for, e.g. 'toptal.com'.
                Scheme and www. prefix are stripped automatically.
    """
    try:
        ctx = await _ensure_context()
        cookies = load_cookies_for_domain(domain)
        if not cookies:
            return json.dumps({"error": f"No Firefox cookies found for {domain}"})
        await ctx.add_cookies(cookies)
        return json.dumps({"injected": len(cookies), "domain": domain})
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": str(e)})


@server.tool()
async def list_available_domains() -> str:
    """Return domains that have login credentials configured."""
    return json.dumps({"domains": list(credentials.domains.keys())})


@server.tool()
async def ensure_authenticated(tab_id: str, domain: str) -> str:
    """Ensure the tab is authenticated for a domain — no selectors needed.

    Workflow:
    1. Inject saved session cookies (if any).
    2. Navigate to the domain's configured login_url.
    3. If post_login_check selector is present on the page → already authenticated.
    4. Otherwise, if credentials + selectors are configured in secrets.yml,
       fill the login form and save the resulting session cookies.
    5. Return status without ever exposing credentials or cookie values.

    Requires login_url, username_selector, and password_selector to be set
    in secrets.yml for the domain. submit_selector and post_login_check are
    optional but recommended.

    Args:
        tab_id: Tab ID from create_tab.
        domain: Domain to authenticate (e.g. 'linkedin.com').
    """
    tab_id = _resolve_tab_id(tab_id)
    page = _tabs.get(tab_id)
    if page is None:
        return json.dumps({"error": f"Unknown tab_id: {tab_id}"})

    normalized = Credentials.normalize_domain(domain)
    ctx = await _ensure_context()

    # Step 1: Inject saved session
    injected = await _inject_session(ctx, normalized)

    login_cfg = credentials.get_login_config(domain)
    if login_cfg is None:
        if injected:
            return json.dumps({"authenticated": True, "method": "session"})
        return json.dumps({"authenticated": False, "method": "none",
                           "reason": "No login config in secrets.yml"})

    # Step 2: Navigate to login URL
    try:
        await page.goto(login_cfg.login_url, wait_until="domcontentloaded", timeout=60_000)
        await asyncio.sleep(1)
        try:
            await page.wait_for_load_state("networkidle", timeout=10_000)
        except Exception:
            pass
    except Exception as e:
        return json.dumps({"error": f"Failed to navigate to login page: {e}"})

    # Step 3: Check if already authenticated
    if login_cfg.post_login_check:
        try:
            await page.wait_for_selector(login_cfg.post_login_check, timeout=3_000)
            return json.dumps({"authenticated": True, "method": "session"})
        except Exception:
            pass  # Not found → need to log in

    # Step 4: Log in with stored credentials
    creds = credentials.get_credentials(domain)
    username = creds.get("username") or creds.get("email", "")
    password = creds.get("password", "")
    if not username or not password:
        return json.dumps({"authenticated": False, "method": "none",
                           "reason": f"Incomplete credentials for {normalized}"})

    try:
        await page.fill(login_cfg.username_selector, username)
        await page.fill(login_cfg.password_selector, password)
        if login_cfg.submit_selector:
            await page.click(login_cfg.submit_selector)
            await page.wait_for_load_state("domcontentloaded", timeout=15_000)
            await asyncio.sleep(1)
            try:
                await page.wait_for_load_state("networkidle", timeout=10_000)
            except Exception:
                pass
    except Exception as e:
        return json.dumps({"error": f"Login interaction failed: {e}"})

    # Verify login succeeded if we have a check selector
    if login_cfg.post_login_check:
        try:
            await page.wait_for_selector(login_cfg.post_login_check, timeout=5_000)
        except Exception:
            return json.dumps({"authenticated": False, "method": "login",
                               "reason": "post_login_check selector not found after login"})

    saved = await _save_session(ctx, normalized)
    return json.dumps({"authenticated": True, "method": "login", "session_cookies_saved": saved})


@server.tool()
async def clear_session(domain: str) -> str:
    """Delete the saved session for a domain, forcing re-login on next visit.

    Args:
        domain: Domain whose session should be cleared (e.g. 'linkedin.com').
    """
    normalized = Credentials.normalize_domain(domain)
    removed = session_store.clear(normalized)
    if removed:
        return json.dumps({"cleared": normalized})
    return json.dumps({"cleared": None, "reason": f"No saved session for {normalized}"})


@server.tool()
async def close_tab(tab_id: str) -> str:
    """Close a browser tab and free its resources. Saves session cookies first.

    Args:
        tab_id: Tab ID from create_tab.
    """
    tab_id = _resolve_tab_id(tab_id)
    page = _tabs.pop(tab_id, None)
    if page is None:
        return json.dumps({"error": f"Unknown tab_id: {tab_id}"})
    # Best-effort session save before closing
    try:
        domain = _domain_from_url(page.url)
        if domain:
            ctx = await _ensure_context()
            await _save_session(ctx, domain)
    except Exception:
        pass
    await page.close()
    return json.dumps({"closed": tab_id})


@server.tool()
async def screenshot(tab_id: str, full_page: bool = False) -> str:
    """Save a PNG screenshot of the current page, return the file path.

    Args:
        tab_id: Tab ID from create_tab.
        full_page: If True, capture the full scrollable page. Default: viewport only.
    """
    tab_id = _resolve_tab_id(tab_id)
    page = _tabs.get(tab_id)
    if page is None:
        return json.dumps({"error": f"Unknown tab_id: {tab_id}"})

    domain = _domain_from_url(page.url)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SCREENSHOT_DIR / f"{domain}_{ts}.png"
    await page.screenshot(path=str(path), full_page=full_page)
    return json.dumps({"path": str(path), "domain": domain})


@server.tool()
async def scrape_page(url: str) -> str:
    """Navigate to a URL and return all visible text in one call.

    Uses BROWSER_HEADLESS env var (default true). Automatically injects
    saved session cookies for the URL's domain. Detects login walls and
    returns an error instead of garbage content. Captures a screenshot
    on every scrape (saved to SCREENSHOT_DIR).

    Args:
        url: Full URL including protocol.
    """
    from urllib.parse import urlparse
    raw_domain = urlparse(url).hostname or ""
    norm_domain = Credentials.normalize_domain(raw_domain) if raw_domain else ""
    cookies: list[dict] = []
    if raw_domain:
        saved = session_store.load(norm_domain) if norm_domain else None
        if saved:
            cookies = saved
            logfire.info(f"loaded {len(cookies)} saved session cookies for {norm_domain}")
        else:
            try:
                cookies = load_cookies_for_domain(raw_domain)
                if cookies:
                    logfire.info(f"loaded {len(cookies)} Firefox cookies for {raw_domain}")
            except Exception as e:
                logfire.warn(f"could not load Firefox cookies for {raw_domain}: {e}")

    headless = _is_headless()
    engine = get_engine()
    try:
        async with launch_browser(engine, headless) as browser:
            ctx = await browser.new_context()
            if cookies:
                await ctx.add_cookies(cookies)
            page = await ctx.new_page()
            try:
                with logfire.span("browser.scrape_page", url=url):
                    logfire.info(f"loading page (engine={engine}, headless={headless})")
                    await page.goto(url, wait_until="domcontentloaded", timeout=60_000)
                    await asyncio.sleep(1)
                    try:
                        await page.wait_for_load_state("networkidle", timeout=15_000)
                    except Exception:
                        pass
                    content = ""
                    for _ in range(3):
                        content = await page.inner_text("body")
                        stripped = content.strip().lower()
                        if stripped and not stripped.startswith("loading"):
                            break
                        logfire.info("page still loading, waiting...")
                        await asyncio.sleep(2)
                    logfire.info("finished loading")

                    # Capture screenshot
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    screenshot_name = f"{norm_domain or 'unknown'}_{ts}.png"
                    screenshot_path = SCREENSHOT_DIR / screenshot_name
                    try:
                        await page.screenshot(path=str(screenshot_path), full_page=False)
                        logfire.info(f"screenshot saved: {screenshot_path}")
                    except Exception as e:
                        logfire.warn(f"screenshot failed: {e}")
                        screenshot_name = None

                    # Detect login walls
                    if _detect_login_wall(content):
                        word_count = len(content.strip().split())
                        return json.dumps({
                            "title": await page.title(),
                            "url": page.url,
                            "content": "",
                            "error": "login_wall_detected",
                            "message": (
                                f"Page appears to be a login wall ({word_count} words, "
                                "login signals found). Use ensure_authenticated or "
                                "manual_login.py to seed session cookies for this domain."
                            ),
                            "screenshot": screenshot_name,
                        })

                return json.dumps(
                    {
                        "title": await page.title(),
                        "url": page.url,
                        "content": content,
                        "screenshot": screenshot_name,
                    }
                )
            except Exception as e:
                return json.dumps({"error": str(e)})
    except BrowserEngineError as exc:
        return json.dumps({"error": str(exc)})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Browser MCP server")
    parser.add_argument(
        "--engine", choices=["camoufox", "chrome"], default=None,
        help="Browser engine (default: BROWSER_ENGINE env or 'camoufox')",
    )
    parser.add_argument("--headless", action="store_true", default=None, help="Run headless")
    parser.add_argument("--headed", dest="headless", action="store_false", help="Run headed")
    args = parser.parse_args()

    configure_engine(engine=args.engine, headless=args.headless)
    server.run(transport="sse", host="0.0.0.0", port=3004)
