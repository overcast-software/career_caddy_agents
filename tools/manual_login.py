#!/usr/bin/env python3
"""
Open a browser, wait for you to log in manually to each domain,
then save all session cookies when you're done.

Usage:
    python scripts/manual_login.py                  # opens to about:blank
    python scripts/manual_login.py monster.com dice.com   # opens each domain
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from browser.credentials import Credentials
from browser.engine import configure as configure_engine, get_engine, launch_browser
from browser.session_store import SessionStore


async def main(domains: list[str], engine: str) -> None:
    session_store = SessionStore()

    print(f"Starting browser (engine={engine})...")
    async with launch_browser(engine, headless=False) as browser:
        ctx = await browser.new_context()

        if domains:
            for domain in domains:
                url = f"https://{domain}"
                page = await ctx.new_page()
                await page.goto(url, wait_until="commit", timeout=60_000)
                print(f"  Opened {url}")
        else:
            page = await ctx.new_page()
            await page.goto("about:blank")
            print("  Opened blank tab")

        print()
        print("Log in to each site in the browser.")
        print("When you're done, press Enter here to save sessions and exit.")
        await asyncio.get_event_loop().run_in_executor(None, input, ">>> Press Enter when done: ")

        # Collect all cookies and bucket them by normalized domain
        all_cookies = await ctx.cookies()
        by_domain: dict[str, list[dict]] = {}
        for cookie in all_cookies:
            raw = cookie.get("domain", "").lstrip(".")
            if not raw:
                continue
            norm = Credentials.normalize_domain(raw)
            by_domain.setdefault(norm, []).append(cookie)

        if not by_domain:
            print("No cookies found — nothing saved.")
            return

        for domain, cookies in sorted(by_domain.items()):
            session_store.save(domain, cookies)
            print(f"  Saved {len(cookies):3d} cookies for {domain}")

        print(f"\nDone. Sessions saved to {session_store.sessions_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual browser login for session capture")
    parser.add_argument(
        "--engine", choices=["camoufox", "chrome"], default=None,
        help="Browser engine (default: BROWSER_ENGINE env or 'camoufox')",
    )
    parser.add_argument("domains", nargs="*", help="Domains to open (e.g. monster.com dice.com)")
    args = parser.parse_args()

    configure_engine(engine=args.engine)
    targets = [Credentials.normalize_domain(d) for d in args.domains]
    asyncio.run(main(targets, get_engine()))
