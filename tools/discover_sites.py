#!/usr/bin/env python3
"""
One-time script to discover and populate login metadata in sites.yml.

For each domain in secrets.yml that is missing from sites.yml, this script:
  1. Navigates to the site's homepage to find the login link
  2. Inspects the login form fields
  3. Actually logs in (using credentials from secrets.yml)
  4. Identifies a CSS selector that is only present when authenticated
  5. Writes the result to sites.yml

Requires browser_server.py to be running:
    python mcp_servers/browser_server.py

Usage:
    python scripts/discover_sites.py                  # all domains missing from sites.yml
    python scripts/discover_sites.py linkedin.com     # specific domain(s)
    python scripts/discover_sites.py --dry-run        # show what would run, don't write
"""

import argparse
import asyncio
import difflib
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import yaml
import logfire
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.mcp import MCPServerSSE

# Project root so imports work when run from any directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from browser.credentials import Credentials, SiteConfig

logfire.configure(service_name="discover_sites", console=False)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SITES_PATH = _PROJECT_ROOT / "sites.yml"
_BROWSER_SSE = "http://localhost:3004/sse"

# YAML notes file (per-domain append-only notes)
_NOTES_PATH = _PROJECT_ROOT / "notes.yml"

def _load_notes() -> dict:
    if not _NOTES_PATH.exists():
        return {}
    data = yaml.safe_load(_NOTES_PATH.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}

def _save_notes(notes: dict) -> None:
    _NOTES_PATH.write_text(
        yaml.safe_dump(notes, allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Structured output the agent produces
# ---------------------------------------------------------------------------

class DiscoveredSiteConfig(BaseModel):
    login_url: str
    username_selector: str
    password_selector: str
    submit_selector: Optional[str] = None
    post_login_check: Optional[str] = None
    notes: Optional[list[str]] = None



# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def _build_discovery_agent(browser_server: MCPServerSSE) -> Agent:
    return Agent(
        "openai:gpt-4o-mini",
        output_type=DiscoveredSiteConfig,
        toolsets=[browser_server],
        system_prompt="""
You are a browser automation expert discovering login metadata for a website.

Your goal is to find:
- login_url: the direct URL of the login page
- username_selector: CSS selector for the email/username input
- password_selector: CSS selector for the password input
- submit_selector: CSS selector for the submit button (if present)
- post_login_check: a CSS selector that is ONLY present when the user is logged in
  (e.g. a profile avatar, nav item, or logout button — something unique to authenticated state)

## Workflow

0. read_site_notes(domain) — review any prior findings, known selectors, quirks, and failure modes.
1. create_tab() — get a tab_id
2. navigate(tab_id, "https://<domain>") — load the homepage
3. determine if already logged in.  Sessions are saved and your are rooting out the ones that need login sessions.
4. get_links(tab_id) — scan for a login or sign-in link; navigate to it
5. get_form_fields(tab_id) — identify username, password, and submit selectors
6. login_to_site(tab_id, domain, username_selector, password_selector, submit_selector)
   — log in using stored credentials (never type credentials yourself)
7. snapshot(tab_id) — look at the authenticated page; identify a stable selector
   for post_login_check that will NOT appear on a logged-out page
8. Check the login state again. the browser is run in a headed environment and the end user can login and sovle captchas
9. If on a login page, wait 30 seconds.  give the end user a change to offer the credentials
10. check if the browser is logged in. look for signs that it's there.
11. close_tab(tab_id)
12. Return a DiscoveredSiteConfig with all fields filled in, including notes that briefly explain how you determined authentication (e.g., "Found selector .global-nav__me; redirected to /feed; user menu visible").
13. append_site_note(domain, "<concise summary of what you found or attempted>", ["discover_sites"]) — append a brief status including any reliable selectors, unresolved issues, or captcha/manual steps.

## Rules
- NEVER type, display, or log credential values. Use login_to_site for all login actions.
- NEVER put user@example.com for the username.  It's never correct.
- Prefer IDs (#id) over class selectors for username/password fields — they're more stable.
- If a submit button lacks an id, use button[type=submit] or a specific class.
- For post_login_check, pick something in the navigation or header that won't appear when
  logged out (e.g. a profile link, avatar container, or user menu button).
- If you cannot determine a selector with confidence, set it to null.
- After confirming you are logged in, include 1–3 concise bullet points in the notes field describing the evidence of authentication (unique selectors, redirects, title changes, etc.).
- You must call read_site_notes at the beginning, before browsing, and append_site_note at the end to record outcomes for future runs.
""",
    )


def _register_tools(agent: Agent) -> None:
    """Register all tools on the agent instance."""
    
    @agent.tool
    async def read_site_notes(ctx: RunContext, domain: str) -> str:
    """
    Read prior notes for the given domain from notes.yml.
    Returns a YAML document with only this domain's notes (or empty string if none).
    """
    key = Credentials.normalize_domain(domain)
    notes = _load_notes()
    entries = notes.get(key) or []
    if not entries:
        return ""
        return yaml.safe_dump({key: entries}, allow_unicode=True, sort_keys=True)

    @agent.tool
    async def append_site_note(ctx: RunContext, domain: str, note: str, tags: Optional[list[str]] = None) -> str:
    """
    Append a note entry for the domain into notes.yml with a UTC timestamp.
    """
    key = Credentials.normalize_domain(domain)
    notes = _load_notes()
    if key not in notes or notes[key] is None:
        notes[key] = []
    entry = {"timestamp": _now_iso(), "note": note}
    if tags:
        entry["tags"] = tags
    notes[key].append(entry)
        _save_notes(notes)
        return f"wrote note for {key}; total={len(notes[key])}"


# ---------------------------------------------------------------------------
# sites.yml helpers
# ---------------------------------------------------------------------------

def _load_sites() -> dict:
    if not _SITES_PATH.exists():
        return {}
    with open(_SITES_PATH) as f:
        return yaml.safe_load(f) or {}


def _save_sites(sites: dict) -> None:
    # Preserve the header comment by reading raw content first
    existing_raw = _SITES_PATH.read_text() if _SITES_PATH.exists() else ""
    comment_lines = [l for l in existing_raw.splitlines() if l.startswith("#")]
    header = "\n".join(comment_lines) + "\n" if comment_lines else ""

    body = yaml.dump(sites, default_flow_style=False, allow_unicode=True, sort_keys=True)
    _SITES_PATH.write_text(header + body)
    logger.info(f"Wrote {_SITES_PATH}")


def _extract_host(s: str) -> str:
    s = s.strip()
    if not s:
        return ""
    if "://" in s:
        try:
            host = urlparse(s).hostname or s
        except Exception:
            host = s
    else:
        host = s.split("/")[0]
    return host.lower()


def _resolve_credential_domain(host: str, creds: Credentials) -> Optional[str]:
    """
    Map a host (possibly a subdomain) to the best matching credentials domain.
    Strategy:
      - Exact match
      - Parent match (host endswith '.' + d)
      - Child match (d endswith '.' + host)
    Choose the most specific (longest) matching domain.
    """
    host = Credentials.normalize_domain(host)
    domains = list(creds.domains.keys())

    # Exact match first
    if host in domains:
        return host

    candidates: list[str] = []
    for d in domains:
        if host.endswith("." + d) or d.endswith("." + host):
            candidates.append(d)

    if not candidates:
        return None

    # Prefer the most specific (longest) candidate
    return max(candidates, key=len)


# ---------------------------------------------------------------------------
# Discovery per domain
# ---------------------------------------------------------------------------

async def discover_domain(target_host: str, credential_domain: str, agent: Agent) -> Optional[SiteConfig]:
    logger.info(f"Discovering login config for {target_host} (credentials: {credential_domain}) …")
    home_url = f"https://{target_host}"
    result = await agent.run(
        "Discover the login metadata.\n"
        f"- Start by navigating to: {home_url}\n"
        f"- When calling login_to_site, use domain: {credential_domain}\n"
        f"- The site may redirect; follow as needed."
    )
    d = result.output
    logger.info(f"  login_url:          {d.login_url}")
    logger.info(f"  username_selector:  {d.username_selector}")
    logger.info(f"  password_selector:  {d.password_selector}")
    logger.info(f"  submit_selector:    {d.submit_selector}")
    logger.info(f"  post_login_check:   {d.post_login_check}")
    return SiteConfig(
        login_url=d.login_url,
        username_selector=d.username_selector,
        password_selector=d.password_selector,
        submit_selector=d.submit_selector,
        post_login_check=d.post_login_check,
        notes=d.notes,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("domains", nargs="*", help="Domains to discover (default: all missing from sites.yml)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would run without writing")
    args = parser.parse_args()

    creds = Credentials.load()
    sites = _load_sites()

    if args.domains:
        raw_targets = [d for d in args.domains]
        targets: list[tuple[str, Optional[str]]] = []
        for raw in raw_targets:
            host = _extract_host(raw).lstrip(".")
            if not host:
                continue
            cred_domain = _resolve_credential_domain(host, creds)
            targets.append((host, cred_domain))
    else:
        # All credential domains not yet in sites.yml
        targets = [(d, d) for d in creds.domains if d not in sites]

    if not targets:
        logger.info("Nothing to discover — all credential domains already have site configs.")
        return

    logger.info("Will discover: " + ", ".join([t[0] for t in targets]))

    if args.dry_run:
        logger.info("Dry run — exiting without making changes.")
        return

    async with MCPServerSSE(_BROWSER_SSE) as browser_server:
        discovery_agent = _build_discovery_agent(browser_server)
        _register_tools(discovery_agent)
        
        for host, cred_domain in targets:
            if cred_domain is None:
                base = Credentials.normalize_domain(host)
                domains = list(creds.domains.keys())
                suggestion = ""
                matches = difflib.get_close_matches(host, domains, n=3, cutoff=0.6)
                if matches:
                    suggestion = f" Did you mean one of: {', '.join(matches)}?"
                logger.warning(
                    f"  {host}: no credentials in secrets.yml (looked for related to '{base}'), skipping.{suggestion}"
                )
                continue
            try:
                cfg = await discover_domain(host, cred_domain, discovery_agent)
                if cfg:
                    new_cfg = cfg.to_dict()
                    # Merge/append notes with any existing notes for this domain
                    existing_notes = list((sites.get(cred_domain, {}) or {}).get("notes") or [])
                    new_notes = list(new_cfg.get("notes") or [])
                    merged_notes = existing_notes + new_notes if (existing_notes or new_notes) else None
                    if merged_notes:
                        new_cfg["notes"] = merged_notes
                    sites[cred_domain] = new_cfg
                    _save_sites(sites)
            except Exception as e:
                logger.error(f"  {host}: discovery failed — {e}")

    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
