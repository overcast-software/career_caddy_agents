"""
Read session cookies from a local Firefox profile and convert them to
Playwright's add_cookies() format. Output works with any Playwright-based
engine (Camoufox or Chromium).

Firefox stores cookies in an SQLite database (cookies.sqlite).  The file is
locked while Firefox is running, so we copy it to a temp file first.

Requires Firefox to be installed on the host. On systems without Firefox
(e.g. Raspberry Pi), use manual_login.py to seed sessions instead.

Usage:
    from lib.browser.firefox_cookies import load_cookies_for_domain

    cookies = load_cookies_for_domain("toptal.com")
    await context.add_cookies(cookies)
"""

import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

# Firefox sameSite integer → Playwright string
_SAME_SITE = {0: "None", 1: "Lax", 2: "Strict"}

# Common Firefox profile base directories, checked in order
_PROFILE_BASES = [
    Path.home() / ".mozilla" / "firefox",
    # Snap-packaged Firefox on Ubuntu
    Path.home() / "snap" / "firefox" / "common" / ".mozilla" / "firefox",
    # macOS
    Path.home() / "Library" / "Application Support" / "Firefox" / "Profiles",
]


def find_firefox_cookies_db() -> Optional[Path]:
    """Return the path to the most likely Firefox cookies.sqlite.

    Prefers the 'default-release' profile, then any profile that has the file.
    Returns None if no Firefox profile is found.
    """
    for base in _PROFILE_BASES:
        if not base.exists():
            continue
        profiles = [p for p in base.iterdir() if p.is_dir()]
        # Prefer default-release, then default, then any
        ordered = (
            [p for p in profiles if "default-release" in p.name]
            + [p for p in profiles if "default" in p.name and "release" not in p.name]
            + [p for p in profiles if "default-release" not in p.name and "default" not in p.name]
        )
        for profile in ordered:
            db = profile / "cookies.sqlite"
            if db.exists():
                return db
    return None


def _normalize_domain(domain: str) -> str:
    """Strip leading dot and scheme, return bare domain (e.g. 'toptal.com')."""
    if domain.startswith(("http://", "https://")):
        domain = urlparse(domain).hostname or domain
    return domain.lstrip(".")


def load_cookies_for_domain(
    domain: str,
    db_path: Optional[Path] = None,
) -> list[dict]:
    """Load Firefox cookies for *domain* and return them in Playwright format.

    Both host-only cookies (stored as 'toptal.com') and domain cookies (stored
    as '.toptal.com') are returned.  Subdomains are included automatically
    (e.g. 'www.toptal.com' cookies are included when domain='toptal.com').

    Args:
        domain: Domain to fetch cookies for, e.g. 'toptal.com' or
                'https://www.toptal.com/jobs/...' (scheme/www stripped).
        db_path: Path to cookies.sqlite.  Auto-detected if omitted.

    Returns:
        List of cookie dicts accepted by Playwright's context.add_cookies().

    Raises:
        FileNotFoundError: If no Firefox profile / cookies.sqlite can be found.
    """
    bare = _normalize_domain(domain)
    db_path = db_path or find_firefox_cookies_db()
    if db_path is None:
        raise FileNotFoundError(
            "Could not find Firefox cookies.sqlite. "
            "Pass db_path= explicitly or check that Firefox is installed."
        )

    # Copy to a temp file — Firefox holds a WAL lock while running
    tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    try:
        shutil.copy2(db_path, tmp_path)
        return _query_cookies(tmp_path, bare)
    finally:
        tmp_path.unlink(missing_ok=True)


def _query_cookies(db: Path, bare_domain: str) -> list[dict]:
    """Query cookies.sqlite for bare_domain and all its subdomains."""
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        # Match 'toptal.com', '.toptal.com', and any subdomain like 'app.toptal.com'
        rows = conn.execute(
            """
            SELECT name, value, host, path, expiry, isSecure, isHttpOnly, sameSite
            FROM moz_cookies
            WHERE host = ?
               OR host = '.' || ?
               OR host LIKE '%.' || ?
            """,
            (bare_domain, bare_domain, bare_domain),
        ).fetchall()
    finally:
        conn.close()

    cookies = []
    for row in rows:
        cookies.append({
            "name": row["name"],
            "value": row["value"],
            "domain": row["host"],          # keep leading dot if present
            "path": row["path"],
            "expires": int(row["expiry"]) if row["expiry"] and row["expiry"] > 0 else -1,
            "httpOnly": bool(row["isHttpOnly"]),
            "secure": bool(row["isSecure"]),
            "sameSite": _SAME_SITE.get(row["sameSite"], "None"),
        })
    return cookies


if __name__ == "__main__":
    import json
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "toptal.com"
    found = find_firefox_cookies_db()
    print(f"Profile db: {found}", file=sys.stderr)

    cookies = load_cookies_for_domain(target)
    print(f"Found {len(cookies)} cookies for {target}", file=sys.stderr)
    print(json.dumps(cookies, indent=2))
