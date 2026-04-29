"""Persistent cookie session storage for browser automation.

Stores per-domain session cookies as JSON files in ~/.career_caddy/sessions/
using atomic writes to prevent corruption.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from browser.credentials import Credentials

logger = logging.getLogger(__name__)

_DEFAULT_SESSIONS_DIR = Path.home() / ".career_caddy" / "sessions"


class SessionStore:
    def __init__(self, sessions_dir: Path = _DEFAULT_SESSIONS_DIR) -> None:
        self.sessions_dir = sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, domain: str) -> Path:
        safe = Credentials.normalize_domain(domain).replace("/", "_")
        return self.sessions_dir / f"{safe}.json"

    def save(self, domain: str, cookies: list[dict]) -> None:
        """Atomically write cookies for a domain to disk."""
        path = self._path(domain)
        data = {
            "domain": Credentials.normalize_domain(domain),
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "cookie_count": len(cookies),
            "cookies": cookies,
        }
        tmp = path.with_suffix(".tmp")
        try:
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, path)
            logger.info(f"Saved {len(cookies)} session cookies for {domain}")
        except Exception as e:
            logger.error(f"Failed to save session for {domain}: {e}")
            tmp.unlink(missing_ok=True)

    def load(self, domain: str) -> list[dict] | None:
        """Load cookies for a domain. Returns None if no session exists."""
        path = self._path(domain)
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            cookies = data.get("cookies", [])
            logger.info(f"Loaded {len(cookies)} session cookies for {domain}")
            return cookies
        except Exception as e:
            logger.warning(f"Failed to load session for {domain}: {e}")
            return None

    def has_session(self, domain: str) -> bool:
        return self._path(domain).exists()

    def clear(self, domain: str) -> bool:
        """Delete saved session for a domain. Returns True if a file was removed."""
        path = self._path(domain)
        if path.exists():
            path.unlink()
            logger.info(f"Cleared session for {domain}")
            return True
        return False

    def list_domains(self) -> list[str]:
        """Return all domains with saved sessions."""
        return [p.stem for p in self.sessions_dir.glob("*.json")]
