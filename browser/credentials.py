from pathlib import Path
from urllib.parse import urlparse
import yaml

from dataclasses import dataclass, field
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class SiteConfig:
    """Non-secret site metadata for login automation, stored in sites.yml.

    Example sites.yml entry::

        linkedin.com:
          login_url: https://www.linkedin.com/login
          username_selector: "#username"
          password_selector: "#password"
          submit_selector: ".login__form_action_container button"
          post_login_check: ".global-nav__me"
    """

    login_url: str
    username_selector: str
    password_selector: str
    submit_selector: Optional[str] = None
    post_login_check: Optional[str] = None  # CSS selector present only when logged in
    notes: Optional[list[str]] = None

    def to_dict(self) -> dict:
        d = {
            "login_url": self.login_url,
            "username_selector": self.username_selector,
            "password_selector": self.password_selector,
        }
        if self.submit_selector:
            d["submit_selector"] = self.submit_selector
        if self.post_login_check:
            d["post_login_check"] = self.post_login_check
        if self.notes:
            d["notes"] = self.notes
        return d


# Backwards-compatible alias
LoginConfig = SiteConfig


@dataclass
class Credentials:
    """Credentials loaded from secrets.yml, optionally merged with sites.yml metadata."""

    domains: dict[str, dict[str, str]]  # domain -> {username, password}
    site_configs: dict[str, SiteConfig] = field(default_factory=dict)  # domain -> SiteConfig

    @staticmethod
    def normalize_domain(domain: str) -> str:
        """Normalize domain by removing www and other subdomains.

        Examples:
            www.linkedin.com -> linkedin.com
            jobs.linkedin.com -> linkedin.com
            linkedin.com -> linkedin.com
            https://www.github.com -> github.com
        """
        if not domain.startswith(("http://", "https://")):
            domain = f"https://{domain}"

        parsed = urlparse(domain)
        hostname = parsed.hostname or parsed.netloc or domain

        parts = hostname.split(".")
        if len(parts) >= 2:
            return ".".join(parts[-2:])

        return hostname

    def get_credentials(self, domain: str) -> dict[str, str]:
        """Get credentials for a specific domain (normalizes domain first)."""
        normalized = self.normalize_domain(domain)
        return self.domains.get(normalized, {})

    def get_login_config(self, domain: str) -> Optional[SiteConfig]:
        """Return login automation config for a domain from sites.yml, or None."""
        normalized = self.normalize_domain(domain)
        return self.site_configs.get(normalized)

    @staticmethod
    def load_credentials() -> "Credentials":
        """Load credentials from secrets.yml only (no site metadata)."""
        return Credentials.load()

    @staticmethod
    def load(
        secrets_path: Optional[Path] = None,
        sites_path: Optional[Path] = None,
    ) -> "Credentials":
        """Load credentials from secrets.yml and optionally merge sites.yml metadata.

        Args:
            secrets_path: Path to secrets.yml. Defaults to project root.
            sites_path: Path to sites.yml. Defaults to project root. Missing file is fine.
        """
        if secrets_path is None:
            secrets_path = _PROJECT_ROOT / "secrets.yml"
        if sites_path is None:
            sites_path = _PROJECT_ROOT / "sites.yml"

        if not secrets_path.exists():
            raise FileNotFoundError(f"secrets.yml not found at {secrets_path.absolute()}")

        with open(secrets_path, "r") as f:
            secrets = yaml.safe_load(f) or {}

        domains: dict[str, dict[str, str]] = {}
        for domain, creds in secrets.items():
            if isinstance(creds, dict):
                domains[domain] = creds

        site_configs: dict[str, SiteConfig] = {}
        if sites_path.exists():
            with open(sites_path, "r") as f:
                sites = yaml.safe_load(f) or {}
            for domain, cfg in sites.items():
                if not isinstance(cfg, dict):
                    continue
                if "login_url" not in cfg or "username_selector" not in cfg or "password_selector" not in cfg:
                    continue
                site_configs[domain] = SiteConfig(
                    login_url=cfg["login_url"],
                    username_selector=cfg["username_selector"],
                    password_selector=cfg["password_selector"],
                    submit_selector=cfg.get("submit_selector"),
                    post_login_check=cfg.get("post_login_check"),
                    notes=cfg.get("notes"),
                )

        return Credentials(domains=domains, site_configs=site_configs)
