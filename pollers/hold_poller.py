"""Deprecation shim — kept for one release.

The canonical scrape runner lives at ``runners/scrape_runner.py``. This
file re-exports its public surface so existing call sites
(``caddy-poller`` console script, anything importing ``pollers.hold_poller``)
keep working while consumers migrate.

Renamed 2026-05-30 as Phase 1 of
``notes.org/Plans/Scrape runner — harden hold-poller``. Remove this
shim after one release once we've confirmed no external consumer is
still pointing at the old path.
"""

import warnings

warnings.warn(
    "pollers.hold_poller is renamed to runners.scrape_runner; update "
    "imports before the next release",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export the public surface unchanged.
from runners.scrape_runner import *  # noqa: F401,F403,E402
from runners.scrape_runner import (  # noqa: E402
    format_scrape_label,
    main_sync,
    poll_once,
    process_scrape,
)

__all__ = [
    "format_scrape_label",
    "main_sync",
    "poll_once",
    "process_scrape",
]
