"""Scrape-graph orchestration.

pydantic-graph-based state machine for the scrape + extract pipeline.
ai/ owns the graph runtime; api/ (via HTTP) handles persistence.

See ~/.claude/plans/snazzy-crafting-whale.md for the full design.
"""
from .state import (  # noqa: F401
    NodeTraceEntry,
    ObstacleAttempt,
    ScrapeGraphState,
    TierAttempt,
)
from .url_canonicalize import canonicalize_url  # noqa: F401
