"""Scrape-graph orchestration.

pydantic-graph-based state machine for the scrape + extract pipeline.
ai/ owns the graph runtime; api/ (via HTTP) handles persistence.
Feature-flagged via SCRAPE_GRAPH_MODE = off | shadow | primary.

See ~/.claude/plans/snazzy-crafting-whale.md for the full design.
"""
from .state import (  # noqa: F401
    GraphMode,
    NodeTraceEntry,
    ObstacleAttempt,
    ScrapeGraphState,
    TierAttempt,
    get_mode,
)
from .url_canonicalize import canonicalize_url  # noqa: F401
