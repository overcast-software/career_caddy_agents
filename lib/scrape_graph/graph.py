"""Graph factory — wires every node into a pydantic-graph Graph.

The static node/edge topology lives in api/job_hunting/api/views/graph.py
for the d3 introspection UI. Keep the two in sync when node topology
changes. (TODO Phase 1d: export from here as single source of truth.)
"""
from __future__ import annotations

from pydantic_graph import Graph

from . import nodes_extract, nodes_obstacle, nodes_scrape
from .state import ScrapeGraphState


def build_scrape_graph() -> "Graph[ScrapeGraphState, None, dict]":
    """Full scrape → extract graph, entry = StartScrape.

    Used by the hold-poller and the browser-scrape endpoint.
    """
    return Graph(
        nodes=[
            nodes_scrape.StartScrape,
            nodes_scrape.LoadProfile,
            nodes_scrape.Navigate,
            nodes_scrape.ResolveFinalUrl,
            nodes_scrape.CheckLinkDedup,
            nodes_scrape.DuplicateShortCircuit,
            nodes_scrape.WaitReadySelector,
            nodes_scrape.SettleWait,
            nodes_scrape.ExpandTruncations,
            nodes_obstacle.DetectObstacle,
            nodes_obstacle.ObstacleRememberMe,
            nodes_obstacle.ObstacleWaitRetry,
            nodes_obstacle.ObstacleAgent,
            nodes_obstacle.ObstacleFail,
            nodes_scrape.Capture,
            nodes_scrape.PersistScrape,
            nodes_extract.StartExtract,
            nodes_extract.Tier0CSS,
            nodes_extract.Tier1Mini,
            nodes_extract.Tier2Haiku,
            nodes_extract.Tier3Sonnet,
            nodes_extract.EvaluateExtraction,
            nodes_extract.PersistJobPost,
            nodes_extract.UpdateProfile,
            nodes_extract.ResolveApplyUrl,
            nodes_extract.ExtractFail,
        ],
        state_type=ScrapeGraphState,
    )


def build_extract_graph() -> "Graph[ScrapeGraphState, None, dict]":
    """Extract-only graph, entry = StartExtract.

    Used for paste-from-text / email-pipeline / chat-ingest — cases
    where there's no Playwright page to scrape. Starts directly at
    the extraction tier ladder.
    """
    return Graph(
        nodes=[
            nodes_extract.StartExtract,
            nodes_extract.Tier0CSS,
            nodes_extract.Tier1Mini,
            nodes_extract.Tier2Haiku,
            nodes_extract.Tier3Sonnet,
            nodes_extract.EvaluateExtraction,
            nodes_extract.PersistJobPost,
            nodes_extract.UpdateProfile,
            nodes_extract.ResolveApplyUrl,
            nodes_extract.ExtractFail,
        ],
        state_type=ScrapeGraphState,
    )
