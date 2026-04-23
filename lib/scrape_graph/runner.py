"""Entrypoints for invoking the scrape-graph from the poller / mcp
server / paste pipeline.

Callers pass an initial ScrapeGraphState; we pick the right graph
shape (scrape+extract vs extract-only) and run to an End node.
"""
from __future__ import annotations

import logging

from .graph import build_extract_graph, build_scrape_graph
from .state import ScrapeGraphState

logger = logging.getLogger(__name__)


async def run_scrape_graph(
    state: ScrapeGraphState,
    *,
    browser_page=None,
    has_browser: bool = True,
):
    """Kick off the full scrape + extract graph.

    When browser_page is None (or has_browser=False) we skip the scrape
    sub-graph and enter at StartExtract. The scrape nodes access the
    page via state (set by the caller prior to run_scrape_graph).
    """
    # Attach browser page so nodes_scrape can reach it via state attr.
    state._browser_page = browser_page  # type: ignore[attr-defined]
    state._has_browser = has_browser  # type: ignore[attr-defined]

    if not has_browser or browser_page is None:
        graph = build_extract_graph()
        from .nodes_extract import StartExtract
        entry = StartExtract()
    else:
        graph = build_scrape_graph()
        from .nodes_scrape import StartScrape
        entry = StartScrape()

    logger.info(
        "scrape-graph entry=%s scrape_id=%s source=%s",
        type(entry).__name__,
        state.scrape_id,
        state.source,
    )
    result = await graph.run(entry, state=state)
    return result


async def run_extract_graph(state: ScrapeGraphState):
    """Run only the extract sub-graph. For paste/email/chat entries
    where there's nothing to fetch from a browser."""
    state._browser_page = None  # type: ignore[attr-defined]
    state._has_browser = False  # type: ignore[attr-defined]
    graph = build_extract_graph()
    from .nodes_extract import StartExtract
    return await graph.run(StartExtract(), state=state)
