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

    Exception — Phase B extension-direct fast path: when
    state.source_mode='extension-direct', enter at StartScrape even
    without a browser page so the StartScrape gate can branch to
    SkipBrowserTier. The fast path needs no browser; routing it through
    StartExtract would run the tier nodes against an empty job_content
    and fail validation.
    """
    # Attach browser page so nodes_scrape can reach it via state attr.
    state._browser_page = browser_page  # type: ignore[attr-defined]
    state._has_browser = has_browser  # type: ignore[attr-defined]

    fast_path = state.source_mode == "extension-direct"

    if fast_path:
        graph = build_scrape_graph()
        from .nodes_scrape import StartScrape
        entry = StartScrape()
    elif not has_browser or browser_page is None:
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
    # pydantic-graph 2.0: run() is keyword-only. The entry node instance is
    # passed as `inputs`; the graph's start node is wired to NodeStep(entry)
    # in graph.py, so the instance is dispatched to the right first node.
    result = await graph.run(inputs=entry, state=state)
    return result


async def run_extract_graph(state: ScrapeGraphState):
    """Run only the extract sub-graph. For paste/email/chat entries
    where there's nothing to fetch from a browser."""
    state._browser_page = None  # type: ignore[attr-defined]
    state._has_browser = False  # type: ignore[attr-defined]
    graph = build_extract_graph()
    from .nodes_extract import StartExtract
    return await graph.run(inputs=StartExtract(), state=state)
