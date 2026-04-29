"""Graph factory — wires every node into a pydantic-graph Graph.

This module is the single source of truth for the scrape-graph's node
and edge topology. api/ consumes a committed snapshot
(api/job_hunting/api/views/graph_static.json) for its d3 introspection
UI; regenerate it with `uv run caddy-export-graph` whenever nodes
change. A drift test (tests/test_scrape_graph_export.py) fails CI if
the committed snapshot doesn't match the live graph.
"""
from __future__ import annotations

from pydantic_graph import Graph

from .nodes_extract import (
    EvaluateExtraction,
    ExtractFail,
    PersistJobPost,
    ResolveApplyUrl,
    StartExtract,
    Tier0CSS,
    Tier1Mini,
    Tier2Haiku,
    Tier3Sonnet,
    UpdateProfile,
    ValidateExtraction,
)
from .nodes_obstacle import (
    DetectObstacle,
    ObstacleAgent,
    ObstacleFail,
    ObstacleRememberMe,
    ObstacleWaitRetry,
)
from .nodes_scrape import (
    Capture,
    CheckLinkDedup,
    DuplicateShortCircuit,
    ExpandTruncations,
    LoadProfile,
    Navigate,
    PersistScrape,
    ResolveFinalUrl,
    SettleWait,
    StartScrape,
    WaitReadySelector,
)
from .state import ScrapeGraphState

# Names imported above are referenced by forward-ref annotations
# across sibling node modules; pydantic-graph's get_type_hints resolves
# them via this module's namespace (parent of the Graph(...) call).
__all_nodes__ = (
    StartScrape, LoadProfile, Navigate, ResolveFinalUrl, CheckLinkDedup,
    DuplicateShortCircuit, WaitReadySelector, SettleWait, ExpandTruncations,
    Capture, PersistScrape, DetectObstacle, ObstacleRememberMe,
    ObstacleWaitRetry, ObstacleAgent, ObstacleFail, StartExtract, Tier0CSS,
    Tier1Mini, Tier2Haiku, Tier3Sonnet, EvaluateExtraction, ValidateExtraction,
    PersistJobPost, UpdateProfile, ResolveApplyUrl, ExtractFail,
)


# Build graphs at module scope so pydantic-graph's forward-ref resolver
# (which reads the caller's f_locals) sees every node class via this
# module's globals. Building inside a function would hide cross-module
# names behind LOAD_GLOBAL and break the lookup.
_SCRAPE_NODES = [
    StartScrape, LoadProfile, Navigate,
    DetectObstacle, ObstacleRememberMe, ObstacleWaitRetry, ObstacleAgent,
    ObstacleFail,
    ResolveFinalUrl, CheckLinkDedup,
    DuplicateShortCircuit, WaitReadySelector, SettleWait, ExpandTruncations,
    Capture, PersistScrape,
    StartExtract, Tier0CSS, Tier1Mini, Tier2Haiku, Tier3Sonnet,
    EvaluateExtraction, ValidateExtraction, PersistJobPost, UpdateProfile,
    ResolveApplyUrl, ExtractFail,
]
_EXTRACT_NODES = [
    StartExtract, Tier0CSS, Tier1Mini, Tier2Haiku, Tier3Sonnet,
    EvaluateExtraction, ValidateExtraction, PersistJobPost, UpdateProfile,
    ResolveApplyUrl, ExtractFail,
]

_SCRAPE_GRAPH = Graph(nodes=_SCRAPE_NODES, state_type=ScrapeGraphState)
_EXTRACT_GRAPH = Graph(nodes=_EXTRACT_NODES, state_type=ScrapeGraphState)


def build_scrape_graph() -> "Graph[ScrapeGraphState, None, dict]":
    """Full scrape → extract graph, entry = StartScrape.

    Used by the hold-poller and the browser-scrape endpoint.
    """
    return _SCRAPE_GRAPH


def build_extract_graph() -> "Graph[ScrapeGraphState, None, dict]":
    """Extract-only graph, entry = StartExtract.

    Used for paste-from-text / email-pipeline / chat-ingest — cases
    where there's no Playwright page to scrape.
    """
    return _EXTRACT_GRAPH


# ---------------------------------------------------------------------------
# Topology export — consumed by api/'s d3 introspection UI.
# ---------------------------------------------------------------------------

# Display metadata the runtime doesn't carry. Hand-maintained alongside
# the node classes. The exporter cross-checks this against the live
# graph and errors if they drift.
NODE_META: dict[str, dict[str, str]] = {
    # Scrape-side
    "StartScrape": {
        "group": "scrape", "label": "Start",
        "description": (
            "Entry point for the full scrape pipeline. Stamps "
            "original_scrape_id for provenance and hands off to "
            "LoadProfile."
        ),
    },
    "LoadProfile": {
        "group": "scrape", "label": "Load profile",
        "description": (
            "Fetches the per-hostname ScrapeProfile (css selectors, "
            "ready_selector, obstacle hints) from the api. Missing "
            "profile is fine — later nodes degrade to generic behavior."
        ),
    },
    "Navigate": {
        "group": "scrape", "label": "Navigate",
        "description": (
            "Drives the browser to the submitted URL and records the "
            "landed final_url. Hands off to DetectObstacle so login "
            "walls / account choosers get cleared before any URL "
            "canonicalization or content waits run."
        ),
    },
    "ResolveFinalUrl": {
        "group": "scrape", "label": "Resolve final URL",
        "description": (
            "Canonicalizes the landed URL (strips tracker params, "
            "fragments). If the browser redirected to a different host, "
            "creates a child scrape via source_scrape FK so the "
            "tracker→destination hop is queryable."
        ),
    },
    "CheckLinkDedup": {
        "group": "scrape", "label": "Check link dedup",
        "description": (
            "Queries /job-posts/?filter[link]=canonical for a non-stub "
            "hit (>= 60 words of description). On hit, short-circuits "
            "to DuplicateShortCircuit without scraping further."
        ),
    },
    "DuplicateShortCircuit": {
        "group": "terminal", "label": "Duplicate short-circuit",
        "description": (
            "Terminal: we already have a full JobPost for this URL. "
            "Returns outcome='duplicate' with the existing job_post_id "
            "so the caller can navigate the user there."
        ),
    },
    "WaitReadySelector": {
        "group": "scrape", "label": "Wait ready selector",
        "description": (
            "When the profile has a ready_selector (signals SPA content "
            "has rendered), waits up to 5s for it. Hit → ExpandTruncations; "
            "miss → SettleWait."
        ),
    },
    "SettleWait": {
        "group": "scrape", "label": "Settle wait",
        "description": (
            "Fallback fixed 2s sleep when no ready_selector is known, "
            "giving SPAs a chance to finish rendering before capture."
        ),
    },
    "ExpandTruncations": {
        "group": "scrape", "label": "Expand truncations",
        "description": (
            "Clicks 'Show more' / 'Read more' affordances so the captured "
            "content isn't a stub. Best-effort — failures don't block. "
            "Hands directly to Capture (obstacles are now caught upstream "
            "after Navigate)."
        ),
    },
    "Capture": {
        "group": "scrape", "label": "Capture",
        "description": (
            "Reads page.inner_text('body') and page.content() into "
            "state.job_content / state.html. This is the moment the "
            "browser is 'done'."
        ),
    },
    "PersistScrape": {
        "group": "scrape", "label": "Persist scrape",
        "description": (
            "PATCHes the scrape with job_content + html + "
            "status='extracting'. Marks the handoff from browser side "
            "to extract side."
        ),
    },
    # Obstacle-side
    "DetectObstacle": {
        "group": "obstacle", "label": "Detect obstacle",
        "description": (
            "Scans the page body for login-wall signals immediately "
            "after Navigate. Clean → ResolveFinalUrl; walled → routes "
            "to the obstacle handler with available retries "
            "(RememberMe → WaitRetry → Agent → Fail)."
        ),
    },
    "ObstacleRememberMe": {
        "group": "obstacle", "label": "Remember-me reauth",
        "description": (
            "Tries the 'Continue as <you>' re-auth path using a known "
            "profile selector. Recorded per-attempt so we don't loop on it."
        ),
    },
    "ObstacleWaitRetry": {
        "group": "obstacle", "label": "Wait + retry",
        "description": (
            "Soaks up transient auth walls by sleeping 3s and re-checking. "
            "Caps at 3 waits before escalating to ObstacleAgent."
        ),
    },
    "ObstacleAgent": {
        "group": "obstacle", "label": "Obstacle agent",
        "description": (
            "LLM-driven fallback: inspects the page and proposes a CSS "
            "selector to click. On success, the winning selector is fed "
            "back into the ScrapeProfile's probation gate."
        ),
    },
    "ObstacleFail": {
        "group": "terminal", "label": "Obstacle fail",
        "description": (
            "Terminal: every obstacle path has been exhausted. Returns "
            "outcome='failure' with failure_reason='login_wall' so the "
            "caller can re-queue after seeding cookies."
        ),
    },
    # Extract-side
    "StartExtract": {
        "group": "extract", "label": "Start extract",
        "description": (
            "Entry point for the extract-only sub-graph. Used by paste / "
            "email / chat ingest, and by the full pipeline once "
            "PersistScrape lands."
        ),
    },
    "Tier0CSS": {
        "group": "extract", "label": "Tier 0 CSS",
        "description": (
            "Phase 1b placeholder: deferred to server-side parse_scrape "
            "until the api ships /tier0-extract/. Records a soft skip and "
            "lets Tier 1 handle it."
        ),
    },
    "Tier1Mini": {
        "group": "extract", "label": "Tier 1 mini",
        "description": (
            "Cheap, fast LLM pass (gpt-4o-mini by default). POSTs to "
            "/api/v1/scrapes/:id/llm-extract/ with the tier's model spec. "
            "EvaluateExtraction decides whether to escalate."
        ),
    },
    "Tier2Haiku": {
        "group": "extract", "label": "Tier 2 haiku",
        "description": (
            "Mid-tier escalation (claude-haiku-4-5). Same endpoint as "
            "Tier1; the tier label on TierAttempt is what distinguishes "
            "them in the retrospective analysis."
        ),
    },
    "Tier3Sonnet": {
        "group": "extract", "label": "Tier 3 sonnet",
        "description": (
            "Expensive last-resort tier (claude-sonnet-4-6). Gated by "
            "SCRAPE_GRAPH_ENABLE_TIER3=1; otherwise the graph terminates "
            "at ExtractFail after Tier 2."
        ),
    },
    "EvaluateExtraction": {
        "group": "extract", "label": "Evaluate extraction",
        "description": (
            "LLM-output quality gate: checks for title, company, and "
            "thin-description. Pass → ValidateExtraction; fail → next "
            "tier or ExtractFail."
        ),
    },
    "ValidateExtraction": {
        "group": "extract", "label": "Validate extraction",
        "description": (
            "Content-quality gate between EvaluateExtraction-passed and "
            "PersistJobPost. Enforces source-text minimum word count and "
            "loading-shell fingerprints (Salesforce Lightning, Workday, "
            "cookie/JS notices). Catches LLM hallucinations off a never-"
            "hydrated SPA bootstrap. Fail → ExtractFail so the debug-"
            "artifact invariant fires."
        ),
    },
    "PersistJobPost": {
        "group": "extract", "label": "Persist job post",
        "description": (
            "POSTs parsed data to /api/v1/scrapes/:id/persist-extraction/ "
            "which handles dedup, stub-upgrade, and posted_date fallback."
        ),
    },
    "UpdateProfile": {
        "group": "extract", "label": "Update profile",
        "description": (
            "Feeds the outcome (success + tier0_hit flag) back into the "
            "ScrapeProfile so per-host statistics and selector probation "
            "can evolve. Shadow-suppressed."
        ),
    },
    "ResolveApplyUrl": {
        "group": "extract", "label": "Resolve apply URL",
        "description": (
            "Phase 1b: no-op that returns outcome='success'. Phase 2 will "
            "resolve the external apply destination (Greenhouse/Lever/"
            "LinkedIn portal) and stash it on the JobPost."
        ),
    },
    "ExtractFail": {
        "group": "terminal", "label": "Extract fail",
        "description": (
            "Terminal: every tier produced something EvaluateExtraction "
            "rejected, or persistence failed. outcome='failure' with "
            "failure_reason='extraction'."
        ),
    },
}


def export_graph_structure() -> dict:
    """Introspect the live scrape-graph and return its {nodes, edges}
    snapshot in the shape api/ ships on /api/v1/admin/graph-structure/.

    Nodes are ordered to match the registration order in _SCRAPE_NODES
    so the export is stable across runs.
    """
    graph = _SCRAPE_GRAPH
    live_ids = {cls.get_node_id() for cls in _SCRAPE_NODES}
    missing = live_ids - NODE_META.keys()
    extra = NODE_META.keys() - live_ids
    if missing or extra:
        raise RuntimeError(
            f"NODE_META drift: missing={sorted(missing)} extra={sorted(extra)} — "
            "update lib/scrape_graph/graph.py::NODE_META to match the live nodes."
        )

    nodes = [
        {"id": cls.get_node_id(), **NODE_META[cls.get_node_id()]}
        for cls in _SCRAPE_NODES
    ]

    edges: list[tuple[str, str]] = []
    for cls in _SCRAPE_NODES:
        src = cls.get_node_id()
        node_def = graph.node_defs.get(src)
        if node_def is None:
            continue
        for target in sorted(node_def.next_node_edges.keys()):
            edges.append((src, target))

    return {
        "nodes": nodes,
        "edges": [{"from": a, "to": b} for (a, b) in edges],
    }
