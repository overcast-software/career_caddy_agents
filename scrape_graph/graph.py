"""Graph factory — wires every node into a pydantic-graph Graph.

This module is the single source of truth for the scrape-graph's node
and edge topology. api/ consumes a committed snapshot
(api/job_hunting/api/views/graph_static.json) for its d3 introspection
UI; regenerate it with `uv run caddy-export-graph` whenever nodes
change. A drift test (tests/test_scrape_graph_export.py) fails CI if
the committed snapshot doesn't match the live graph.
"""
from __future__ import annotations

import typing

from pydantic_graph import BaseNode, Graph, GraphBuilder
from pydantic_graph.step import NodeStep

from .nodes_extract import (
    EvaluateExtraction,
    ExtractFail,
    PersistJobPost,
    ResolveApplyUrl,
    ReviewCompleteness,
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
    DetectClosedState,
    DuplicateShortCircuit,
    ExpandTruncations,
    LoadProfile,
    Navigate,
    PersistScrape,
    ResolveFinalUrl,
    ScrollToLoad,
    SettleWait,
    SkipBrowserTier,
    StartScrape,
    WaitReadySelector,
)
from .state import ScrapeGraphState

# Names imported above are referenced by forward-ref annotations
# across sibling node modules; pydantic-graph's get_type_hints resolves
# them via this module's namespace (parent of the Graph(...) call).
__all_nodes__ = (
    StartScrape, LoadProfile, Navigate, ResolveFinalUrl, CheckLinkDedup,
    DuplicateShortCircuit, WaitReadySelector, SettleWait, ScrollToLoad,
    ExpandTruncations, Capture, DetectClosedState, PersistScrape, DetectObstacle,
    ObstacleRememberMe, ObstacleWaitRetry, ObstacleAgent, ObstacleFail,
    StartExtract, Tier0CSS, Tier1Mini, Tier2Haiku, Tier3Sonnet,
    EvaluateExtraction, ValidateExtraction, PersistJobPost,
    ReviewCompleteness, UpdateProfile, ResolveApplyUrl, ExtractFail,
    SkipBrowserTier,
)


# Build graphs at module scope so pydantic-graph's forward-ref resolver
# sees every node class via this module's globals. pydantic-graph 2.0's
# GraphBuilder.node() resolves each node's run() return-type annotation
# with typing.get_type_hints, using the CALLER frame's namespace (its
# f_locals) as the local namespace. At module scope that namespace is this
# module's globals, which imports every node class — so cross-module
# forward-refs (e.g. StartScrape.run -> ExtractFail, which nodes_scrape.py
# does NOT import) resolve here. Wrapping the build in a function would
# expose only that function's locals and break the lookup, so the
# GraphBuilder node/edge loop below is deliberately inline.
_SCRAPE_NODES = [
    StartScrape, LoadProfile, Navigate,
    DetectObstacle, ObstacleRememberMe, ObstacleWaitRetry, ObstacleAgent,
    ObstacleFail,
    ResolveFinalUrl, CheckLinkDedup,
    DuplicateShortCircuit, WaitReadySelector, SettleWait, ScrollToLoad,
    ExpandTruncations,
    Capture, DetectClosedState, PersistScrape,
    StartExtract, Tier0CSS, Tier1Mini, Tier2Haiku, Tier3Sonnet,
    EvaluateExtraction, ValidateExtraction, PersistJobPost,
    ReviewCompleteness, UpdateProfile, ResolveApplyUrl, ExtractFail,
    SkipBrowserTier,
]
_EXTRACT_NODES = [
    StartExtract, Tier0CSS, Tier1Mini, Tier2Haiku, Tier3Sonnet,
    EvaluateExtraction, ValidateExtraction, PersistJobPost,
    ReviewCompleteness, UpdateProfile, ResolveApplyUrl, ExtractFail,
]

# GraphBuilder.node(cls) registers a BaseNode subclass and infers its
# outgoing edges from the run() return-type union; add_edge wires the
# graph's start node to the entry so `graph.run(inputs=Entry(), state=...)`
# dispatches there. validate_graph_structure=False preserves the pre-2.0
# `Graph(nodes=...)` semantics (no reachability/dead-end validation):
# runtime routing between BaseNode steps is by node-class id, independent
# of the inferred edge topology, so structural validation isn't required
# for correctness and an intentionally not-yet-wired node can't break the
# module import.
_scrape_builder = GraphBuilder(state_type=ScrapeGraphState)
for _node_cls in _SCRAPE_NODES:
    _scrape_builder.add(_scrape_builder.node(_node_cls))
_scrape_builder.add_edge(_scrape_builder.start_node, NodeStep(StartScrape))
_SCRAPE_GRAPH = _scrape_builder.build(validate_graph_structure=False)

_extract_builder = GraphBuilder(state_type=ScrapeGraphState)
for _node_cls in _EXTRACT_NODES:
    _extract_builder.add(_extract_builder.node(_node_cls))
_extract_builder.add_edge(_extract_builder.start_node, NodeStep(StartExtract))
_EXTRACT_GRAPH = _extract_builder.build(validate_graph_structure=False)


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
            "profile is fine — later nodes degrade to generic behavior. "
            "Emits ready_selector_count / ready_selector_hash so later "
            "nodes' selector misses are correlatable to which profile "
            "version was loaded."
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
            "Iterates the profile's ready_selector list, trying each "
            "via locator(s).first.wait_for(state='visible') with a "
            "1.5s per-selector budget. Always routes to SettleWait — "
            "a heading match only proves a section header is in DOM, "
            "not that the description body has finished streaming, so "
            "we always pay a post-match sleep. Emits matched_selector "
            "/ matched_index / timed_out / per-attempt durations + "
            "error class so parser errors are distinguishable from "
            "real timeouts."
        ),
    },
    "SettleWait": {
        "group": "scrape", "label": "Settle wait",
        "description": (
            "Fixed 5s sleep on every path. After WaitReadySelector "
            "matches the heading we still need the body to finish "
            "rendering; on miss we need any chance the section will "
            "appear. 5s of guaranteed wait is cheaper than maintaining "
            "a tail-anchor selector and works on any site."
        ),
    },
    "ScrollToLoad": {
        "group": "scrape", "label": "Scroll to load",
        "description": (
            "Scrolls the page in ~250ms increments to trigger "
            "IntersectionObserver-driven lazy hydration (LinkedIn's "
            "'About the job' card is the canonical case). Stops on "
            "ready_selector match, scrollHeight stall, or ~5s budget. "
            "Emits ticks / matched_selector / final_scroll_y telemetry."
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
    "DetectClosedState": {
        "group": "scrape", "label": "Detect closed state",
        "description": (
            "Probes the captured page for closed-posting signal in "
            "priority order: per-host CSS selectors against the live "
            "DOM, then per-host text phrases (curated + learned regex) "
            "against state.job_content, then a Haiku LLM bootstrap when "
            "no host config exists and the capture is substantive "
            "(>= min_chars_for_llm). LLM hits PROMOTE the matched "
            "phrase back into ScrapeProfile.css_selectors."
            "closed_state_config.learned_phrases for next time. "
            "Verdict + evidence land on Scrape.detected_posting_status / "
            "detected_closed_evidence; downstream JobPostExtractor reads "
            "those as the priority-1 channel for posting_status flips. "
            "Always routes to PersistScrape — closed-state is metadata, "
            "never terminal."
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
    "SkipBrowserTier": {
        "group": "scrape", "label": "Skip browser tier",
        "description": (
            "Fast-path entry for source_mode='extension-direct' scrapes. "
            "The extension content-script already extracted title + "
            "company + description from the user-rendered DOM; this "
            "node copies those fields onto state.parsed (ParsedJobData "
            "shape) and routes straight to PersistJobPost. Browser-tier "
            "nodes (Navigate → Capture → Tier*) never run. When the "
            "payload carries apply_url, routes directly to PersistJobPost; "
            "otherwise routes through ResolveApplyUrl (no-op without a "
            "browser page) and into PersistJobPost."
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
            "Entry point for the extract sub-graph. Routes on the "
            "per-host profile's preferred_tier so a known-good domain "
            "enters at the tier it needs: '0'/'auto'/missing → Tier0CSS "
            "($0 deterministic), '1' → Tier1Mini, '2' → Tier2Haiku, "
            "'3' → Tier3Sonnet. Used by paste / email / chat ingest and "
            "by the full pipeline after PersistScrape."
        ),
    },
    "Tier0CSS": {
        "group": "extract", "label": "Tier 0 CSS",
        "description": (
            "Deterministic, $0 CSS extraction tier. When the profile "
            "carries graduated css_selectors.job_data AND captured HTML "
            "is present, parses title/company/description/location with "
            "bs4 (no LLM); a complete parse routes to EvaluateExtraction, "
            "a miss or absent job_data falls through to Tier1Mini so the "
            "api learning loop can auto-demote preferred_tier after "
            "sustained tier0 misses."
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
            "LLM-output quality gate: checks for title, company, "
            "thin-description, and — for LLM tiers — whether the "
            "description is actually grounded in the captured source "
            "(an ungrounded description was invented, and escalates "
            "rather than passing). Pass → ValidateExtraction; fail → "
            "next tier. When escalation is exhausted and only the "
            "description is missing, replaces it with an honest "
            "[DESCRIPTION NOT CAPTURED …] stub, records state."
            "stub_reason for ReviewCompleteness, and continues to "
            "ValidateExtraction; with no title/company it is "
            "ExtractFail."
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
            "which handles dedup, stub-upgrade, and posted_date fallback. "
            "The persist endpoint also runs the CompletenessReviewer as a "
            "side effect, which may flip JobPost.complete=False on rejection."
        ),
    },
    "ReviewCompleteness": {
        "group": "extract", "label": "Review completeness",
        "description": (
            "Marks the persisted JobPost complete=False when the graph "
            "already knows the extraction is a stub — state.stub_reason "
            "set by EvaluateExtraction (partial_render / ungrounded / "
            "no_description) — via PATCH /api/v1/job-posts/:id/. Free "
            "and deterministic; it does not ask a model. Complements "
            "api's LLM CompletenessReviewer, which fires as a side "
            "effect of persist-extraction and judges descriptions the "
            "graph believed were real. The JobPost row always stays; "
            "the flag is what keeps the failure visible and the repair "
            "paths (re-scrape, extension re-send) open. Routes to "
            "UpdateProfile either way."
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


def _node_edge_targets(cls) -> list[str]:
    """Return the sorted node-ids ``cls.run`` can transition to.

    Derived from the run() return-type annotation exactly the way
    pydantic-graph infers edges: every ``BaseNode`` subclass named in the
    return union is an outgoing edge; ``End`` (the graph terminal) is not.
    Forward-refs are resolved against this module's namespace because some
    return hints reference node classes defined in sibling modules
    (e.g. ``StartScrape.run -> ExtractFail``) that the defining module does
    not import. This reproduces the pre-2.0 ``next_node_edges`` edge set
    without depending on pydantic-graph's internal Graph representation.
    """
    hints = typing.get_type_hints(cls.run, localns=dict(globals()))
    ret = hints.get("return")
    targets: set[str] = set()
    for arg in typing.get_args(ret) or (ret,):
        origin = typing.get_origin(arg) or arg
        if isinstance(origin, type) and issubclass(origin, BaseNode) and origin is not BaseNode:
            targets.add(origin.get_node_id())
    return sorted(targets)


def export_graph_structure() -> dict:
    """Introspect the live scrape-graph and return its {nodes, edges}
    snapshot in the shape api/ ships on /api/v1/admin/graph-structure/.

    Nodes are ordered to match the registration order in _SCRAPE_NODES
    so the export is stable across runs.
    """
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
        for target in _node_edge_targets(cls):
            edges.append((src, target))

    return {
        "nodes": nodes,
        "edges": [{"from": a, "to": b} for (a, b) in edges],
    }
