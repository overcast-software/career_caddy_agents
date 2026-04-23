"""Shared state carried through the scrape-graph run.

One dataclass per full run; nodes mutate named fields (see comments on
each field for the single-writer rule). Serializable to JSON for the
tracing payload and d3 trace UI.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class GraphMode(str, Enum):
    """Feature-flag modes for the scrape-graph runtime.

    - OFF: legacy pipeline runs unchanged; graph code never executes.
    - SHADOW: legacy runs; graph runs afterwards on the same input but
      does NOT persist a JobPost. Used to verify parity before cutover.
    - PRIMARY: graph is authoritative; legacy runs only as fallback
      when the graph terminates in ExtractFail or ObstacleFail.
    """

    OFF = "off"
    SHADOW = "shadow"
    PRIMARY = "primary"


def get_mode() -> GraphMode:
    """Read SCRAPE_GRAPH_MODE from env, default off."""
    raw = (os.environ.get("SCRAPE_GRAPH_MODE") or "off").strip().lower()
    try:
        return GraphMode(raw)
    except ValueError:
        return GraphMode.OFF


@dataclass
class ObstacleAttempt:
    """One attempt at clearing a stuck page. Appended to
    ScrapeGraphState.obstacle_history by obstacle-sub-graph nodes."""

    node: str  # e.g. "ObstacleRememberMe" / "ObstacleAgent"
    selector_tried: Optional[str] = None
    succeeded: bool = False
    note: Optional[str] = None


@dataclass
class TierAttempt:
    """One LLM tier invocation. Appended by Tier1/2/3 nodes.

    Records enough to retro-analyze tier regret: same input → which
    tier should have been tried first.
    """

    tier: str  # "tier0" / "tier1" / "tier2" / "tier3"
    model: Optional[str] = None
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    duration_ms: int = 0
    produced_output: bool = False
    error: Optional[str] = None


@dataclass
class NodeTraceEntry:
    """One node transition, appended by the BaseNode tracing mixin."""

    node: str
    t_start: float  # time.time()
    t_end: float
    inputs_digest: Optional[str] = None
    outputs_digest: Optional[str] = None
    routed_to: Optional[str] = None
    payload: dict = field(default_factory=dict)


@dataclass
class ScrapeGraphState:
    """All state carried through a scrape-graph run.

    Mutability rules — each field names its writer(s):
    - Identity: writer = entrypoint (StartScrape or StartExtract).
    - profile: writer = LoadProfile.
    - html / job_content / screenshot_name / candidate_*_selector /
      obstacle_history: writers = the scrape sub-graph nodes only.
    - tier_attempts / parsed / evaluation: writers = extract sub-graph.
    - outcome / failure_reason / job_post_id / was_duplicate:
      writer = whichever node routes to End(...).
    - node_trace: writer = BaseNode mixin (via tracing.record_transition).
    """

    # Identity — set once
    scrape_id: int = 0  # mutable: ResolveFinalUrl may flip to a new scrape id
    original_scrape_id: int = 0  # set once
    submitted_url: str = ""
    source: str = "manual"  # poller/paste/email/chat/manual/extension
    feature_flag_variant: str = "off"

    # URL resolution
    final_url: Optional[str] = None
    canonical_url: Optional[str] = None
    did_redirect: bool = False

    # Scrape-side
    profile: Optional[dict] = None
    html: Optional[str] = None
    job_content: Optional[str] = None
    screenshot_name: Optional[str] = None
    candidate_ready_selector: Optional[str] = None
    candidate_obstacle_click_selector: Optional[str] = None
    obstacle_history: list[ObstacleAttempt] = field(default_factory=list)

    # Extract-side
    tier_attempts: list[TierAttempt] = field(default_factory=list)
    parsed: Optional[dict] = None  # ParsedJobData as dict (serializable)
    evaluation: Optional[dict] = None  # {passed: bool, reasons: [str]}

    # Outcome
    outcome: Optional[str] = None  # "success" / "duplicate" / "failure"
    failure_reason: Optional[str] = None
    job_post_id: Optional[int] = None
    was_duplicate: bool = False

    # Trace — appended by BaseNode mixin
    node_trace: list[NodeTraceEntry] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        """Serialize for the graph-transition endpoint payload."""
        return {
            "scrape_id": self.scrape_id,
            "original_scrape_id": self.original_scrape_id,
            "canonical_url": self.canonical_url,
            "did_redirect": self.did_redirect,
            "tier_attempts": [ta.__dict__ for ta in self.tier_attempts],
            "obstacle_history": [oa.__dict__ for oa in self.obstacle_history],
            "outcome": self.outcome,
            "failure_reason": self.failure_reason,
            "job_post_id": self.job_post_id,
        }
