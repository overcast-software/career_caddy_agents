"""Shared state carried through the scrape-graph run.

One dataclass per full run; nodes mutate named fields (see comments on
each field for the single-writer rule). Serializable to JSON for the
tracing payload and d3 trace UI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


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
    # When True, PersistScrape routes to ResolveApplyUrl and skips the
    # StartExtract → Tier* → PersistJobPost → ReviewCompleteness →
    # UpdateProfile chain. Used by the staff "Resolve & dedupe" action
    # on jp.edit — runs the browser fetch + apply-url capture but
    # leaves extraction to a future explicit pass. Loaded from the
    # Scrape's `skip_extract` attribute by the hold-poller.
    skip_extract: bool = False
    # Phase B — Extension direct-POST plan. How this scrape was captured.
    # "browser" (default) → full scrape sub-graph (Navigate → Capture → …).
    # "extension-direct" → the extension content-script already extracted
    # title + company + description from the user-rendered DOM and POSTed
    # them via `captured_payload`. StartScrape branches to SkipBrowserTier
    # so no browser-tier node ever runs.
    source_mode: str = "browser"
    # Phase B — Extension direct-POST plan. Set by the runner from the
    # claimed Scrape attribute when source_mode='extension-direct'. Shape
    # is the Phase-A contract enforced by ScrapeSerializer:
    #     {
    #       "title": str,            # required, non-empty
    #       "company": str,          # required, non-empty
    #       "description": str,      # required, non-empty
    #       "apply_url": str | None, # optional
    #       "location": str | None,  # optional
    #       "extraction_hints": dict | None,  # optional
    #     }
    # SkipBrowserTier maps these fields onto state.parsed (ParsedJobData
    # shape) and feeds them to PersistJobPost.
    captured_payload: Optional[dict] = None

    # URL resolution
    final_url: Optional[str] = None
    canonical_url: Optional[str] = None
    rewritten_url: Optional[str] = None  # set by Navigate when profile.url_rewrites fires
    did_redirect: bool = False

    # Scrape-side
    profile: Optional[dict] = None
    html: Optional[str] = None
    job_content: Optional[str] = None
    screenshot_name: Optional[str] = None
    candidate_ready_selector: Optional[str] = None
    candidate_obstacle_click_selector: Optional[str] = None
    discovered_selectors: Optional[dict] = None  # e.g. {"title": "h1", "company": ".company"}
    obstacle_history: list[ObstacleAttempt] = field(default_factory=list)

    # Extract-side
    tier_attempts: list[TierAttempt] = field(default_factory=list)
    parsed: Optional[dict] = None  # ParsedJobData as dict (serializable)
    evaluation: Optional[dict] = None  # {passed: bool, reasons: [str]}

    # Closed-state detection — writer = DetectClosedState (single).
    # Verdict is "closed" or None; None means either "open" (the curated
    # selectors / phrases ran and didn't fire) or "unknown" (no config,
    # capture too thin to invoke LLM). The distinction lives in
    # closed_detection_method, not in the verdict, because downstream
    # JobPostExtractor only flips posting_status on "closed" — None is
    # a no-op for that channel.
    detected_posting_status: Optional[str] = None
    detected_closed_evidence: Optional[str] = None
    closed_detection_method: Optional[str] = None  # css|phrase|llm|no_signal|skipped_thin_capture|no_config

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
