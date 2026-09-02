"""Extract-side nodes. Phase 1b skeleton — tier nodes call api's
llm-extract endpoint (lands in Phase 1c when wired); EvaluateExtraction
escalates through Tier1→Tier2; PersistJobPost POSTs parsed data back
to api for dedup/posted_date/stub-merge handling.
"""
# ruff: noqa: F811
# The forward-declare-then-redefine pattern below is how we give
# pydantic-graph's get_type_hints enough info to resolve Union[...]
# return annotations at class-body time. The second definition is the
# real node; the stubs are intentional.
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Union
from urllib.parse import urlparse

import httpx
from pydantic_graph import BaseNode, End, GraphRunContext

from lib.scrape_inspector import css_extract_job_data, jsonld_extract_job_data
from .state import ScrapeGraphState, TierAttempt
from .tracing import trace_node

logger = logging.getLogger(__name__)


_TIER1_MODEL = os.environ.get("SCRAPE_GRAPH_TIER1_MODEL", "openai:gpt-4o-mini")
_TIER2_MODEL = os.environ.get("SCRAPE_GRAPH_TIER2_MODEL", "anthropic:claude-haiku-4-5")
_TIER3_MODEL = os.environ.get("SCRAPE_GRAPH_TIER3_MODEL", "anthropic:claude-sonnet-4-6")
_STUB_MIN_WORDS = 60

# When a profile's extraction_hints instructs the LLM to emit a
# placeholder description for a documented partial-render state (see
# the LinkedIn SDUI hint), the description will be short — well under
# _STUB_MIN_WORDS — but is intentional, not a thin extraction. Detect
# the literal prefix so EvaluateExtraction lets it through with
# title/company instead of escalating to higher tiers and failing.
_PARTIAL_RENDER_DESCRIPTION_PREFIX = "[DESCRIPTION NOT CAPTURED"


def _is_partial_render_placeholder(description: str) -> bool:
    return description.lstrip().startswith(_PARTIAL_RENDER_DESCRIPTION_PREFIX)


_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Grounding check — the anti-fabrication invariant.
#
# The api-side extractor prompt gives the model no way to say "there was
# no job description on this page", and ParsedJobData requires a
# non-empty title + company_name, so a model handed a content-free
# capture has exactly one way to produce a schema-valid answer: invent
# one. Worse, EvaluateExtraction used to escalate only on *empty* /
# *thin* output — so a confident 60-word fabrication PASSED and stopped
# the ladder, while an honest short answer escalated. Fabrication was
# the cheaper strategy.
#
# This closes that hole deterministically, with no extra LLM call: a
# real extraction copies phrases out of the page, so most of its
# n-grams appear verbatim in the captured source. An invented one has
# almost no overlap. Same technique as the api's `closed_evidence`
# validator (verbatim-substring-or-drop), applied to the description.
#
# Tokenising on `[a-z0-9]+` makes the comparison immune to the
# reformatting a real extraction legitimately does — markdown bullets,
# re-wrapped lines, collapsed whitespace, punctuation changes.
_GROUNDING_NGRAM = 6
_GROUNDING_MIN_RATIO = 0.30
# Below this many tokens there aren't enough n-grams for the ratio to
# mean anything; the thin_description gate already covers that range.
_GROUNDING_MIN_DESC_TOKENS = 40
# Only LLM tiers can fabricate. Tier 0 is deterministic bs4/JSON-LD
# parsing of `state.html`, and a JSON-LD description lives inside a
# <script> block that never appears in the visible-text `job_content` —
# checking it here would reject good $0 extractions.
_FABRICATION_CAPABLE_TIERS = ("tier1", "tier2", "tier3")


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def _description_grounding_ratio(
    description: str, source: str,
) -> float | None:
    """Fraction of the description's n-grams that also occur in `source`.

    Returns None when the check cannot be applied (no source captured,
    or the description is too short for the ratio to carry signal) —
    callers must treat None as "no verdict", never as a failure.
    """
    desc_tokens = _TOKEN_RE.findall((description or "").lower())
    source_tokens = _TOKEN_RE.findall((source or "").lower())
    if len(desc_tokens) < _GROUNDING_MIN_DESC_TOKENS:
        return None
    if len(source_tokens) < _GROUNDING_NGRAM:
        return None
    desc_ngrams = _ngrams(desc_tokens, _GROUNDING_NGRAM)
    if not desc_ngrams:
        return None
    source_ngrams = _ngrams(source_tokens, _GROUNDING_NGRAM)
    return len(desc_ngrams & source_ngrams) / len(desc_ngrams)


# The honest stub. When every tier the ladder is allowed to try comes
# back with no usable description — or with one it invented — we persist
# the fields we DID read off the page (title, company, location, link)
# and say plainly that the body is missing, instead of shipping prose
# nobody wrote. Reuses the established
# `_PARTIAL_RENDER_DESCRIPTION_PREFIX` sentinel so every "we didn't get
# the body" description has one recognisable shape for the frontend, the
# extension, and `_is_partial_render_placeholder` itself.
_STUB_DESCRIPTION_TEMPLATE = (
    "[DESCRIPTION NOT CAPTURED — the scrape reached this posting but "
    "could not read its description ({reason}). Re-scrape the link, or "
    "send the page from the browser extension while it is open.]"
)
_STUB_REASON_TEXT = {
    "ungrounded": "the extracted text did not match anything on the page",
    "no_description": "the page returned no description body",
}


def _stub_description(reason: str) -> str:
    return _STUB_DESCRIPTION_TEMPLATE.format(
        reason=_STUB_REASON_TEXT.get(reason, reason),
    )


def _api_base() -> str:
    return os.environ.get("CC_API_BASE_URL", "").rstrip("/")


def _api_headers() -> dict[str, str]:
    token = os.environ.get("CC_API_TOKEN", "")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _call_llm_extract(
    state: ScrapeGraphState, tier_label: str, model_spec: str,
) -> dict | None:
    """Call api's llm-extract endpoint (Phase 1c), records TierAttempt.

    Returns parsed dict on success, None on failure. The endpoint
    doesn't exist yet in Phase 1a; 404 is recorded as a soft failure
    so Phase 1b skeleton tests can run without a live api sidecar.
    """
    t0 = time.time()
    error: str | None = None
    parsed_dict: dict | None = None
    try:
        resp = httpx.post(
            f"{_api_base()}/api/v1/scrapes/{state.scrape_id}/llm-extract/",
            json={"model": model_spec},
            headers={**_api_headers(), "Content-Type": "application/json"},
            timeout=120.0,
        )
        if resp.status_code == 200:
            parsed_dict = (resp.json() or {}).get("data", {}).get("attributes") or None
        elif resp.status_code == 404:
            error = "llm-extract endpoint not deployed yet"
        else:
            error = f"HTTP {resp.status_code}"
    except Exception as exc:
        error = repr(exc)
        logger.warning("%s LLM call failed: %s", tier_label, exc)
    duration_ms = int((time.time() - t0) * 1000)
    state.tier_attempts.append(
        TierAttempt(
            tier=tier_label,
            model=model_spec,
            duration_ms=duration_ms,
            produced_output=parsed_dict is not None,
            error=error,
        )
    )
    return parsed_dict


def _preferred_tier(state: ScrapeGraphState) -> str:
    """Normalized per-host extract entry tier from the loaded profile.

    Returns one of ``'0'/'1'/'2'/'3'/'auto'``. The api stores
    ``ScrapeProfile.preferred_tier`` as a top-level attribute (an int
    tier or the string ``'auto'``); LoadProfile surfaces it on
    ``state.profile``. We coerce to a comparable lowercased string and
    default to ``'auto'`` for a missing / unrecognized value so it lands
    on the Tier0→Tier1 ladder.
    """
    profile = state.profile if isinstance(state.profile, dict) else {}
    raw = profile.get("preferred_tier")
    if raw is None:
        return "auto"
    val = str(raw).strip().lower()
    return val if val in ("0", "1", "2", "3", "auto") else "auto"


def _profile_job_data(state: ScrapeGraphState) -> dict:
    """The per-host ``css_selectors.job_data`` selector map, or ``{}``.

    LoadProfile flattens the ``css_selectors`` JSONB blob onto
    ``state.profile``, so the discovered job_data selectors live at
    ``state.profile['job_data']`` (the same access pattern
    DetectClosedState / _discover_selectors use). Returns ``{}`` when the
    map is absent or malformed.
    """
    profile = state.profile if isinstance(state.profile, dict) else {}
    job_data = profile.get("job_data")
    return job_data if isinstance(job_data, dict) and job_data else {}


def _tier0_fields_complete(parsed: dict | None) -> bool:
    """A Tier-0 parse is 'complete enough' to skip the LLM tiers when the
    three load-bearing fields are non-empty — the same title+company+
    description gate EvaluateExtraction applies. Shared by both Tier-0
    paths (JSON-LD and per-host CSS)."""
    if not isinstance(parsed, dict):
        return False
    title = (parsed.get("title") or "").strip()
    company = (parsed.get("company_name") or "").strip()
    description = (parsed.get("description") or "").strip()
    return bool(title and company and description)


def _record_tier0_hit(
    state: ScrapeGraphState, parsed: dict, method: str, started: float,
) -> "EvaluateExtraction":
    """Record a $0/no-LLM Tier-0 success and route to EvaluateExtraction.

    ``method`` is ``'jsonld'`` or ``'css'`` — captured on the trace so the
    eval loop can tell which deterministic path paid off.
    """
    state.parsed = parsed
    state.tier_attempts.append(
        TierAttempt(
            tier="tier0", model=None, cost_usd=0.0, produced_output=True,
        )
    )
    trace_node(
        state, "Tier0CSS", "EvaluateExtraction", started,
        {"method": method, "fields": sorted(k for k, v in parsed.items() if v)},
    )
    return EvaluateExtraction()


# Forward refs
class Tier0CSS(BaseNode[ScrapeGraphState, None, dict]):
    pass


class Tier1Mini(BaseNode[ScrapeGraphState, None, dict]):
    pass


class Tier2Haiku(BaseNode[ScrapeGraphState, None, dict]):
    pass


class Tier3Sonnet(BaseNode[ScrapeGraphState, None, dict]):
    pass


class EvaluateExtraction(BaseNode[ScrapeGraphState, None, dict]):
    pass


class ValidateExtraction(BaseNode[ScrapeGraphState, None, dict]):
    pass


class PersistJobPost(BaseNode[ScrapeGraphState, None, dict]):
    pass


class ReviewCompleteness(BaseNode[ScrapeGraphState, None, dict]):
    pass


class UpdateProfile(BaseNode[ScrapeGraphState, None, dict]):
    pass


class ResolveApplyUrl(BaseNode[ScrapeGraphState, None, dict]):
    pass


class ExtractFail(BaseNode[ScrapeGraphState, None, dict]):
    pass


@dataclass
class StartExtract(BaseNode[ScrapeGraphState, None, dict]):
    """Entry point for the extract sub-graph. Routes on the per-host
    profile's ``preferred_tier`` so a domain the learning loop already
    knows enters at the tier it needs instead of always paying the
    Tier0→Tier1 ladder:

        '0' / 'auto' / missing → Tier0CSS  (deterministic CSS, $0)
        '1'                    → Tier1Mini  (gpt-4o-mini)
        '2'                    → Tier2Haiku (claude-haiku)
        '3'                    → Tier3Sonnet (claude-sonnet)

    A higher pin only skips the *cheaper* tiers on entry; escalation
    above the entry tier still flows through EvaluateExtraction's gates.
    """

    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> Union[Tier0CSS, Tier1Mini, Tier2Haiku, Tier3Sonnet]:
        started = time.time()
        state = ctx.state
        tier = _preferred_tier(state)
        routes = {
            "1": ("Tier1Mini", Tier1Mini),
            "2": ("Tier2Haiku", Tier2Haiku),
            "3": ("Tier3Sonnet", Tier3Sonnet),
        }
        target_name, target_cls = routes.get(tier, ("Tier0CSS", Tier0CSS))
        trace_node(
            state, "StartExtract", target_name, started,
            {"preferred_tier": tier},
        )
        return target_cls()


@dataclass
class Tier0CSS(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    """Deterministic, $0 extraction tier — two parsers, no LLM, no cost.

    Tried in cost/robustness order:

      1. **JSON-LD** (``schema.org/JobPosting`` in
         ``<script type="application/ld+json">``). Needs ONLY the captured
         HTML — no per-host profile — so it fires on first contact with
         any domain that emits Google-for-Jobs structured data
         (NEOGOV/governmentjobs, Greenhouse, Lever, Workday, iCIMS,
         LinkedIn). Robust to the per-host CSS-selector rot that motivated
         CC-27; also recovers fields CSS can't reliably hook (salary,
         posted_date).
      2. **Per-host CSS** (graduated ``css_selectors.job_data``). When the
         profile carries job_data selectors AND the HTML is present, parse
         the four core fields with bs4 (``lib.scrape_inspector``).

    A complete parse from either path (title + company + description)
    records a tier0 hit and hands straight to EvaluateExtraction. A CSS
    miss (selectors present but the parse came up short) records
    ``produced_output=False`` and falls through to Tier1Mini so the api
    update-from-outcome loop can auto-demote ``preferred_tier`` after
    sustained tier0 misses rather than hard-failing. An absent JSON-LD
    JobPosting AND absent ``job_data`` map preserves the Phase-1b skeleton
    soft skip → Tier1Mini.
    """

    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> Union[Tier1Mini, EvaluateExtraction]:
        started = time.time()
        state = ctx.state
        html = state.html or ""

        # (1) JSON-LD structured data — profile-free, tried first.
        if html:
            jsonld_parsed = jsonld_extract_job_data(html)
            if _tier0_fields_complete(jsonld_parsed):
                return _record_tier0_hit(state, jsonld_parsed, "jsonld", started)

        # (2) Per-host CSS selectors (graduated css_selectors.job_data).
        job_data = _profile_job_data(state)
        if job_data and html:
            parsed = css_extract_job_data(html, job_data)
            if _tier0_fields_complete(parsed):
                return _record_tier0_hit(state, parsed, "css", started)
            # Selectors present but the parse came up short — record the
            # miss so UpdateProfile reports tier0_hit=False and the api
            # learning loop can demote preferred_tier after enough misses.
            state.tier_attempts.append(
                TierAttempt(
                    tier="tier0", model=None,
                    cost_usd=0.0, produced_output=False,
                )
            )
            trace_node(
                state, "Tier0CSS", "Tier1Mini", started, {"tier0_miss": True},
            )
            return Tier1Mini()

        # No JSON-LD JobPosting, no job_data selectors (or no captured
        # HTML) — preserve the Phase-1b skeleton behavior: soft skip, let
        # Tier1 handle it.
        state.tier_attempts.append(
            TierAttempt(tier="tier0", produced_output=False, model=None)
        )
        trace_node(state, "Tier0CSS", "Tier1Mini", started)
        return Tier1Mini()


@dataclass
class Tier1Mini(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> EvaluateExtraction:
        started = time.time()
        parsed = _call_llm_extract(ctx.state, "tier1", _TIER1_MODEL)
        if parsed:
            ctx.state.parsed = parsed
        trace_node(ctx.state, "Tier1Mini", "EvaluateExtraction", started)
        return EvaluateExtraction()


@dataclass
class Tier2Haiku(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> EvaluateExtraction:
        started = time.time()
        parsed = _call_llm_extract(ctx.state, "tier2", _TIER2_MODEL)
        if parsed:
            ctx.state.parsed = parsed
        trace_node(ctx.state, "Tier2Haiku", "EvaluateExtraction", started)
        return EvaluateExtraction()


@dataclass
class Tier3Sonnet(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    """Wired but disabled — EvaluateExtraction gates on an env flag."""

    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> EvaluateExtraction:
        started = time.time()
        parsed = _call_llm_extract(ctx.state, "tier3", _TIER3_MODEL)
        if parsed:
            ctx.state.parsed = parsed
        trace_node(ctx.state, "Tier3Sonnet", "EvaluateExtraction", started)
        return EvaluateExtraction()


@dataclass
class EvaluateExtraction(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    """Quality gate + escalation router for the tier ladder.

    Three outcomes:

    - **pass** → ValidateExtraction. Title, company, and a description
      that is both substantial and *grounded in the captured source*.
    - **escalate** → the next tier. Any reason at all, including a
      description the model appears to have invented.
    - **stub** → ValidateExtraction carrying an honest placeholder.
      Escalation is exhausted, the description is unusable, but title
      and company are real. We persist what we actually read and say
      plainly that the body is missing; ReviewCompleteness then marks
      the JobPost `complete=False` so the failure stays visible and the
      repair paths stay open.

    The stub branch exists because the alternative outcomes are both
    worse. ExtractFail throws away a title and company we genuinely
    read off the page. Passing the model's invented prose through is
    worse still: fabricated text is indistinguishable from real text to
    every downstream consumer, and a post that reads `complete=True`
    actively suppresses the affordances (extension re-send, re-scrape
    prompts) that would have repaired it.
    """

    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> Union[ValidateExtraction, Tier2Haiku, Tier3Sonnet, ExtractFail]:
        started = time.time()
        state = ctx.state
        parsed = state.parsed or {}
        # EvaluateExtraction runs once per tier. Clear any verdict left
        # by the previous rung so a tier that escalated as a stub and
        # then succeeded on retry doesn't drag a stale stub_reason into
        # ReviewCompleteness and mark a good post incomplete.
        state.stub_reason = None
        reasons: list[str] = []
        title = (parsed.get("title") or "").strip()
        company = (parsed.get("company_name") or "").strip()
        description = (parsed.get("description") or "").strip()
        if not title:
            reasons.append("missing_title")
        if not company:
            reasons.append("missing_company")
        partial_render = _is_partial_render_placeholder(description)
        if (
            description
            and not partial_render
            and len(description.split()) < _STUB_MIN_WORDS
        ):
            reasons.append("thin_description")

        last_tier = state.tier_attempts[-1].tier if state.tier_attempts else ""

        # Anti-fabrication: a description an LLM tier returned that does
        # not appear in the captured source was invented. Treat it as a
        # first-class escalation reason so a cheap tier that hallucinates
        # gets ESCALATED past rather than rewarded with an early exit.
        grounding: float | None = None
        if (
            description
            and not partial_render
            and last_tier in _FABRICATION_CAPABLE_TIERS
        ):
            grounding = _description_grounding_ratio(
                description, state.job_content or "",
            )
            if grounding is not None and grounding < _GROUNDING_MIN_RATIO:
                reasons.append("ungrounded_description")

        passed = not reasons
        state.evaluation = {
            "passed": passed,
            "reasons": reasons,
            "partial_render": partial_render,
            "grounding_ratio": grounding,
        }

        # The documented partial-render placeholder is an honest stub the
        # profile's extraction_hints asked for. It passes the gate — but
        # it is NOT real content, and saying so is the whole point.
        if partial_render:
            state.stub_reason = "partial_render"

        tier3_enabled = os.environ.get("SCRAPE_GRAPH_ENABLE_TIER3") == "1"

        if passed:
            trace_node(
                state, "EvaluateExtraction", "ValidateExtraction", started,
                {"stub_reason": state.stub_reason, "grounding_ratio": grounding},
            )
            return ValidateExtraction()
        if last_tier in ("tier0", "tier1"):
            trace_node(
                state, "EvaluateExtraction", "Tier2Haiku", started,
                {"reasons": reasons, "grounding_ratio": grounding},
            )
            return Tier2Haiku()
        if last_tier == "tier2" and tier3_enabled:
            trace_node(
                state, "EvaluateExtraction", "Tier3Sonnet", started,
                {"reasons": reasons, "grounding_ratio": grounding},
            )
            return Tier3Sonnet()

        # Escalation exhausted. Degrade to an honest stub when the only
        # thing we failed to get is the description — title and company
        # came off the page and are worth keeping. Anything else (no
        # title, no company) means we never identified the posting at
        # all, so there is nothing truthful to persist: fail hard and let
        # ExtractFail capture the debug artifact.
        description_only = bool(
            title
            and company
            and not ({"missing_title", "missing_company"} & set(reasons))
        )
        if description_only:
            stub_reason = (
                "ungrounded" if "ungrounded_description" in reasons
                else "no_description"
            )
            state.stub_reason = stub_reason
            parsed["description"] = _stub_description(stub_reason)
            state.parsed = parsed
            state.evaluation = {
                **state.evaluation,
                "stubbed": True,
                "stub_reason": stub_reason,
            }
            logger.info(
                "EvaluateExtraction: degrading to honest stub scrape_id=%s "
                "reason=%s reasons=%s grounding=%s",
                state.scrape_id, stub_reason, reasons, grounding,
            )
            trace_node(
                state, "EvaluateExtraction", "ValidateExtraction", started,
                {"stubbed": True, "stub_reason": stub_reason,
                 "reasons": reasons, "grounding_ratio": grounding},
            )
            return ValidateExtraction()

        trace_node(
            state, "EvaluateExtraction", "ExtractFail", started,
            {"reasons": reasons, "grounding_ratio": grounding},
        )
        return ExtractFail()


# Loading-shell fingerprints — substrings that, when several appear in
# the visible page text, signal a never-hydrated SPA shell rather than
# real content. Anything below the source-word-count floor is rejected
# outright; these catch pages that are long but empty (cookie notices,
# CSS errors, Salesforce Lightning / Workday / Oracle Cloud bootstraps).
_LOADING_SHELL_PHRASES = (
    "sorry to interrupt",
    "css error",
    "cookieenabled",
    "enable cookies",
    "please enable javascript",
    "this page requires javascript",
    "wd-body-loading",  # Workday
)
_LOADING_SHELL_MIN_HITS = 2

_SOURCE_MIN_WORDS = 40

# UI-chrome-only description fingerprint. LinkedIn's lazy-hydrated
# "About the job" card occasionally produces an extraction whose
# `description` is just the visible chrome around it (filter pills,
# salary banner, Apply/Save buttons, the "Use AI to assess how you
# fit" CTA). It looks plausible to EvaluateExtraction because the
# title and company come from the page header — but the body has no
# real prose. Reject when the description is short AND has no real-
# job vocabulary AND is dominated by chrome words.
_UI_CHROME_DESC_MAX_CHARS = 200
_UI_CHROME_REAL_JOB_PHRASES = (
    "responsibilities", "qualifications", "requirements", "experience",
    "years", "you will", "we are looking", "about the role",
    "about the job", "our team",
)
_UI_CHROME_PILL_PHRASES = (
    "remote", "hybrid", "on-site", "onsite", "full-time", "fulltime",
    "part-time", "parttime", "contract", "apply", "save",
    "use ai to assess", "how you fit",
)
# Salary/timeframe units that show up in the chrome banner without
# carrying real prose (e.g. "$215K/yr - $250K/yr").
_UI_CHROME_UNIT_TOKENS = frozenset({"yr", "hr", "wk", "mo", "k"})
_UI_CHROME_PILL_RATIO = 0.6


def _ui_chrome_vocab() -> frozenset[str]:
    """Words that appear inside any chrome pill phrase. Built once."""
    vocab: set[str] = set()
    for phrase in _UI_CHROME_PILL_PHRASES:
        for word in phrase.replace("-", " ").split():
            vocab.add(word)
    return frozenset(vocab)


_UI_CHROME_VOCAB = _ui_chrome_vocab()


def _is_ui_chrome_description(description: str) -> bool:
    """Heuristic: does this description look like it's just LinkedIn
    UI chrome (pills + salary + Apply/Save) instead of real prose?

    Tokenises on non-alphanumerics so '$215K/yr' yields ['215k', 'yr']
    — both count as chrome. A token counts as chrome if it's in the
    pill vocabulary, contains a digit (salary fragment), or is a
    short unit word like 'yr'.
    """
    text = (description or "").strip()
    if not text or len(text) > _UI_CHROME_DESC_MAX_CHARS:
        return False
    lowered = text.lower()
    if any(p in lowered for p in _UI_CHROME_REAL_JOB_PHRASES):
        return False
    tokens = _TOKEN_RE.findall(lowered)
    if not tokens:
        return False
    chrome = 0
    for tok in tokens:
        if tok in _UI_CHROME_VOCAB:
            chrome += 1
        elif any(c.isdigit() for c in tok):
            chrome += 1
        elif tok in _UI_CHROME_UNIT_TOKENS:
            chrome += 1
    return (chrome / len(tokens)) >= _UI_CHROME_PILL_RATIO


@dataclass
class ValidateExtraction(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    """Content-quality gate between EvaluateExtraction-passed and PersistJobPost.

    EvaluateExtraction only asks "does the LLM output *look* like job
    data?" — that check was fooled by a Salesforce loading shell
    (scrape 172) where the LLM hallucinated a plausible job from
    `Loading…Sorry to interrupt` text. This node adds invariants the
    LLM can't judge: is the *source* material big enough to contain a
    real posting? Does it match known SPA-shell fingerprints?

    Failing here routes to ExtractFail so PR 35's debug-artifact
    invariant fires — we get a screenshot + DOM snapshot on the
    scrape row instead of silently poisoning the ScrapeProfile
    learning loop with a fake success.
    """

    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> Union[PersistJobPost, ExtractFail]:
        started = time.time()
        state = ctx.state
        source = (state.job_content or "").strip()
        reasons: list[str] = []

        if len(source.split()) < _SOURCE_MIN_WORDS:
            reasons.append("source_too_short")

        lowered = source.lower()
        hits = sum(1 for p in _LOADING_SHELL_PHRASES if p in lowered)
        if hits >= _LOADING_SHELL_MIN_HITS:
            reasons.append("loading_shell_fingerprint")

        # The ui-chrome heuristic asks "is this description just the
        # page furniture?". An honest stub is neither — it is our own
        # sentinel, deliberately emitted by EvaluateExtraction, and
        # rejecting it here would throw away the title + company we did
        # read and lose the visible-failure record Doug asked for.
        parsed_description = ((state.parsed or {}).get("description") or "")
        if (
            not _is_partial_render_placeholder(parsed_description)
            and _is_ui_chrome_description(parsed_description)
        ):
            reasons.append("ui_chrome_only")

        state.evaluation = {
            **(state.evaluation or {}),
            "validate_passed": not reasons,
            "validate_reasons": reasons,
        }

        if reasons:
            state.failure_reason = f"validate_failed: {','.join(reasons)}"
            trace_node(
                state, "ValidateExtraction", "ExtractFail", started,
                {"reasons": reasons, "source_words": len(source.split())},
            )
            return ExtractFail()
        trace_node(state, "ValidateExtraction", "PersistJobPost", started)
        return PersistJobPost()


@dataclass
class PersistJobPost(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> Union[ReviewCompleteness, ExtractFail, End[dict]]:
        from .nodes_scrape import _patch_scrape_status
        started = time.time()
        state = ctx.state
        try:
            resp = httpx.post(
                f"{_api_base()}/api/v1/scrapes/{state.scrape_id}/persist-extraction/",
                json={"attributes": state.parsed or {}},
                headers={**_api_headers(), "Content-Type": "application/json"},
                timeout=60.0,
            )
            body = resp.json() if resp.status_code < 500 else {}
            meta = (body or {}).get("meta") or {}
            state.job_post_id = meta.get("job_post_id")
            state.was_duplicate = (meta.get("outcome") == "duplicate")
        except Exception:
            logger.warning("PersistJobPost: post failed", exc_info=True)
            trace_node(state, "PersistJobPost", "ExtractFail", started)
            return ExtractFail()
        # Fast-path terminal: extension-direct scrapes skip the
        # ReviewCompleteness → UpdateProfile → ResolveApplyUrl tail,
        # because UpdateProfile probes browser-tier selector candidates
        # (none on this path) and ResolveApplyUrl needs a browser page
        # (we already ran it as a no-op upstream when apply_url was
        # null). Close the scrape row and End.
        if state.source_mode == "extension-direct":
            state.outcome = (
                "duplicate" if state.was_duplicate else "success"
            )
            note = (
                f"extension-direct: job_post {state.job_post_id}"
                if state.job_post_id
                else "extension-direct: persisted"
            )
            _patch_scrape_status(state.scrape_id, "completed", note=note)
            trace_node(
                state, "PersistJobPost", "End", started,
                payload={
                    "outcome": state.outcome,
                    "fast_path": True,
                    "job_post_id": state.job_post_id,
                },
            )
            return End({
                "outcome": state.outcome,
                "job_post_id": state.job_post_id,
                "scrape_id": state.scrape_id,
                "fast_path": True,
            })
        trace_node(state, "PersistJobPost", "ReviewCompleteness", started)
        return ReviewCompleteness()


@dataclass
class ReviewCompleteness(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    """Final gate on the persisted JobPost — is this a real posting or a
    stub we should flag?

    Two independent reviewers converge here, and they answer different
    questions:

    - **api's LLM CompletenessReviewer** fires as a side effect inside
      `/persist-extraction/` (called by the upstream PersistJobPost
      node). It asks a cheap model "does this read like a job posting?"
      and flips `JobPost.complete=False` when it says no. It is a
      judgement call, it costs a call, and it is instructed to default
      to ACCEPT when uncertain.
    - **This node** answers the question the graph already knows the
      answer to, for free: *we* emitted this description as a stub. That
      is not a judgement, it is a fact recorded in `state.stub_reason` by
      EvaluateExtraction. It must not depend on a model agreeing.

    Before this node did the write, that fact died in the graph:
    EvaluateExtraction set `partial_render=True`, PersistJobPost POSTed
    only `state.parsed`, and the JobPost landed on
    `JobPost.complete`'s `default=True`. The one component that knew the
    post was a stub told nobody — so the record presented as a normal
    complete post, and `complete=True` suppressed the extension's
    re-send affordance, disabling the repair path. jp `rHeRo6qWCG`
    (Siemens / LinkedIn, scrape `X04b4IjnTi`) is the worked example.

    The JobPost row always stays — we never delete or hide it. The flag
    is the load-bearing signal.
    """

    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> Union[UpdateProfile, ExtractFail]:
        started = time.time()
        state = ctx.state
        marked = False
        if state.stub_reason and state.job_post_id:
            marked = _mark_job_post_incomplete(
                state.job_post_id, reason=state.stub_reason,
            )
        # CC-248 persistence hop. Capture may have adopted the page's own
        # declared canonical into state.canonical_url; without this write
        # it would die with the graph run and the whole ladder would
        # deliver nothing observable.
        canonical_written = False
        if state.canonical_source != "resolved" and state.job_post_id:
            canonical_written = _persist_declared_canonical(
                state.job_post_id,
                state.canonical_url or "",
                source=state.canonical_source,
            )
        payload = {}
        if state.stub_reason:
            payload = {
                "stub_reason": state.stub_reason,
                "marked_incomplete": marked,
            }
        if state.canonical_source != "resolved":
            payload["canonical_source"] = state.canonical_source
            payload["canonical_written"] = canonical_written
        trace_node(
            state, "ReviewCompleteness", "UpdateProfile", started,
            payload or None,
        )
        return UpdateProfile()


def _persist_declared_canonical(
    job_post_id: str, canonical_url: str, *, source: str
) -> bool:
    """PATCH ``JobPost.canonical_link`` with a page-DECLARED canonical.
    Returns True when the api took it.

    Gated on the value being a declaration (``state.canonical_source`` is
    one of link_rel / og_url / profile_selector), never on a merely
    RESOLVED one. That is not squeamishness: the resolved value already
    has its own propagation path — ``_propagate_canonical_to_parent_jp``
    in the ResolveFinalUrl redirect branch — which targets the PARENT
    scrape's pre-existing JobPost. Writing resolved values from here too
    would duplicate that path onto a different row for no gain and put a
    second writer on the same column.

    ``JobPost.save()`` only re-derives ``canonical_link`` when it is
    empty, so a direct PATCH sticks. ``link`` is deliberately untouched —
    Doug's 2026-08-25 ruling is that the stored original link is
    preserved, and the api's PATCH path leaves it alone.

    Best-effort. This is dedupe-recall enrichment, not load-bearing for
    the scrape: a failure is logged and the graph continues to
    UpdateProfile with the JobPost intact.
    """
    if not canonical_url:
        return False
    try:
        resp = httpx.patch(
            f"{_api_base()}/api/v1/job-posts/{job_post_id}/",
            json={
                "data": {
                    "type": "job-post",
                    "id": str(job_post_id),
                    "attributes": {"canonical_link": canonical_url},
                }
            },
            headers={**_api_headers(), "Content-Type": "application/vnd.api+json"},
            timeout=10.0,
        )
    except Exception:
        logger.warning(
            "ReviewCompleteness: declared canonical_link PATCH errored "
            "job_post_id=%s source=%s url=%s",
            job_post_id, source, canonical_url, exc_info=True,
        )
        return False
    if resp.status_code >= 400:
        logger.warning(
            "ReviewCompleteness: declared canonical_link PATCH rejected "
            "job_post_id=%s source=%s status=%s body=%s",
            job_post_id, source, resp.status_code, resp.text[:300],
        )
        return False
    logger.info(
        "ReviewCompleteness: stored declared canonical_link on JobPost %s "
        "(source=%s url=%s)",
        job_post_id, source, canonical_url,
    )
    return True


def _mark_job_post_incomplete(job_post_id: str, *, reason: str) -> bool:
    """PATCH ``JobPost.complete=False``. Returns True when the api took it.

    `complete` is writable on `PATCH /api/v1/job-posts/<id>/` for the
    owner or staff, and the runner's `CC_API_TOKEN` is a staff key. The
    api never flips False back to True on its own — only a later
    successful extraction does, which is exactly the repair we want to
    stay possible.

    Ordering matters and is safe: PersistJobPost's `/persist-extraction/`
    call runs the api-side create/upgrade branches that set
    `complete=True`, and it has already returned by the time this runs.

    Failure is logged loudly rather than raised. A stub that stays
    marked complete is the bug this node exists to prevent, so losing
    the write silently would defeat the purpose — but it must not cost
    us the JobPost we just persisted.
    """
    try:
        resp = httpx.patch(
            f"{_api_base()}/api/v1/job-posts/{job_post_id}/",
            json={
                "data": {
                    "type": "job-post",
                    "id": str(job_post_id),
                    "attributes": {"complete": False},
                }
            },
            headers={**_api_headers(), "Content-Type": "application/vnd.api+json"},
            timeout=10.0,
        )
    except Exception:
        logger.warning(
            "ReviewCompleteness: complete=False PATCH errored "
            "job_post_id=%s reason=%s — post will read as complete",
            job_post_id, reason, exc_info=True,
        )
        return False
    if resp.status_code >= 400:
        logger.warning(
            "ReviewCompleteness: complete=False PATCH rejected "
            "job_post_id=%s reason=%s status=%s body=%s",
            job_post_id, reason, resp.status_code, resp.text[:300],
        )
        return False
    logger.info(
        "ReviewCompleteness: marked JobPost %s incomplete (stub_reason=%s)",
        job_post_id, reason,
    )
    return True


@dataclass
class UpdateProfile(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> ResolveApplyUrl:
        started = time.time()
        state = ctx.state
        host = (urlparse(state.canonical_url or state.submitted_url or "").hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        if host:
            try:
                tier0_hit = any(
                    t.tier == "tier0" and t.produced_output
                    for t in state.tier_attempts
                )
                httpx.post(
                    f"{_api_base()}/api/v1/scrape-profiles/{host}/update-from-outcome/",
                    json={
                        "scrape_id": state.scrape_id,
                        "success": bool(state.job_post_id),
                        "tier0_hit": tier0_hit,
                    },
                    headers={**_api_headers(), "Content-Type": "application/json"},
                    timeout=10.0,
                )
            except Exception:
                # Learning loop — losing this silently means the profile's
                # success_rate stat stops updating. Warn so it's visible.
                logger.warning(
                    "UpdateProfile: update-from-outcome POST failed host=%s scrape_id=%s",
                    host, state.scrape_id, exc_info=True,
                )
            _write_selector_candidates(host, state)
        trace_node(state, "UpdateProfile", "ResolveApplyUrl", started)
        return ResolveApplyUrl()


_READY_PROBATION = 2
_OBSTACLE_PROBATION = 2


def _write_selector_candidates(host: str, state: ScrapeGraphState) -> None:
    """Apply the probation-gated selector writes. Reads the current
    profile, diffs against the scrape's candidates, and PATCHes via the
    api's scrape-profiles endpoint.
    """
    discovered = state.discovered_selectors
    cand_ready = state.candidate_ready_selector
    cand_obstacle = state.candidate_obstacle_click_selector
    if not (discovered or cand_ready or cand_obstacle):
        return
    try:
        resp = httpx.get(
            f"{_api_base()}/api/v1/scrape-profiles/",
            params={"filter[hostname]": host},
            headers=_api_headers(),
            timeout=10.0,
        )
        payload = resp.json() if resp.status_code == 200 else {}
        rows = payload.get("data") or []
        if not rows:
            return
        row = rows[0]
        profile_id = row.get("id")
        attrs = row.get("attributes") or {}
        existing = attrs.get("css-selectors") or attrs.get("css_selectors") or {}
    except Exception:
        # If we can't read the current profile we can't safely diff-and-patch.
        # Warn: this is the path by which new selector signals get lost.
        logger.warning(
            "UpdateProfile: profile fetch failed host=%s",
            host, exc_info=True,
        )
        return

    updated = dict(existing)
    changed = False

    if discovered and not existing.get("job_data"):
        updated["job_data"] = discovered
        changed = True

    if _apply_probation(updated, cand_ready, "_ready_selector_candidate",
                       "ready_selector", _READY_PROBATION, existing):
        changed = True
    if _apply_probation(updated, cand_obstacle, "_obstacle_click_candidate",
                       "obstacle_click_selector", _OBSTACLE_PROBATION, existing):
        changed = True

    if not (changed and profile_id):
        return
    try:
        httpx.patch(
            f"{_api_base()}/api/v1/scrape-profiles/{profile_id}/",
            json={
                "data": {
                    "type": "scrape-profile",
                    "id": str(profile_id),
                    "attributes": {"css-selectors": updated},
                }
            },
            headers={**_api_headers(), "Content-Type": "application/vnd.api+json"},
            timeout=10.0,
        )
    except Exception:
        # Final write-back of graduated selectors. If this fails silently
        # the probation gate resets next run and selectors never stick.
        logger.warning(
            "UpdateProfile: PATCH profile failed profile_id=%s",
            profile_id, exc_info=True,
        )


def _apply_probation(
    updated: dict, candidate: str | None, candidate_key: str,
    graduated_key: str, threshold: int, existing: dict,
) -> bool:
    """Ported from hold_poller. Candidate must match `threshold` consecutive
    scrapes before it's promoted from `candidate_key` to `graduated_key`."""
    if not candidate or existing.get(graduated_key):
        return False
    prev = existing.get(candidate_key) or {}
    prev_sel = prev.get("selector") if isinstance(prev, dict) else None
    prev_count = prev.get("matches", 0) if isinstance(prev, dict) else 0
    if prev_sel == candidate:
        matches = prev_count + 1
        if matches >= threshold:
            updated[graduated_key] = candidate
            updated.pop(candidate_key, None)
        else:
            updated[candidate_key] = {"selector": candidate, "matches": matches}
    else:
        updated[candidate_key] = {"selector": candidate, "matches": 1}
    return True


def _demote_graduated_selector(
    host: str, graduated_key: str, *, reason: str,
) -> None:
    """Terminal failure handler: roll a graduated selector back to candidate.

    Used by ObstacleFail (and ExtractFail for ready_selector) when a
    previously-graduated selector apparently stopped working — the DOM
    drifted, the site redesigned, whatever. Demoting back to candidate
    with matches=0 means the next run will either re-graduate (if the
    selector STILL matches, we had a transient) or graduate a replacement
    (if a new candidate shows up). Saves us from being permanently
    trapped on a stale selector.

    Silent-fail: the goal is learning-loop hygiene, not a hard dependency.
    """
    if not host or not graduated_key:
        return
    try:
        resp = httpx.get(
            f"{_api_base()}/api/v1/scrape-profiles/",
            params={"filter[hostname]": host},
            headers=_api_headers(),
            timeout=10.0,
        )
        payload = resp.json() if resp.status_code == 200 else {}
        rows = payload.get("data") or []
        if not rows:
            return
        row = rows[0]
        profile_id = row.get("id")
        attrs = row.get("attributes") or {}
        existing = attrs.get("css-selectors") or attrs.get("css_selectors") or {}
    except Exception:
        logger.warning(
            "Demote %s: profile fetch failed host=%s", graduated_key,
            host, exc_info=True,
        )
        return

    stale_selector = existing.get(graduated_key)
    if not stale_selector:
        return  # nothing to demote

    updated = dict(existing)
    updated.pop(graduated_key, None)
    # Return selector to candidate pool with a fresh count so it has to
    # re-prove itself over the normal probation threshold. Key by the
    # canonical candidate name — "obstacle_click_selector" → "_obstacle_click_candidate".
    candidate_key = f"_{graduated_key.rsplit('_selector', 1)[0]}_candidate"
    updated[candidate_key] = {"selector": stale_selector, "matches": 0}

    if not profile_id:
        return
    try:
        httpx.patch(
            f"{_api_base()}/api/v1/scrape-profiles/{profile_id}/",
            json={
                "data": {
                    "type": "scrape-profile",
                    "id": str(profile_id),
                    "attributes": {"css-selectors": updated},
                }
            },
            headers={**_api_headers(), "Content-Type": "application/vnd.api+json"},
            timeout=10.0,
        )
        logger.info(
            "Demoted %s on %s (reason=%s): %s → candidate pool",
            graduated_key, host, reason, stale_selector,
        )
    except Exception:
        logger.warning(
            "Demote %s: PATCH failed profile_id=%s", graduated_key,
            profile_id, exc_info=True,
        )


@dataclass
class ResolveApplyUrl(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    """Phase 2: resolve the apply destination (the URL behind the
    posting's "Apply" button). Reads ``profile.apply_resolver_config``
    and tries internal markers → link selectors → button selectors. The
    result PATCHes ``/scrapes/{id}/apply-url/`` which writes through to
    the JobPost.

    Routing: on the browser-tier path (default), terminates the graph
    with End. On the extension-direct fast path (source_mode=
    'extension-direct'), routes to PersistJobPost instead — apply_url
    is null in the payload so we resolve it best-effort first, then
    persist the JobPost. No-op when state._browser_page is None.
    """

    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> Union[End[dict], PersistJobPost]:
        from .nodes_scrape import _patch_scrape_status
        from .apply_resolver import resolve_apply_url, scan_apply_candidates

        started = time.time()
        state = ctx.state

        page = getattr(state, "_browser_page", None)
        config: dict | None = None
        if state.profile and isinstance(state.profile, dict):
            config = state.profile.get("apply_resolver_config")

        try:
            result = await resolve_apply_url(page, config)
        except Exception:
            logger.warning(
                "ResolveApplyUrl resolver crashed scrape_id=%s",
                state.scrape_id, exc_info=True,
            )
            result = {
                "apply_url": None,
                "apply_url_status": "failed",
                "reason": "resolver_crashed",
            }

        # Phase 3 learning loop: when the configured resolver missed,
        # scan the page for "Apply"-shaped candidates and pass them
        # along with the PATCH so they accumulate on the Scrape row.
        # No-op when the resolver succeeded (resolved/internal) — those
        # are confident outcomes that don't need refinement.
        candidates: list[dict] = []
        if result.get("apply_url_status") in ("unknown", "failed") and page is not None:
            try:
                candidates = await scan_apply_candidates(page)
            except Exception:
                logger.warning(
                    "ResolveApplyUrl candidate scan crashed scrape_id=%s",
                    state.scrape_id, exc_info=True,
                )

        if state.scrape_id:
            try:
                attrs: dict = {
                    "apply_url": result.get("apply_url"),
                    "apply_url_status": result.get("apply_url_status"),
                }
                if candidates:
                    attrs["apply_candidates"] = candidates
                resp = httpx.patch(
                    f"{_api_base()}/api/v1/scrapes/{state.scrape_id}/apply-url/",
                    json={"data": {"attributes": attrs}},
                    headers={**_api_headers(), "Content-Type": "application/json"},
                    timeout=10.0,
                )
                if resp.status_code >= 400:
                    logger.warning(
                        "ResolveApplyUrl PATCH failed status=%s body=%s",
                        resp.status_code, resp.text[:300],
                    )
            except Exception:
                logger.warning(
                    "ResolveApplyUrl PATCH errored scrape_id=%s",
                    state.scrape_id, exc_info=True,
                )

        # Fast-path branch: extension-direct scrapes route through
        # ResolveApplyUrl (no-op when no browser page) on their way to
        # PersistJobPost, which terminates the graph. Don't terminal-
        # close the scrape here — PersistJobPost handles status closeout
        # on the fast path.
        if state.source_mode == "extension-direct":
            trace_node(
                state, "ResolveApplyUrl", "PersistJobPost", started,
                payload={
                    "apply_url_status": result.get("apply_url_status"),
                    "reason": result.get("reason"),
                    "fast_path": True,
                },
            )
            return PersistJobPost()

        state.outcome = "success"
        note = (
            f"extracted job_post {state.job_post_id}"
            if state.job_post_id
            else f"extracted ({len(state.job_content or '')} chars)"
        )
        _patch_scrape_status(state.scrape_id, "completed", note=note)
        trace_node(
            state, "ResolveApplyUrl", "End", started,
            payload={
                "apply_url_status": result.get("apply_url_status"),
                "reason": result.get("reason"),
            },
        )
        return End({
            "outcome": "success",
            "job_post_id": state.job_post_id,
            "scrape_id": state.scrape_id,
            "apply_url": result.get("apply_url"),
            "apply_url_status": result.get("apply_url_status"),
        })


@dataclass
class ExtractFail(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> End[dict]:
        from .nodes_scrape import _patch_scrape_status
        from ._artifacts import capture_debug_artifact
        started = time.time()
        state = ctx.state
        state.outcome = "failure"
        state.failure_reason = state.failure_reason or "extraction"

        # Debug-artifact invariant — see ObstacleFail for the same
        # pattern. Extraction fails after Capture already ran, so html
        # is usually populated; the helper is still useful for the
        # screenshot (capture timing drift) and the rare path where
        # PersistScrape's PATCH failed and html is empty.
        page = getattr(state, "_browser_page", None)
        artifact_info: dict = {}
        try:
            artifact_info = await capture_debug_artifact(
                page, state, reason="extract_fail",
            )
        except Exception:
            logger.warning(
                "ExtractFail: debug artifact capture failed scrape_id=%s",
                state.scrape_id, exc_info=True,
            )

        _patch_scrape_status(state.scrape_id, "failed", note=state.failure_reason)
        trace_node(
            state, "ExtractFail", "End", started,
            payload=artifact_info or None,
        )
        return End({
            "outcome": "failure",
            "failure_reason": state.failure_reason,
            "scrape_id": state.scrape_id,
        })


__all__ = [
    "StartExtract",
    "Tier0CSS",
    "Tier1Mini",
    "Tier2Haiku",
    "Tier3Sonnet",
    "EvaluateExtraction",
    "ValidateExtraction",
    "PersistJobPost",
    "ReviewCompleteness",
    "UpdateProfile",
    "ResolveApplyUrl",
    "ExtractFail",
]
