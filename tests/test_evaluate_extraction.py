"""Tests for the EvaluateExtraction node.

Covers three behaviours:

1. The partial-render escape hatch landed for LinkedIn SDUI: when the
   LLM emits the documented placeholder description ("[DESCRIPTION NOT
   CAPTURED — ...]") because the page only rendered the top card,
   EvaluateExtraction must let it through with title/company instead of
   tripping `thin_description` and escalating to higher tiers.
2. The anti-fabrication grounding gate: a description an LLM tier
   returned that does not appear in the captured source was invented,
   and must ESCALATE rather than pass. Before this gate existed a
   confident fabrication was the cheapest way for a model to satisfy
   EvaluateExtraction — inventing prose stopped the ladder that would
   have produced a real answer.
3. The honest-stub degrade: when escalation is exhausted and the only
   thing missing is the description, persist what we actually read
   (title, company) with a plainly-labelled stub and a `stub_reason`,
   so ReviewCompleteness can mark the JobPost incomplete. A refusal
   must be strictly better than a fabrication.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from scrape_graph.nodes_extract import (
    EvaluateExtraction,
    ExtractFail,
    Tier2Haiku,
    ValidateExtraction,
    _PARTIAL_RENDER_DESCRIPTION_PREFIX,
)
from scrape_graph.state import ScrapeGraphState, TierAttempt


def _run(node, state: ScrapeGraphState):
    ctx = SimpleNamespace(state=state)
    return asyncio.run(node.run(ctx))


def _state_with(
    parsed: dict, last_tier: str = "tier1", job_content: str | None = None,
) -> ScrapeGraphState:
    state = ScrapeGraphState(scrape_id=1, submitted_url="https://x.com/job/1")
    state.parsed = parsed
    # Default to a source that contains the description verbatim, so the
    # grounding gate is a no-op for tests that aren't about it.
    state.job_content = (
        job_content if job_content is not None
        else (parsed.get("description") or "")
    )
    state.tier_attempts.append(
        TierAttempt(tier=last_tier, model="x", produced_output=True)
    )
    return state


_LONG_DESC = " ".join(["word"] * 80)
_PLACEHOLDER = (
    "[DESCRIPTION NOT CAPTURED — LinkedIn page rendered only the top "
    "card; the job description body did not hydrate within the scrape "
    "window. Visit the link directly to read the full posting.]"
)

# Plausible-looking invented prose — schema-valid, well over
# _STUB_MIN_WORDS, and shaped exactly like a real posting. This is what
# a model hands back when the page gave it nothing and the required
# title/company fields leave it no way to say so.
_FABRICATED_DESC = " ".join([
    "We are seeking a talented Software Engineer to join our growing",
    "engineering team. In this role you will design, build and maintain",
    "scalable backend services, collaborate closely with product and",
    "design partners, and mentor junior engineers. The ideal candidate",
    "has strong fundamentals in distributed systems, experience with",
    "cloud infrastructure, and a track record of shipping reliable",
    "software. We offer competitive compensation, comprehensive health",
    "benefits, and a flexible hybrid working arrangement for all staff.",
])

# What the browser actually captured for jp rHeRo6qWCG / scrape
# X04b4IjnTi: the LinkedIn top card and page chrome, ~800 characters,
# with no description body anywhere in it. Long enough to clear
# ValidateExtraction's source-word floor, which is why the fabrication
# sailed through.
_TOP_CARD_ONLY = " ".join([
    "Software Engineer Siemens Bellevue, WA Remote Full-time",
    "Posted 2 weeks ago Over 100 applicants Apply Save",
    "Use AI to assess how you fit Sign in to see who Siemens has hired",
    "for this role See who you know Get notified about new Software",
    "Engineer jobs in Bellevue, WA Show more Show less LinkedIn",
    "Corporation 2026 About Accessibility User Agreement Privacy Policy",
    "Cookie Policy Copyright Policy Brand Policy Guest Controls",
    "Community Guidelines Language",
])


def test_evaluate_passes_normal_extraction():
    state = _state_with({
        "title": "Backend Engineer",
        "company_name": "Acme",
        "description": _LONG_DESC,
    })
    next_node = _run(EvaluateExtraction(), state)
    assert isinstance(next_node, ValidateExtraction)
    assert state.evaluation["passed"] is True
    assert state.evaluation["partial_render"] is False


def test_evaluate_rejects_thin_description():
    state = _state_with({
        "title": "Backend Engineer",
        "company_name": "Acme",
        "description": "We need a great engineer. Apply now.",
    })
    next_node = _run(EvaluateExtraction(), state)
    # last_tier=tier1 → escalates to Tier2Haiku.
    assert isinstance(next_node, Tier2Haiku)
    assert "thin_description" in state.evaluation["reasons"]
    assert state.evaluation["partial_render"] is False


def test_evaluate_accepts_partial_render_placeholder():
    """LinkedIn SDUI partial-render: the LLM was instructed by the
    profile's extraction_hints to emit the placeholder. The placeholder
    is short — well under _STUB_MIN_WORDS — but is intentional, not a
    thin extraction. Must pass through to ValidateExtraction."""
    state = _state_with({
        "title": "Forward Deployed Engineer",
        "company_name": "Magnitude Consulting",
        "description": _PLACEHOLDER,
    })
    next_node = _run(EvaluateExtraction(), state)
    assert isinstance(next_node, ValidateExtraction)
    assert state.evaluation["passed"] is True
    assert state.evaluation["partial_render"] is True
    assert state.evaluation["reasons"] == []


def test_partial_render_placeholder_is_recorded_as_a_stub():
    """The placeholder passes the gate, but it is NOT real content and
    the graph must say so. Without `stub_reason` the fact died here:
    PersistJobPost POSTs only `state.parsed`, so the JobPost landed on
    JobPost.complete's default=True and the post presented as a normal
    complete record — which also suppressed the extension's re-send
    affordance, disabling the repair path. jp rHeRo6qWCG is the worked
    example."""
    state = _state_with({
        "title": "Software Engineer",
        "company_name": "Siemens",
        "description": _PLACEHOLDER,
    })
    _run(EvaluateExtraction(), state)
    assert state.stub_reason == "partial_render"


def test_evaluate_partial_render_still_requires_title_and_company():
    """The placeholder waiver only covers the thin-description gate.
    Title and company must still be present."""
    state = _state_with({
        "title": "",
        "company_name": "Magnitude Consulting",
        "description": _PLACEHOLDER,
    })
    next_node = _run(EvaluateExtraction(), state)
    assert not isinstance(next_node, ValidateExtraction)
    assert "missing_title" in state.evaluation["reasons"]
    assert state.evaluation["partial_render"] is True


def test_unidentified_posting_still_terminates_after_tier2():
    """When we never identified the posting at all — no title, no
    company — there is nothing truthful to persist, so the ladder still
    ends at ExtractFail and the debug-artifact invariant fires."""
    state = _state_with(
        {"title": "", "company_name": "", "description": "tiny"},
        last_tier="tier2",
    )
    next_node = _run(EvaluateExtraction(), state)
    assert isinstance(next_node, ExtractFail)
    assert state.stub_reason is None


# --- anti-fabrication: the grounding gate ---------------------------------


def test_fabricated_description_escalates_instead_of_passing():
    """THE regression this gate exists for.

    A model handed a content-free capture has no way to say so — the api's
    ParsedJobData requires a non-empty title + company_name and the prompt
    offers no failure channel — so it invents a plausible posting. That
    fabrication is long, confident, and schema-valid, so the old gate
    (title + company + >= 60 words) PASSED it and stopped the ladder.
    Fabricating was strictly cheaper for the model than answering
    honestly: an honest empty answer escalated, an invented one did not.

    None of the fabricated prose appears in the captured source, so the
    grounding check catches it and escalates to the next tier.
    """
    state = _state_with(
        {
            "title": "Software Engineer",
            "company_name": "Siemens",
            "description": _FABRICATED_DESC,
        },
        last_tier="tier1",
        job_content=_TOP_CARD_ONLY,
    )
    next_node = _run(EvaluateExtraction(), state)

    assert isinstance(next_node, Tier2Haiku), (
        "an invented description must escalate, not terminate the ladder"
    )
    assert "ungrounded_description" in state.evaluation["reasons"]
    assert state.evaluation["grounding_ratio"] < 0.30


def test_real_description_is_grounded_and_passes():
    """The counterweight: a genuine extraction copies phrases out of the
    page, so it clears the grounding floor even though the captured
    source also carries page chrome the description omits."""
    real_desc = " ".join([
        "Siemens is hiring a Software Engineer to build and maintain the",
        "services behind our building automation platform. You will own",
        "backend components end to end, work with product managers on",
        "roadmap scoping, and help improve our deployment tooling. We are",
        "looking for someone with several years of production experience",
        "in a typed language and a pragmatic approach to testing. Siemens",
        "offers competitive compensation and a hybrid schedule.",
    ])
    state = _state_with(
        {
            "title": "Software Engineer",
            "company_name": "Siemens",
            "description": real_desc,
        },
        last_tier="tier1",
        job_content=f"{_TOP_CARD_ONLY} {real_desc} Show more Show less",
    )
    next_node = _run(EvaluateExtraction(), state)

    assert isinstance(next_node, ValidateExtraction)
    assert state.evaluation["reasons"] == []
    assert state.evaluation["grounding_ratio"] > 0.30
    assert state.stub_reason is None


def test_tier0_output_is_never_grounding_checked():
    """Tier 0 is deterministic bs4 / JSON-LD parsing of `state.html` — it
    cannot fabricate, and a JSON-LD description lives inside a <script>
    block that never reaches the visible-text `job_content`. Checking it
    here would reject good $0 extractions."""
    state = _state_with(
        {
            "title": "Software Engineer",
            "company_name": "Siemens",
            "description": _FABRICATED_DESC,
        },
        last_tier="tier0",
        job_content=_TOP_CARD_ONLY,
    )
    next_node = _run(EvaluateExtraction(), state)

    assert isinstance(next_node, ValidateExtraction)
    assert "ungrounded_description" not in state.evaluation["reasons"]
    assert state.evaluation["grounding_ratio"] is None


def test_grounding_skipped_when_no_source_was_captured():
    """No source means no verdict — the check must abstain rather than
    fail an extraction it cannot judge. ValidateExtraction's
    source-word floor owns the empty-capture case."""
    state = _state_with(
        {
            "title": "Software Engineer",
            "company_name": "Siemens",
            "description": _FABRICATED_DESC,
        },
        last_tier="tier1",
        job_content="",
    )
    next_node = _run(EvaluateExtraction(), state)

    assert isinstance(next_node, ValidateExtraction)
    assert state.evaluation["grounding_ratio"] is None


# --- the honest stub ------------------------------------------------------


def test_exhausted_ladder_stubs_rather_than_keeping_fabrication():
    """Doug's requirement: 'I'd rather it offer a boilerplate stub than
    try to trick the system.'

    Escalation is exhausted (tier2, tier3 disabled) and every tier came
    back with invented prose. We keep the title + company we genuinely
    read off the page, replace the invented description with a plainly
    labelled stub, and record why — so the failure is visible and the
    repair path stays open.
    """
    state = _state_with(
        {
            "title": "Software Engineer",
            "company_name": "Siemens",
            "description": _FABRICATED_DESC,
        },
        last_tier="tier2",
        job_content=_TOP_CARD_ONLY,
    )
    next_node = _run(EvaluateExtraction(), state)

    assert isinstance(next_node, ValidateExtraction)
    assert state.stub_reason == "ungrounded"
    description = state.parsed["description"]
    assert description.startswith(_PARTIAL_RENDER_DESCRIPTION_PREFIX)
    # The invented prose must be gone, not merely annotated.
    assert "distributed systems" not in description
    assert _FABRICATED_DESC not in description


def test_exhausted_ladder_stubs_a_thin_description():
    """Same degrade for the honest-but-useless case: the tiers returned
    something, it was never enough to be a description, and there is no
    higher tier left."""
    state = _state_with(
        {"title": "Software Engineer", "company_name": "Siemens",
         "description": "Great opportunity. Apply today."},
        last_tier="tier2",
    )
    next_node = _run(EvaluateExtraction(), state)

    assert isinstance(next_node, ValidateExtraction)
    assert state.stub_reason == "no_description"
    assert state.parsed["description"].startswith(
        _PARTIAL_RENDER_DESCRIPTION_PREFIX
    )


def test_stub_reason_resets_between_tiers():
    """EvaluateExtraction runs once per rung. A tier that escalated as a
    stub and then succeeded must not drag a stale `stub_reason` into
    ReviewCompleteness and mark a good post incomplete."""
    state = _state_with(
        {"title": "Software Engineer", "company_name": "Siemens",
         "description": _PLACEHOLDER},
        last_tier="tier1",
    )
    _run(EvaluateExtraction(), state)
    assert state.stub_reason == "partial_render"

    # Tier 2 comes back with the real body.
    real_desc = " ".join(["shipping reliable backend services daily"] * 12)
    state.parsed = {
        "title": "Software Engineer",
        "company_name": "Siemens",
        "description": real_desc,
    }
    state.job_content = real_desc
    state.tier_attempts.append(
        TierAttempt(tier="tier2", model="x", produced_output=True)
    )
    next_node = _run(EvaluateExtraction(), state)

    assert isinstance(next_node, ValidateExtraction)
    assert state.stub_reason is None
