"""Closed-state detection — pure logic split out of the DetectClosedState
node so each path is testable without pydantic-graph plumbing.

Three detection paths, in order of preference per host:

1. ``css_selectors`` — query the live Playwright page DOM. Deterministic,
   cheapest, host-curated. First non-None ``page.query_selector`` hit
   wins; the inner-text snippet of the matched element is returned as
   evidence.

2. ``text_phrases`` + ``learned_phrases`` — case-insensitive regex /
   substring scan against the captured ``job_content`` text.
   ``text_phrases`` are full regex patterns (curated by humans);
   ``learned_phrases`` come from the LLM-promotion path and are
   ``re.escape``-ed at match-time so a snippet like ``[CLOSED]`` isn't
   a runaway regex.

3. ``Haiku LLM`` — only when no host config is defined AND the captured
   text passes the cost guard (``len(text) >= min_chars_for_llm``).
   Returns a structured ``{is_closed, evidence_quote}`` and the caller
   substring-validates the quote against ``text`` before trusting it.
   Validated hits get **promoted** back into the host's
   ``closed_state_config.learned_phrases`` for future deterministic use.

Config lives inside the existing ``ScrapeProfile.css_selectors`` JSONB
blob (same pattern as ``apply_resolver_config``):

  closed_state_config:
    css_selectors: ["..."]
    text_phrases:  ["..."]
    learned_phrases: [{phrase, promoted_at, from_scrape_id}, ...]
    min_chars_for_llm: 1000   # optional; default below

Missing or empty config → fall through to LLM (cost-guarded).
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_MIN_CHARS_FOR_LLM = 1000
EVIDENCE_SNIPPET_MAX = 240   # cap stored evidence so a giant matched element doesn't bloat the row
TRACE_QUOTE_MAX = 120        # tighter cap for trace payload


async def detect_via_css(
    page: Any, selectors: list[str]
) -> tuple[list[dict], Optional[dict]]:
    """Probe each CSS selector against the live DOM. Returns
    ``(attempts, hit)`` where ``attempts`` is per-selector telemetry
    (whether each ran, matched, errored, with ms timing) and ``hit`` is
    ``{"selector": str, "snippet": str}`` for the first successful
    match — or ``None`` if nothing matched.

    Caller decides what to do on no-hit (fall through to phrase, then
    LLM). The two-tuple shape is so the trace payload can record every
    attempted selector even when none fired.
    """
    import time

    attempts: list[dict] = []
    hit: Optional[dict] = None
    for sel in selectors:
        started = time.time()
        attempt: dict = {"selector": sel, "matched": False, "duration_ms": 0}
        try:
            handle = await page.query_selector(sel)
        except Exception as exc:
            attempt["error"] = type(exc).__name__
            attempt["duration_ms"] = int((time.time() - started) * 1000)
            attempts.append(attempt)
            continue
        attempt["duration_ms"] = int((time.time() - started) * 1000)
        if handle is None:
            attempts.append(attempt)
            continue
        snippet = ""
        try:
            text = await handle.inner_text()
            snippet = (text or "").strip()[:EVIDENCE_SNIPPET_MAX]
        except Exception:
            pass
        attempt["matched"] = True
        attempt["snippet"] = snippet
        attempts.append(attempt)
        if hit is None:
            hit = {"selector": sel, "snippet": snippet}
        # Keep iterating so the trace records ALL attempted selectors,
        # not just the prefix up to the first match — useful for diff /
        # regret analysis ("would the third selector also have fired?").
    return attempts, hit


def detect_via_phrases(
    text: str,
    curated: list[str],
    learned: list[dict] | None = None,
) -> Optional[dict]:
    """Scan ``text`` for the first matching closed-state phrase.

    ``curated`` entries are treated as full regex patterns (case-
    insensitive). ``learned`` entries are dicts ``{phrase, ...}`` whose
    ``phrase`` is ``re.escape``-ed before compilation. Curated takes
    precedence over learned — a curated rule that fires shouldn't also
    surface a learned-phrase entry in the trace.

    Returns ``{"matched_pattern": str, "matched_substring": str,
    "source": "curated"|"learned"}`` on hit, or ``None``.
    """
    if not text:
        return None
    for pattern in curated:
        try:
            m = re.search(pattern, text, flags=re.IGNORECASE)
        except re.error:
            logger.warning("closed_state_detector: bad curated regex %r", pattern)
            continue
        if m:
            return {
                "matched_pattern": pattern,
                "matched_substring": (m.group(0) or "")[:EVIDENCE_SNIPPET_MAX],
                "source": "curated",
            }
    for entry in learned or []:
        if not isinstance(entry, dict):
            continue
        phrase = entry.get("phrase") or ""
        if not phrase:
            continue
        try:
            m = re.search(re.escape(phrase), text, flags=re.IGNORECASE)
        except re.error:
            continue
        if m:
            return {
                "matched_pattern": phrase,
                "matched_substring": (m.group(0) or "")[:EVIDENCE_SNIPPET_MAX],
                "source": "learned",
            }
    return None


async def detect_via_llm(text: str, model: str = "anthropic:claude-haiku-4-5") -> dict:
    """Ask Haiku whether the posting is closed and to quote the snippet
    that proves it. Returns a dict with telemetry — caller is
    responsible for validating ``evidence_quote in text`` before
    trusting ``is_closed``.

    Returns:
      {
        "model": str,
        "is_closed": bool | None,
        "evidence_quote": str | None,
        "duration_ms": int,
        "error": str | None,
      }

    Imports pydantic_ai lazily so test code can monkey-patch this whole
    function without pulling the dependency tree.
    """
    import time
    started = time.time()
    try:
        from pydantic import BaseModel
        from pydantic_ai import Agent

        class _Verdict(BaseModel):
            is_closed: bool
            evidence_quote: Optional[str] = None

        agent = Agent(
            model=model,
            output_type=_Verdict,
            system_prompt=(
                "You determine whether a job posting is closed (no longer "
                "accepting applications). Read the captured page text and "
                "answer with structured output:\n"
                "- is_closed=True ONLY if the page contains an explicit "
                "  closure phrase (e.g. 'no longer accepting applications', "
                "  'this position has been filled', 'applications are closed'). "
                "  Quote that exact phrase verbatim in evidence_quote.\n"
                "- is_closed=False if the page has an active 'Apply' button, "
                "  no closure phrase, or shows normal job-listing content.\n"
                "Do NOT infer closed-state from chrome like 'Promoted by hirer', "
                "'Save', 'Reposted N days ago', or footer/navigation strings."
            ),
        )
        result = await agent.run(text)
        verdict = result.output
        return {
            "model": model,
            "is_closed": bool(verdict.is_closed),
            "evidence_quote": (verdict.evidence_quote or "").strip() or None,
            "duration_ms": int((time.time() - started) * 1000),
            "error": None,
        }
    except Exception as exc:
        return {
            "model": model,
            "is_closed": None,
            "evidence_quote": None,
            "duration_ms": int((time.time() - started) * 1000),
            "error": f"{type(exc).__name__}: {exc}",
        }
