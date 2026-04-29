"""Obstacle agent — resolves UI blockers on a live Playwright page.

Invoked only when the fast deterministic paths (`_try_rememberme_reauth`,
`_try_expand_truncations`) fail. Receives the current page text, free-text
hints from the ScrapeProfile, AND optionally a screenshot of the current
viewport — the LLM can reason over what it sees instead of inferring from
text alone. The latter matters for iframe'd login walls, image-only
captchas, and any obstacle whose visual cue outweighs its text.

Tools exposed to the agent:
- try_click(selector): click a CSS selector on the live page.
- get_text(): return the current visible body text to confirm the obstacle.
- verify_resolved(): independently run the same login-wall detector the
  caller uses — lets the agent self-correct inside its own loop instead
  of blind-clicking and hoping.

Scoped to those three — no fill/submit — to keep blast radius tiny.
Rejects clicks that look like sign-out, cancel, back, or close.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, BinaryContent, RunContext

from agents.agent_factory import get_model

logger = logging.getLogger(__name__)

OBSTACLE_MODEL_ENV = "OBSTACLE_AGENT_MODEL"

SYSTEM_PROMPT = """\
You are resolving an obstacle blocking a web scraper from reaching real page content.

You have three tools:
- try_click(selector): click a CSS selector on the live page. Returns the
  element text + a hint about how the DOM changed.
- get_text(): return the current visible body text.
- verify_resolved(): return {resolved: bool} — True when the obstacle is
  cleared and real content is visible.

Workflow:
1. Read the page text + screenshot (if provided) + site-specific hints.
2. Pick the single CSS selector most likely to clear the obstacle — a
   specific account's "Continue" button, a "See more" expander, an
   accept/continue on a cookie/consent banner, or whatever the hints point
   at.
3. Call try_click(selector).
4. Call verify_resolved(). If resolved, stop.
5. If not resolved, reason about WHY — read the fresh text via get_text(),
   pick a different selector, click once more. Max 3 clicks total.

Never click anything labelled sign out, log out, cancel, back, or close.
Prefer stable selectors (attributes, roles, classes) over bare tags.
Over-clicking is worse than failing — call verify_resolved after every
click so you stop the moment the page is usable.
"""

DISALLOWED_ACTION_WORDS = (
    "sign out", "signout", "log out", "logout",
    "cancel", " back", "close", "dismiss",
)


@dataclass
class _PageDeps:
    page: Any  # Playwright Page; kept untyped to avoid forcing the import here.


async def run_obstacle_agent(
    page,
    hints: str,
    page_text: str,
    screenshot_bytes: bytes | None = None,
    page_structure: str = "",
    extra_login_wall_signals: list[str] | None = None,
    max_clicks: int = 3,
) -> dict:
    """Run the obstacle agent against a live Playwright page.

    Args:
        page: Playwright Page.
        hints: free-text from ScrapeProfile.interaction_hints.
        page_text: pre-agent snapshot of the page body text (for prompt).
        screenshot_bytes: PNG bytes of the current viewport. When provided
            AND the configured model supports vision, the agent sees the
            page visually instead of reasoning from text alone. Pass None
            to skip — text-only fallback still works.
        page_structure: free-text DOM-shape guidance from
            ScrapeProfile.page_structure. Where the deterministic
            RememberMe path bails out (e.g. URL-pattern pre-condition
            mismatch), this is what arms the agent with the structural
            cues + selector list it needs to clear the obstacle.
        max_clicks: safety cap. Agent is instructed to stop earlier once
            verify_resolved() returns True.

    Returns:
        {"resolved": bool, "actions": [selector, ...],
         "verified": bool, "note": str}

    `verified` is True iff the agent itself called verify_resolved() and
    got a truthy response after its last click. The caller can still
    re-verify with its own tooling — `resolved` here only reflects the
    agent's internal view.
    """
    model = os.environ.get(OBSTACLE_MODEL_ENV) or get_model("browser_scraper")
    agent: Agent[_PageDeps, str] = Agent(
        model, deps_type=_PageDeps, system_prompt=SYSTEM_PROMPT,
    )
    clicks: list[str] = []
    verification_history: list[bool] = []

    @agent.tool
    async def try_click(ctx: RunContext[_PageDeps], selector: str) -> str:
        """Attempt to click the element matching `selector`."""
        if len(clicks) >= max_clicks:
            return "click_limit_reached"
        lowered = selector.lower()
        for word in DISALLOWED_ACTION_WORDS:
            if word in lowered:
                return f"rejected: selector contains {word!r}"
        try:
            el = await ctx.deps.page.query_selector(selector)
        except Exception as exc:
            return f"selector_error: {exc}"
        if not el:
            return "not_found"
        try:
            text = ((await el.inner_text()) or "").strip().lower()
        except Exception:
            text = ""
        for word in DISALLOWED_ACTION_WORDS:
            if word in text:
                return f"rejected: element text contains {word!r}"
        try:
            # force=True bypasses Playwright's actionability checks
            # (visible/stable/enabled). LinkedIn renders the
            # account-chooser inside a `glimmer`-class container that
            # animates the skeleton-loading effect indefinitely, so the
            # "wait for element to be stable" gate inside
            # scroll_into_view_if_needed and click() never resolves —
            # every right-selector click was timing out at 2s with
            # "waiting for element to be stable". The button is in the
            # DOM, visible, and clickable; we just have to tell
            # Playwright to skip its stability heuristic.
            await el.click(timeout=5_000, force=True)
            try:
                await ctx.deps.page.wait_for_load_state("networkidle", timeout=10_000)
            except Exception:
                pass
            await asyncio.sleep(1)
            clicks.append(selector)
            return f"clicked ({text or 'no-text'}) — now call verify_resolved() to confirm"
        except Exception as exc:
            return f"click_error: {exc}"

    @agent.tool
    async def get_text(ctx: RunContext[_PageDeps]) -> str:
        """Return current visible page text."""
        try:
            return await ctx.deps.page.inner_text("body")
        except Exception as exc:
            return f"error: {exc}"

    @agent.tool
    async def verify_resolved(ctx: RunContext[_PageDeps]) -> dict:
        """Check whether the obstacle has cleared.

        Reads current body text, runs the same login-wall detector the
        graph's `DetectObstacle` node uses, and returns
        {"resolved": bool, "text_len": int}. Call this after every
        try_click — stop as soon as it returns resolved=True.
        """
        try:
            text = await ctx.deps.page.inner_text("body")
        except Exception as exc:
            verification_history.append(False)
            return {"resolved": False, "reason": f"page_read_failed: {exc}"}
        try:
            # Imported lazily so this module doesn't have a hard dep on
            # mcp_servers at import time.
            from mcp_servers.browser_server import _detect_login_wall
            walled = bool(
                _detect_login_wall(text, extra_strong_signals=extra_login_wall_signals)
            )
        except Exception as exc:
            verification_history.append(False)
            return {"resolved": False, "reason": f"detector_failed: {exc}"}
        cleared = not walled
        verification_history.append(cleared)
        return {"resolved": cleared, "text_len": len(text)}

    # Keep text prompt compact — most obstacles are decidable from a few kB,
    # and vision (when available) carries the rest.
    snippet = page_text.strip()
    if len(snippet) > 4000:
        snippet = snippet[:4000] + "\n...[truncated]"

    structure_block = (
        f"Page-structure guidance (from ScrapeProfile.page_structure):\n"
        f"{page_structure}\n\n"
        if page_structure
        else ""
    )
    text_prompt = (
        f"{structure_block}"
        f"Site-specific hints for resolving obstacles:\n{hints or '(no hints on this site yet)'}\n\n"
        f"Current page visible text:\n---\n{snippet}\n---\n\n"
        "Resolve the obstacle. Verify after each click."
    )

    # Multi-modal input if we have a screenshot. pydantic-ai's BinaryContent
    # gets routed to the model as a vision-enabled part; text-only models
    # silently drop it (most providers do, per the BinaryContent docs).
    if screenshot_bytes:
        prompt: list[Any] | str = [
            BinaryContent(data=screenshot_bytes, media_type="image/png"),
            text_prompt,
        ]
    else:
        prompt = text_prompt

    try:
        result = await agent.run(prompt, deps=_PageDeps(page=page))
        note = getattr(result, "output", None) or getattr(result, "data", "") or ""
        verified = bool(verification_history and verification_history[-1])
        return {
            "resolved": bool(clicks) and verified,
            "actions": clicks,
            "verified": verified,
            "note": str(note)[:500],
        }
    except Exception as exc:
        logger.warning("obstacle agent run failed: %s", exc, exc_info=True)
        return {
            "resolved": False,
            "actions": clicks,
            "verified": False,
            "note": f"agent_error: {exc}",
        }
