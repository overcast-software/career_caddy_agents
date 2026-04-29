"""Obstacle sub-graph — login walls, captchas, blocked pages.

Each node's run() has a concrete Union return type so pydantic-graph
can build edges. Tracing happens inline via trace_node().
"""
# ruff: noqa: F811
# Forward-declare stubs then redefine — see nodes_extract for rationale.
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

from pydantic_graph import BaseNode, End, GraphRunContext

from .state import ObstacleAttempt, ScrapeGraphState
from .tracing import trace_node

if TYPE_CHECKING:
    from .nodes_scrape import ResolveFinalUrl

logger = logging.getLogger(__name__)


_MAX_REMEMBER_ATTEMPTS = 2
_MAX_WAIT_RETRIES = 3
_MAX_AGENT_ATTEMPTS = 1


def _obstacle_count(state: ScrapeGraphState, node_name: str) -> int:
    return sum(1 for oa in state.obstacle_history if oa.node == node_name)


# Forward refs so Union annotations resolve at class-body time.
class ObstacleRememberMe(BaseNode[ScrapeGraphState, None, dict]):
    pass


class ObstacleWaitRetry(BaseNode[ScrapeGraphState, None, dict]):
    pass


class ObstacleAgent(BaseNode[ScrapeGraphState, None, dict]):
    pass


class ObstacleFail(BaseNode[ScrapeGraphState, None, dict]):
    pass


class DetectObstacle(BaseNode[ScrapeGraphState, None, dict]):
    pass


# --------- Real node implementations (shadowing the stubs) ---------

@dataclass
class DetectObstacle(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> Union[
        ObstacleRememberMe,
        ObstacleWaitRetry,
        ObstacleAgent,
        ObstacleFail,
        "ResolveFinalUrl",
    ]:
        from .nodes_scrape import ResolveFinalUrl
        started = time.time()
        state = ctx.state
        page = getattr(state, "_browser_page", None)
        if page is None:
            trace_node(state, "DetectObstacle", "ResolveFinalUrl", started)
            return ResolveFinalUrl()
        walled = False
        try:
            from mcp_servers.browser_server import _detect_login_wall
            text = await page.inner_text("body")
            walled = bool(_detect_login_wall(text))
        except Exception:
            # Changes routing — if detection itself fails we fall through
            # to ResolveFinalUrl so the scrape can still attempt content
            # capture. Loud so we notice when a host's DOM breaks our reader.
            logger.warning(
                "DetectObstacle: login-wall read failed scrape_id=%s",
                state.scrape_id, exc_info=True,
            )
        if not walled:
            trace_node(state, "DetectObstacle", "ResolveFinalUrl", started)
            return ResolveFinalUrl()
        if _obstacle_count(state, "ObstacleRememberMe") < _MAX_REMEMBER_ATTEMPTS:
            trace_node(state, "DetectObstacle", "ObstacleRememberMe", started)
            return ObstacleRememberMe()
        if _obstacle_count(state, "ObstacleWaitRetry") < _MAX_WAIT_RETRIES:
            trace_node(state, "DetectObstacle", "ObstacleWaitRetry", started)
            return ObstacleWaitRetry()
        if _obstacle_count(state, "ObstacleAgent") < _MAX_AGENT_ATTEMPTS:
            trace_node(state, "DetectObstacle", "ObstacleAgent", started)
            return ObstacleAgent()
        trace_node(state, "DetectObstacle", "ObstacleFail", started)
        return ObstacleFail()


@dataclass
class ObstacleRememberMe(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> Union[DetectObstacle, ObstacleWaitRetry]:
        started = time.time()
        state = ctx.state
        succeeded = False
        page = getattr(state, "_browser_page", None)
        if page:
            try:
                from mcp_servers.browser_server import _try_rememberme_reauth
                # All site-specific selector knowledge lives in
                # ScrapeProfile.css_selectors. Pass both the seeded
                # rememberme_candidates list and the probation-graduated
                # obstacle_click_selector so the helper can iterate
                # graduated-first then candidates.
                profile = state.profile or {}
                graduated = (
                    profile.get("obstacle_click_selector")
                    if isinstance(profile, dict) else None
                )
                candidates = (
                    profile.get("rememberme_candidates")
                    if isinstance(profile, dict) else None
                )
                succeeded = bool(
                    await _try_rememberme_reauth(
                        page,
                        profile_candidates=candidates if isinstance(candidates, list) else None,
                        graduated_selector=graduated if isinstance(graduated, str) else None,
                    )
                )
            except Exception:
                # Changes routing to ObstacleWaitRetry. Worth a warning.
                logger.warning(
                    "ObstacleRememberMe failed scrape_id=%s",
                    state.scrape_id, exc_info=True,
                )
        state.obstacle_history.append(
            ObstacleAttempt(node="ObstacleRememberMe", succeeded=succeeded)
        )
        nxt = "DetectObstacle" if succeeded else "ObstacleWaitRetry"
        trace_node(state, "ObstacleRememberMe", nxt, started)
        return DetectObstacle() if succeeded else ObstacleWaitRetry()


@dataclass
class ObstacleWaitRetry(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> Union[DetectObstacle, ObstacleAgent]:
        import asyncio
        started = time.time()
        state = ctx.state
        page = getattr(state, "_browser_page", None)
        if page:
            try:
                await asyncio.sleep(3.0)
            except Exception:
                pass
        state.obstacle_history.append(
            ObstacleAttempt(node="ObstacleWaitRetry", succeeded=True)
        )
        if _obstacle_count(state, "ObstacleWaitRetry") >= _MAX_WAIT_RETRIES \
                and _obstacle_count(state, "ObstacleAgent") < _MAX_AGENT_ATTEMPTS:
            trace_node(state, "ObstacleWaitRetry", "ObstacleAgent", started)
            return ObstacleAgent()
        trace_node(state, "ObstacleWaitRetry", "DetectObstacle", started)
        return DetectObstacle()


@dataclass
class ObstacleAgent(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> Union[DetectObstacle, ObstacleFail]:
        started = time.time()
        state = ctx.state
        succeeded = False
        selector: str | None = None
        agent_error: str | None = None
        page = getattr(state, "_browser_page", None)
        if page:
            try:
                from agents.obstacle_agent import run_obstacle_agent
                # Pre-fetch text and a viewport screenshot. The screenshot
                # is best-effort — if Playwright can't take one (detached
                # page, stale frame) we fall back to text-only and let
                # the LLM reason from that. Contract:
                #   {"resolved": bool, "actions": [selector, ...],
                #    "verified": bool, "note": str}
                try:
                    page_text = await page.inner_text("body")
                except Exception:
                    page_text = ""
                try:
                    screenshot_bytes = await page.screenshot(full_page=False)
                except Exception:
                    screenshot_bytes = None
                hints_raw = (state.profile or {}).get("interaction_hints")
                hints = hints_raw if isinstance(hints_raw, str) else ""
                result = await run_obstacle_agent(
                    page,
                    hints=hints,
                    page_text=page_text,
                    screenshot_bytes=screenshot_bytes,
                )
                if isinstance(result, dict):
                    actions = result.get("actions") or []
                    selector = actions[-1] if actions else None
                    # Agent now verifies internally via its own
                    # verify_resolved() tool. result["resolved"] already
                    # encodes (at-least-one-click) AND (last-verification-
                    # True). Re-check from the caller side as belt-and-
                    # suspenders — if the agent's verification raced a
                    # transient intermediate state we still catch it.
                    if bool(result.get("resolved")):
                        try:
                            fresh_text = await page.inner_text("body")
                        except Exception:
                            fresh_text = page_text
                        try:
                            from mcp_servers.browser_server import _detect_login_wall
                            succeeded = not _detect_login_wall(fresh_text)
                        except Exception:
                            # Trust the agent's own verification if the
                            # caller-side detector fails outright.
                            succeeded = bool(result.get("verified"))
                state.candidate_obstacle_click_selector = selector
            except Exception as exc:
                # Loud on purpose: this was the silent TypeError that caused
                # scrape 164 to fall straight through ObstacleAgent → ObstacleFail
                # in 20 ms without the LLM ever running.
                agent_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "ObstacleAgent failed scrape_id=%s: %s",
                    state.scrape_id, agent_error, exc_info=True,
                )
        state.obstacle_history.append(
            ObstacleAttempt(
                node="ObstacleAgent",
                selector_tried=selector,
                succeeded=succeeded,
            )
        )
        nxt = "DetectObstacle" if succeeded else "ObstacleFail"
        payload = {"selector_tried": selector, "succeeded": succeeded}
        if agent_error:
            payload["error"] = agent_error
        trace_node(state, "ObstacleAgent", nxt, started, payload=payload)
        return DetectObstacle() if succeeded else ObstacleFail()


@dataclass
class ObstacleFail(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> End[dict]:
        from .nodes_scrape import _patch_scrape_status
        from ._artifacts import capture_debug_artifact
        started = time.time()
        state = ctx.state
        state.outcome = "failure"
        state.failure_reason = state.failure_reason or "login_wall"

        # Debug-artifact invariant: before finalizing the Fail, snapshot
        # what the blocked page looked like so the admin UI has something
        # to render in the post-mortem. Best-effort.
        page = getattr(state, "_browser_page", None)
        artifact_info: dict = {}
        try:
            artifact_info = await capture_debug_artifact(
                page, state, reason="obstacle_fail",
            )
        except Exception:
            logger.warning(
                "ObstacleFail: debug artifact capture failed scrape_id=%s",
                state.scrape_id, exc_info=True,
            )

        # Learning-loop closing touch: if this host had a graduated
        # obstacle_click_selector AND we still couldn't clear the wall,
        # the selector has drifted. Demote it back to candidate so the
        # next run gets a fresh shot via RememberMe's heuristic list or
        # the agent.
        try:
            from urllib.parse import urlparse
            from .nodes_extract import _demote_graduated_selector
            host = urlparse(state.canonical_url or state.submitted_url or "").hostname or ""
            if host:
                _demote_graduated_selector(
                    host, "obstacle_click_selector",
                    reason="obstacle_fail",
                )
        except Exception:
            logger.warning(
                "ObstacleFail: selector demotion failed scrape_id=%s",
                state.scrape_id, exc_info=True,
            )

        _patch_scrape_status(state.scrape_id, "failed", note=state.failure_reason)
        trace_node(
            state, "ObstacleFail", "End", started,
            payload=artifact_info or None,
        )
        return End({
            "outcome": "failure",
            "failure_reason": state.failure_reason,
            "scrape_id": state.scrape_id,
        })


__all__ = [
    "DetectObstacle",
    "ObstacleRememberMe",
    "ObstacleWaitRetry",
    "ObstacleAgent",
    "ObstacleFail",
]
