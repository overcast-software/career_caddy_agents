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
    from .nodes_scrape import Capture

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
        "Capture",
    ]:
        from .nodes_scrape import Capture
        started = time.time()
        state = ctx.state
        page = getattr(state, "_browser_page", None)
        if page is None:
            trace_node(state, "DetectObstacle", "Capture", started)
            return Capture()
        walled = False
        try:
            from mcp_servers.browser_server import _detect_login_wall
            text = await page.inner_text("body")
            walled = bool(_detect_login_wall(text))
        except Exception:
            pass
        if not walled:
            trace_node(state, "DetectObstacle", "Capture", started)
            return Capture()
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
                succeeded = bool(
                    await _try_rememberme_reauth(page, (state.profile or {}))
                )
            except Exception:
                logger.debug("ObstacleRememberMe failed", exc_info=True)
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
        page = getattr(state, "_browser_page", None)
        if page:
            try:
                from agents.obstacle_agent import run_obstacle_agent
                result = await run_obstacle_agent(page, state.profile or {})
                if isinstance(result, dict):
                    selector = result.get("selector")
                    succeeded = bool(result.get("cleared"))
                state.candidate_obstacle_click_selector = selector
            except Exception:
                logger.debug("ObstacleAgent failed", exc_info=True)
        state.obstacle_history.append(
            ObstacleAttempt(
                node="ObstacleAgent",
                selector_tried=selector,
                succeeded=succeeded,
            )
        )
        nxt = "DetectObstacle" if succeeded else "ObstacleFail"
        trace_node(state, "ObstacleAgent", nxt, started)
        return DetectObstacle() if succeeded else ObstacleFail()


@dataclass
class ObstacleFail(BaseNode[ScrapeGraphState, None, dict]):  # type: ignore[no-redef]
    async def run(
        self, ctx: GraphRunContext[ScrapeGraphState, None]
    ) -> End[dict]:
        started = time.time()
        state = ctx.state
        state.outcome = "failure"
        state.failure_reason = state.failure_reason or "login_wall"
        trace_node(state, "ObstacleFail", "End", started)
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
