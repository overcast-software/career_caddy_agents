"""Regression tests for the ObstacleAgent graph node.

Pre-fix bug: the node called `run_obstacle_agent(page, state.profile or {})`
against a function whose signature is
`run_obstacle_agent(page, hints: str, page_text: str, max_clicks: int = 3)`.
Every call raised `TypeError` at argument binding, got silently swallowed by
an `except Exception: logger.debug(...)` block, and the node fell through to
ObstacleFail in ~20 ms without ever invoking the LLM. Scrape 164 reproduces
the symptom: ObstacleAgent → ObstacleFail with suspiciously short duration.

After the fix:
- signature is right
- return-shape keys are the ones `run_obstacle_agent` actually returns
  ("resolved" + "actions", not "cleared" + "selector")
- success requires post-click verification via `_detect_login_wall`,
  not just the agent's own `resolved` flag (a click landing ≠ wall clearing).
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture(autouse=True)
def _quarantine_browser_imports():
    """The node's `_detect_login_wall` import lazily pulls in
    lib.browser.*, which would break test_toolsets' security-isolation
    check if we left those modules in sys.modules between test modules.
    Snapshot + prune same pattern we use in test_browser_server_retry.py."""
    before = set(sys.modules)
    yield
    for name in list(sys.modules):
        if name in before:
            continue
        if name.startswith("lib.browser") or name.startswith("mcp_servers.browser_server"):
            sys.modules.pop(name, None)


from scrape_graph.nodes_obstacle import ObstacleAgent  # noqa: E402
from scrape_graph.state import ScrapeGraphState  # noqa: E402


class _FakePage:
    """Minimal Playwright-page stub — just enough for the node to read
    inner_text('body') twice (pre- and post-click)."""

    def __init__(self, pre_text: str, post_text: str | None = None):
        self._pre = pre_text
        self._post = post_text if post_text is not None else pre_text
        self._reads = 0

    async def inner_text(self, _selector: str) -> str:
        self._reads += 1
        return self._pre if self._reads == 1 else self._post


class _StubCtx:
    def __init__(self, state):
        self.state = state


def _make_state(profile=None):
    state = ScrapeGraphState(
        scrape_id=9999,
        submitted_url="https://linkedin.com/jobs/view/1",
        original_scrape_id=9999,
        profile=profile or {},
        source="poller",
    )
    # The real graph injects this from the poller via
    # _process_scrape_primary. Stub it directly for unit tests.
    state._browser_page = _FakePage(
        pre_text="Sign in to continue",
        post_text="Senior Engineer at Acme — full description…",
    )
    return state


@pytest.mark.asyncio
class TestObstacleAgentNode:
    async def test_agent_clicked_and_wall_cleared_routes_to_detect(self):
        """Agent clicks a button, the post-click page no longer trips
        _detect_login_wall → succeeded=True, route to DetectObstacle."""
        state = _make_state()
        agent_result = {
            "resolved": True,
            "actions": ["button.continue-as-doug"],
            "note": "clicked the continue button",
        }
        with patch(
            "agents.obstacle_agent.run_obstacle_agent",
            new=AsyncMock(return_value=agent_result),
        ):
            node = ObstacleAgent()
            out = await node.run(_StubCtx(state))
        # Post-click page was clean → route to DetectObstacle (not ObstacleFail).
        assert type(out).__name__ == "DetectObstacle"
        last = state.obstacle_history[-1]
        assert last.node == "ObstacleAgent"
        assert last.succeeded is True
        assert last.selector_tried == "button.continue-as-doug"

    async def test_agent_clicked_but_wall_persists_routes_to_fail(self):
        """Agent clicks, but the post-click page still shows the wall →
        succeeded=False, route to ObstacleFail."""
        state = _make_state()
        # Override page so post_text ALSO triggers login-wall detection.
        state._browser_page = _FakePage(
            pre_text="Sign in to continue",
            post_text="Sign in — please log in to continue",
        )
        agent_result = {
            "resolved": True,
            "actions": ["button.oops-wrong-button"],
            "note": "clicked something",
        }
        with patch(
            "agents.obstacle_agent.run_obstacle_agent",
            new=AsyncMock(return_value=agent_result),
        ):
            node = ObstacleAgent()
            out = await node.run(_StubCtx(state))
        assert type(out).__name__ == "ObstacleFail"
        assert state.obstacle_history[-1].succeeded is False

    async def test_agent_raises_typeerror_does_not_crash_node(self):
        """Pre-fix symptom: wrong signature → TypeError → swallowed silent.
        Post-fix: still routes to ObstacleFail (agent effectively failed),
        but the exception must now be LOGGED (warning) with a type name
        recorded in the graph payload."""
        state = _make_state()
        with patch(
            "agents.obstacle_agent.run_obstacle_agent",
            new=AsyncMock(side_effect=TypeError("bad call")),
        ):
            node = ObstacleAgent()
            out = await node.run(_StubCtx(state))
        assert type(out).__name__ == "ObstacleFail"
        assert state.obstacle_history[-1].succeeded is False

    async def test_agent_resolved_false_skips_verification(self):
        """If the agent itself says it didn't click anything, we don't
        re-read the page just to confirm — go straight to Fail."""
        state = _make_state()
        # Count page reads: we should see exactly ONE (pre-agent fetch).
        agent_result = {"resolved": False, "actions": [], "note": "no plan"}
        with patch(
            "agents.obstacle_agent.run_obstacle_agent",
            new=AsyncMock(return_value=agent_result),
        ):
            node = ObstacleAgent()
            out = await node.run(_StubCtx(state))
        assert type(out).__name__ == "ObstacleFail"
        assert state._browser_page._reads == 1  # no verification round-trip
