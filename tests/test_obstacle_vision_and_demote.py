"""Tests for P3 obstacle upgrades:

- run_obstacle_agent now accepts a screenshot_bytes kwarg and routes it
  to the Agent as a BinaryContent part alongside the text prompt.
- The agent's return dict now carries a `verified` flag (True iff the
  agent itself called verify_resolved() and it returned resolved=True).
  `resolved` in the return is now (clicks-landed) AND (verified).
- `_demote_graduated_selector` rolls a graduated selector back to
  candidate-with-matches=0 so stale DOM-drifted selectors don't trap
  us in RememberMe-fails-every-time loops.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _quarantine_browser_imports():
    before = set(sys.modules)
    yield
    for name in list(sys.modules):
        if name in before:
            continue
        if name.startswith("lib.browser") or name.startswith("mcp_servers.browser_server"):
            sys.modules.pop(name, None)


class TestRunObstacleAgentVisionPassthrough:
    """run_obstacle_agent should attach a BinaryContent image to the prompt
    when screenshot_bytes is provided. Using a mocked Agent.run so we
    can inspect the exact prompt shape without a real LLM call."""

    @pytest.mark.asyncio
    async def test_screenshot_bytes_produce_multimodal_prompt(self):
        from agents import obstacle_agent as mod

        captured_prompts = []

        class _FakeAgentResult:
            output = "ok"

        class _FakeAgent:
            def __init__(self, *a, **kw):
                pass

            def tool(self, fn):
                return fn

            async def run(self, prompt, deps=None):
                captured_prompts.append(prompt)
                return _FakeAgentResult()

        with patch.object(mod, "Agent", _FakeAgent):
            result = await mod.run_obstacle_agent(
                page=MagicMock(),
                hints="click continue",
                page_text="Sign in",
                screenshot_bytes=b"\x89PNG\r\n\x1a\nfake",
            )

        assert len(captured_prompts) == 1
        p = captured_prompts[0]
        # multimodal → list with BinaryContent first, text prompt second
        assert isinstance(p, list), f"expected list prompt, got {type(p)!r}"
        assert len(p) == 2
        # first part should be an image carrier with media_type image/png
        assert getattr(p[0], "media_type", "") == "image/png"
        assert isinstance(p[1], str) and "Site-specific hints" in p[1]
        # No clicks happened, so resolved must be False even if agent "finished"
        assert result["resolved"] is False
        assert result["verified"] is False

    @pytest.mark.asyncio
    async def test_no_screenshot_uses_text_only_prompt(self):
        from agents import obstacle_agent as mod

        captured_prompts = []

        class _FakeAgentResult:
            output = "ok"

        class _FakeAgent:
            def __init__(self, *a, **kw):
                pass

            def tool(self, fn):
                return fn

            async def run(self, prompt, deps=None):
                captured_prompts.append(prompt)
                return _FakeAgentResult()

        with patch.object(mod, "Agent", _FakeAgent):
            await mod.run_obstacle_agent(
                page=MagicMock(),
                hints="click continue",
                page_text="Sign in",
                screenshot_bytes=None,
            )

        assert len(captured_prompts) == 1
        assert isinstance(captured_prompts[0], str), \
            "text-only path should pass a str prompt"

    @pytest.mark.asyncio
    async def test_agent_exception_becomes_structured_result(self):
        """Network/LLM failures return the dict contract, not raise."""
        from agents import obstacle_agent as mod

        class _FakeAgent:
            def __init__(self, *a, **kw):
                pass

            def tool(self, fn):
                return fn

            async def run(self, prompt, deps=None):
                raise RuntimeError("upstream 429")

        with patch.object(mod, "Agent", _FakeAgent):
            result = await mod.run_obstacle_agent(
                page=MagicMock(), hints="", page_text="",
            )

        assert result == {
            "resolved": False,
            "actions": [],
            "verified": False,
            "note": "agent_error: upstream 429",
        }


class TestDemoteGraduatedSelector:
    """_demote_graduated_selector is the learning-loop closing touch:
    when a graduated selector apparently stopped working, roll it back
    to candidate pool so the next run re-evaluates it (or a replacement)
    through the normal probation gate."""

    @pytest.mark.asyncio  # for api parity with the rest of the file
    async def test_rolls_graduated_selector_back_to_candidate(self):
        from lib.scrape_graph import nodes_extract as mod

        host = "linkedin.com"
        profile_id = 42

        fake_get_resp = MagicMock(
            status_code=200,
            json=MagicMock(return_value={
                "data": [
                    {
                        "id": profile_id,
                        "attributes": {
                            "css-selectors": {
                                "obstacle_click_selector": "button.continue",
                                "job_data": {"title": "h1"},
                            },
                        },
                    },
                ],
            }),
        )
        fake_patch_resp = MagicMock(status_code=200)

        with patch.object(mod, "httpx") as httpx_mod:
            httpx_mod.get = MagicMock(return_value=fake_get_resp)
            httpx_mod.patch = MagicMock(return_value=fake_patch_resp)
            mod._demote_graduated_selector(
                host, "obstacle_click_selector", reason="obstacle_fail",
            )

            assert httpx_mod.patch.called
            patch_kwargs = httpx_mod.patch.call_args.kwargs
            attrs = patch_kwargs["json"]["data"]["attributes"]["css-selectors"]
            # Graduated key gone, candidate pool restored with fresh count.
            assert "obstacle_click_selector" not in attrs
            assert attrs["_obstacle_click_candidate"] == {
                "selector": "button.continue",
                "matches": 0,
            }
            # Unrelated selector families untouched.
            assert attrs["job_data"] == {"title": "h1"}

    @pytest.mark.asyncio
    async def test_no_graduated_selector_noops(self):
        from lib.scrape_graph import nodes_extract as mod

        fake_get_resp = MagicMock(
            status_code=200,
            json=MagicMock(return_value={
                "data": [
                    {
                        "id": 1,
                        "attributes": {
                            "css-selectors": {
                                # No graduated obstacle_click_selector.
                                "_obstacle_click_candidate": {
                                    "selector": "button.a",
                                    "matches": 1,
                                },
                            },
                        },
                    },
                ],
            }),
        )

        with patch.object(mod, "httpx") as httpx_mod:
            httpx_mod.get = MagicMock(return_value=fake_get_resp)
            httpx_mod.patch = MagicMock()
            mod._demote_graduated_selector(
                "linkedin.com", "obstacle_click_selector", reason="obstacle_fail",
            )
            # No PATCH when there's nothing to demote.
            assert not httpx_mod.patch.called

    @pytest.mark.asyncio
    async def test_empty_host_noops(self):
        from lib.scrape_graph import nodes_extract as mod

        with patch.object(mod, "httpx") as httpx_mod:
            httpx_mod.get = MagicMock()
            mod._demote_graduated_selector(
                "", "obstacle_click_selector", reason="obstacle_fail",
            )
            assert not httpx_mod.get.called
