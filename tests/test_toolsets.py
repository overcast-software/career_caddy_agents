"""Tests for lib.toolsets — CareerCaddyToolset scoping, wrappers, and security."""

import inspect
from unittest.mock import AsyncMock

import pytest
from pydantic_ai import Agent, RunContext

from lib.toolsets import (
    CareerCaddyToolset,
    CareerCaddyDeps,
    TOOL_REGISTRY,
    SCOPES,
    _make_tool_wrapper,
)


# ---------------------------------------------------------------------------
# Scope configuration
# ---------------------------------------------------------------------------


class TestScopes:
    def test_all_scope_matches_registry(self):
        assert SCOPES["all"] == set(TOOL_REGISTRY.keys())

    def test_career_caddy_scope_has_28_tools(self):
        # 21 core CRUD + 7 Agent Wizard tools (show/edit resume + cover letter,
        # import_resume_from_url, edit_profile_onboarding, reconcile_onboarding).
        # AW intentionally does NOT get a tool for writing user account fields
        # (first_name, email, etc.) — those edits go through the settings UI.
        # Core CRUD count bumped 19→20 when create_question was added; bumped
        # 20→21 when get_current_user was added to give the chat agent a
        # first-class identity-lookup tool (so it stops calling get_career_data
        # for "what's my name?" questions).
        assert len(SCOPES["career_caddy"]) == 28

    def test_career_caddy_scope_includes_agent_wizard_tools(self):
        aw_tools = {
            "show_resume", "edit_resume",
            "show_cover_letter", "edit_cover_letter",
            "import_resume_from_url",
            "edit_profile_onboarding",
            "reconcile_onboarding",
        }
        assert aw_tools <= SCOPES["career_caddy"]

    def test_career_caddy_scope_excludes_user_account_writes(self):
        """AW must not have a tool for writing first_name / email / etc.
        Those writes go through /settings/profile, not through chat."""
        assert "edit_user_profile" not in SCOPES["career_caddy"]
        assert "edit_user_profile" not in TOOL_REGISTRY

    def test_every_scope_is_subset_of_all(self):
        for name, tools in SCOPES.items():
            assert tools <= SCOPES["all"], f"Scope '{name}' has unknown tools: {tools - SCOPES['all']}"

    def test_no_empty_scopes(self):
        for name, tools in SCOPES.items():
            assert len(tools) > 0, f"Scope '{name}' is empty"


# ---------------------------------------------------------------------------
# Toolset construction
# ---------------------------------------------------------------------------


class TestToolsetConstruction:
    def test_named_scope(self):
        ts = CareerCaddyToolset(scope="scoring")
        assert len(ts.tools) == len(SCOPES["scoring"])
        assert set(ts.tools) == SCOPES["scoring"]

    def test_custom_tool_list(self):
        ts = CareerCaddyToolset(scope=["score_job_post", "get_career_data"])
        assert set(ts.tools) == {"score_job_post", "get_career_data"}

    def test_all_scope(self):
        ts = CareerCaddyToolset(scope="all")
        assert len(ts.tools) == len(TOOL_REGISTRY)

    def test_toolset_id(self):
        ts = CareerCaddyToolset(scope="scoring", id="my-scorer")
        assert ts.id == "my-scorer"

    def test_default_id(self):
        ts = CareerCaddyToolset(scope="scoring")
        assert ts.id == "career-caddy"

    def test_invalid_scope_raises(self):
        with pytest.raises(KeyError):
            CareerCaddyToolset(scope="nonexistent")

    def test_invalid_tool_name_in_custom_list(self):
        with pytest.raises(KeyError):
            CareerCaddyToolset(scope=["fake_tool"])


# ---------------------------------------------------------------------------
# Wrapper signature preservation
# ---------------------------------------------------------------------------


class TestWrapperSignatures:
    def test_wrapper_drops_api_param(self):
        from lib import api_tools
        wrapper = _make_tool_wrapper(api_tools.score_job_post)
        sig = inspect.signature(wrapper)
        param_names = list(sig.parameters.keys())
        assert "api" not in param_names
        assert "ctx" == param_names[0]
        assert "job_post_id" in param_names

    def test_wrapper_preserves_docstring(self):
        from lib import api_tools
        wrapper = _make_tool_wrapper(api_tools.get_career_data)
        assert wrapper.__doc__ == api_tools.get_career_data.__doc__

    def test_wrapper_preserves_type_annotations(self):
        from lib import api_tools
        wrapper = _make_tool_wrapper(api_tools.score_job_post)
        sig = inspect.signature(wrapper)
        assert sig.parameters["job_post_id"].annotation is int

    def test_wrapper_preserves_defaults(self):
        from lib import api_tools
        wrapper = _make_tool_wrapper(api_tools.get_job_posts)
        sig = inspect.signature(wrapper)
        assert sig.parameters["id"].default is None
        assert sig.parameters["sort"].default is None

    def test_wrapper_ctx_annotation(self):
        from lib import api_tools
        wrapper = _make_tool_wrapper(api_tools.get_companies)
        sig = inspect.signature(wrapper)
        ctx_param = sig.parameters["ctx"]
        assert "RunContext" in str(ctx_param.annotation)
        assert "CareerCaddyDeps" in str(ctx_param.annotation)

    def test_all_registry_functions_wrap_cleanly(self):
        """Every function in TOOL_REGISTRY should produce a valid wrapper."""
        for name, fn in TOOL_REGISTRY.items():
            wrapper = _make_tool_wrapper(fn)
            sig = inspect.signature(wrapper)
            assert "ctx" in sig.parameters, f"{name} wrapper missing ctx param"
            assert "api" not in sig.parameters, f"{name} wrapper still has api param"
            assert wrapper.__doc__, f"{name} wrapper lost docstring"


# ---------------------------------------------------------------------------
# Agent integration
# ---------------------------------------------------------------------------


class TestAgentIntegration:
    def test_agent_accepts_toolset(self):
        agent = Agent(
            "test",
            deps_type=CareerCaddyDeps,
            output_type=str,
            toolsets=[CareerCaddyToolset(scope="scoring")],
        )
        assert agent.deps_type is CareerCaddyDeps
        assert len(agent._user_toolsets) == 1

    def test_agent_toolset_has_correct_tools(self):
        agent = Agent(
            "test",
            deps_type=CareerCaddyDeps,
            output_type=str,
            toolsets=[CareerCaddyToolset(scope="job_discovery")],
        )
        ts = agent._user_toolsets[0]
        assert set(ts.tools) == SCOPES["job_discovery"]

    def test_multiple_scopes_via_custom_list(self):
        combined = list(SCOPES["scoring"] | SCOPES["scrape_management"])
        agent = Agent(
            "test",
            deps_type=CareerCaddyDeps,
            output_type=str,
            toolsets=[CareerCaddyToolset(scope=combined)],
        )
        ts = agent._user_toolsets[0]
        assert set(ts.tools) == SCOPES["scoring"] | SCOPES["scrape_management"]


# ---------------------------------------------------------------------------
# Security: import isolation
# ---------------------------------------------------------------------------


class TestSecurityIsolation:
    def test_toolsets_does_not_import_browser(self):
        """Loading lib.toolsets must not pull in any browser modules.

        Run the check in a subprocess so sys.modules is pristine — an
        in-process check is order-dependent (any prior test importing
        mcp_servers.browser_server transitively loads lib.browser.* and
        contaminates the sys.modules snapshot for this assertion).
        """
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import lib.toolsets; "
                    "leaked = [m for m in sys.modules if 'lib.browser' in m]; "
                    "print('LEAKED:' + ','.join(leaked) if leaked else 'OK')"
                ),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "OK", result.stdout

    def test_api_tools_has_no_browser_imports(self):
        """api_tools.py import lines must not reference browser, credentials, or secrets."""
        import lib.api_tools as mod
        source = inspect.getsource(mod)
        import_lines = [line.strip() for line in source.splitlines() if line.strip().startswith(("import ", "from "))]
        for line in import_lines:
            assert "credentials" not in line.lower(), f"Bad import: {line}"
            assert "secrets" not in line.lower(), f"Bad import: {line}"
            assert "camoufox" not in line.lower(), f"Bad import: {line}"
            assert "browser" not in line.lower(), f"Bad import: {line}"

    def test_deps_has_no_credential_fields(self):
        """Credentials live in api_token only. All other fields must be
        plain contextual data (profile text, onboarding snapshot, page
        info) — no secrets / passwords / keys."""
        fields = {f.name for f in CareerCaddyDeps.__dataclass_fields__.values()}
        # api_token is the only credential-bearing field.
        assert "api_token" in fields
        assert "base_url" in fields
        # No field name should hint at a secondary credential surface.
        lowered = str(fields).lower()
        assert "secret" not in lowered
        assert "password" not in lowered
        assert "private_key" not in lowered
        assert "access_key" not in lowered


# ---------------------------------------------------------------------------
# Wrapper execution (mocked API calls)
# ---------------------------------------------------------------------------


class TestWrapperExecution:
    @pytest.mark.asyncio
    async def test_wrapper_passes_kwargs_through(self):
        """Wrapper should forward keyword args to the underlying function."""
        calls = []

        async def fake_score(api, job_post_id):
            calls.append({"base_url": api.base_url, "token": api._headers, "job_post_id": job_post_id})
            return '{"success": true}'

        wrapper = _make_tool_wrapper(fake_score)

        ctx = AsyncMock(spec=RunContext)
        ctx.deps = CareerCaddyDeps(api_token="jh_abc", base_url="http://localhost:9000")

        result = await wrapper(ctx, job_post_id=42)

        assert result == '{"success": true}'
        assert len(calls) == 1
        assert calls[0]["job_post_id"] == 42
        assert calls[0]["base_url"] == "http://localhost:9000"
        assert "jh_abc" in calls[0]["token"]["Authorization"]
