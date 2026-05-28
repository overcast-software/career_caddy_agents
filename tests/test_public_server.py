"""Tests for mcp_servers.public_server — tool registration and security."""

import asyncio
import inspect

import pytest


class TestPublicServerTools:
    @pytest.fixture(autouse=True)
    def load_server(self):
        from mcp_servers.public_server import server
        self.server = server
        # `_list_tools()` is the internal that BYPASSES middleware — what
        # we want for the "is every tool registered?" assertion. The
        # request-handler `list_tools()` goes through the
        # StaffOnlyToolFilter middleware and would return the filtered
        # non-staff view; see TestStaffOnlyToolFilter below for that
        # surface's behavior.
        self.tools = asyncio.run(server._list_tools())
        self.tool_names = {t.name for t in self.tools}

    def test_tool_count(self):
        # 27→28 when find_duplicate_candidates was added — composite pre-POST
        # check (title + link or company) over existing primitives.
        # 28→31 when the scrape-profile-enhancer tools landed
        # (inspect_scrape_html, test_url_rewrite, find_selectors_for_text).
        # Asserts the REGISTERED total, not the filtered surface a
        # non-staff client sees (see TestStaffOnlyToolFilter for that).
        assert len(self.tools) == 31

    def test_has_all_expected_tools(self):
        expected = {
            "create_company", "find_company_by_name", "search_companies", "get_companies",
            "create_job_post_with_company_check", "find_job_post_by_link",
            "search_job_posts", "get_job_posts", "update_job_post",
            "get_duplicate_candidates", "find_duplicate_candidates",
            "create_job_application", "get_job_applications",
            "get_applications_for_job_post", "update_job_application",
            "get_career_data", "get_current_user",
            "create_scrape", "get_scrapes", "update_scrape",
            "list_scrape_screenshots", "fetch_scrape_screenshot",
            "get_scrape_graph_trace", "get_scrape_statuses",
            "get_scrape_profile", "update_scrape_profile",
            "score_job_post", "get_scores",
            # scrape-profile-enhancer support (BS4 trim + selector test
            # + url_rewrites dry-run, all backed by lib/scrape_inspector)
            "inspect_scrape_html", "test_url_rewrite", "find_selectors_for_text",
        }
        assert self.tool_names == expected

    def test_no_browser_tools(self):
        browser_names = {n for n in self.tool_names if "browser" in n or "scrape_page" in n or "navigate" in n}
        assert not browser_names

    def test_no_email_tools(self):
        email_names = {n for n in self.tool_names if "email" in n or "tag" in n or "notmuch" in n}
        assert not email_names


class TestStaffOnlyToolFilter:
    """The StaffOnlyToolFilter middleware hides three enhancer tools
    from non-staff `tools/list` responses.

    The api enforces authorization on every endpoint these tools call,
    so the filter is a UX / prompt-context concern — regular users'
    LLMs shouldn't burn tokens on tool definitions they can't usefully
    invoke. These tests pin the visibility contract; the api-side
    authorization is covered by api/job_hunting/tests.
    """

    @pytest.fixture(autouse=True)
    def load_server(self):
        from mcp_servers.public_server import server
        from mcp_servers.staff_tool_filter import STAFF_ONLY_TOOLS
        self.server = server
        self.staff_only = STAFF_ONLY_TOOLS

    def _list_via_middleware(self, *, is_staff: bool) -> set[str]:
        """Run the request-handler `list_tools()` with a stubbed
        access token. Returns the visible tool name set.
        """
        from unittest.mock import patch
        from fastmcp.server.auth import AccessToken
        access = AccessToken(
            token="jh_stub", client_id="stub",
            scopes=["read", "write"] + (["staff"] if is_staff else []),
            claims={"user_id": "stub", "is_staff": is_staff},
        )
        # Patch in BOTH the middleware module's import site (where the
        # filter calls `get_access_token`) AND any other read sites.
        with patch(
            "mcp_servers.staff_tool_filter.get_access_token",
            return_value=access,
        ):
            tools = asyncio.run(self.server.list_tools())
        return {t.name for t in tools}

    def test_non_staff_session_hides_enhancer_tools(self):
        visible = self._list_via_middleware(is_staff=False)
        assert not (visible & self.staff_only), (
            "non-staff client must not see the enhancer tools in tools/list"
        )

    def test_staff_session_sees_enhancer_tools(self):
        visible = self._list_via_middleware(is_staff=True)
        assert self.staff_only <= visible, (
            "staff client must see all three enhancer tools"
        )

    def test_filtered_count_drops_by_exactly_three(self):
        non_staff = self._list_via_middleware(is_staff=False)
        staff = self._list_via_middleware(is_staff=True)
        assert len(staff) - len(non_staff) == len(self.staff_only)

    def test_unauthenticated_session_treated_as_non_staff(self):
        """No AccessToken in scope → defaults to non-staff. The fallback
        is intentional: an unauthenticated probe shouldn't see the
        staff-only surface, even though api gates would catch any
        actual attempt to invoke."""
        # The base `server.list_tools()` (used by TestPublicServerTools
        # above through _list_tools) goes through middleware with no
        # active session — middleware should hide the enhancer tools.
        tools = asyncio.run(self.server.list_tools())
        visible = {t.name for t in tools}
        assert not (visible & self.staff_only)


class TestPublicServerSecurity:
    def test_no_browser_imports(self):
        """public_server.py import lines must not reference browser, email, or credentials."""
        import mcp_servers.public_server as mod
        source = inspect.getsource(mod)
        import_lines = [line.strip() for line in source.splitlines() if line.strip().startswith(("import ", "from "))]
        for line in import_lines:
            assert "browser_server" not in line, f"Bad import: {line}"
            assert "email_server" not in line, f"Bad import: {line}"
            assert "Credentials" not in line, f"Bad import: {line}"
            assert "secrets" not in line.lower(), f"Bad import: {line}"

    def test_no_cc_api_token_env_read(self):
        """public_server.py must not read CC_API_TOKEN from environment."""
        import mcp_servers.public_server as mod
        source = inspect.getsource(mod)
        code_lines = [line for line in source.splitlines() if not line.strip().startswith("#") and not line.strip().startswith('"""') and "Security" not in line]
        for line in code_lines:
            if "CC_API_TOKEN" in line and "os.environ" in line:
                pytest.fail(f"Reads CC_API_TOKEN from env: {line.strip()}")
