"""Tests for mcp_servers.public_server — tool registration and security."""

import asyncio
import inspect

import pytest


class TestPublicServerTools:
    @pytest.fixture(autouse=True)
    def load_server(self):
        from mcp_servers.public_server import server
        self.server = server
        self.tools = asyncio.get_event_loop().run_until_complete(server.list_tools())
        self.tool_names = {t.name for t in self.tools}

    def test_tool_count(self):
        assert len(self.tools) == 24

    def test_has_all_expected_tools(self):
        expected = {
            "create_company", "find_company_by_name", "search_companies", "get_companies",
            "create_job_post_with_company_check", "find_job_post_by_link",
            "search_job_posts", "get_job_posts", "update_job_post",
            "create_job_application", "get_job_applications",
            "get_applications_for_job_post", "update_job_application",
            "get_career_data", "get_current_user",
            "create_scrape", "get_scrapes", "update_scrape",
            "list_scrape_screenshots", "fetch_scrape_screenshot",
            "get_scrape_profile", "update_scrape_profile",
            "score_job_post", "get_scores",
        }
        assert self.tool_names == expected

    def test_no_browser_tools(self):
        browser_names = {n for n in self.tool_names if "browser" in n or "scrape_page" in n or "navigate" in n}
        assert not browser_names

    def test_no_email_tools(self):
        email_names = {n for n in self.tool_names if "email" in n or "tag" in n or "notmuch" in n}
        assert not email_names


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
