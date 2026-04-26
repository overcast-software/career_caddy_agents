"""Tests for the chat server module — import safety and basic structure."""

import ast
import importlib


class TestChatServerSecurity:
    """Ensure the chat server has the same security invariants as public_server."""

    def _get_imports(self):
        """Parse the source and extract all import strings."""
        mod = importlib.import_module("mcp_servers.chat_server")
        tree = ast.parse(open(mod.__file__).read())
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
        return imports

    def test_no_browser_imports(self):
        """chat_server must not import browser-related modules."""
        imports = self._get_imports()
        for imp in imports:
            assert "browser" not in imp.lower(), f"Forbidden import: {imp}"
            assert "email_server" not in imp, f"Forbidden import: {imp}"

    def test_no_secrets_import(self):
        """chat_server must not import credentials or secrets modules."""
        imports = self._get_imports()
        for imp in imports:
            assert "credentials" not in imp.lower(), f"Forbidden import: {imp}"
            assert "secrets" not in imp.lower() or imp == "secrets", f"Forbidden: {imp}"

    def test_starlette_app_exists(self):
        """chat_server exposes a Starlette ASGI app."""
        mod = importlib.import_module("mcp_servers.chat_server")
        assert hasattr(mod, "app")

    def test_chat_route_exists(self):
        """The /chat route is registered."""
        mod = importlib.import_module("mcp_servers.chat_server")
        routes = [r.path for r in mod.app.routes]
        assert "/chat" in routes

    def test_health_route_exists(self):
        """The /health route is registered."""
        mod = importlib.import_module("mcp_servers.chat_server")
        routes = [r.path for r in mod.app.routes]
        assert "/health" in routes


class TestChatSystemPromptDuplicateRule:
    """Regression: SYSTEM_PROMPT enforces find_job_post_by_link before any
    create path, including when the URL is embedded in pasted text. See
    todo.org "Chat: detect duplicate JobPost by link" [#A] :bug:.
    """

    def _prompt(self) -> str:
        mod = importlib.import_module("mcp_servers.chat_server")
        return mod.SYSTEM_PROMPT

    def test_mentions_find_job_post_by_link(self):
        assert "find_job_post_by_link" in self._prompt()

    def test_explicitly_covers_pasted_text_with_url(self):
        """The bug class is the agent skipping the dedup check when the URL
        is buried inside a longer message. The prompt must say so verbatim
        — vague instructions failed in prod."""
        body = self._prompt().lower()
        assert "pasted text" in body
        assert "embedded" in body or "buried" in body or "inside" in body

    def test_duplicate_hit_navigates_via_propose_actions(self):
        """On a find_job_post_by_link hit, the agent must propose a navigate
        action rather than create a duplicate or a redundant scrape."""
        body = self._prompt()
        assert "propose_actions" in body
        assert "/job-posts/" in body

    def test_documents_api_side_enforcement(self):
        """The prompt must reflect that create_scrape itself enforces
        dedup-by-link. A skipped pre-check cannot produce a duplicate
        because the api returns meta.duplicate=true. Documenting this in
        the prompt lets the agent react to the response shape rather than
        rely on remembering the discipline."""
        body = self._prompt()
        assert "meta.duplicate" in body
        assert "existing_job_post_id" in body
