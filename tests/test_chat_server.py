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
