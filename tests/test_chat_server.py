"""Tests for the chat server module — import safety and basic structure."""

import ast
import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest


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
        because the api returns 409 with errors[0].meta.existing_job_post_id.
        Documenting this in the prompt lets the agent react to the
        response shape rather than rely on remembering the discipline."""
        body = self._prompt()
        assert "409" in body
        assert "existing_job_post_id" in body


class TestFetchUserContext:
    """The merged /me/ fetch returns (profile_string, is_staff_bool).

    Replaces the earlier `_fetch_user_profile` + `_fetch_user_is_staff` pair
    which hit /me/ twice per chat turn and doubled the failure surface.
    Regression target: see todo "chat_server httpx.ReadTimeout on api v1 me
    — root cause for no-response fallback".
    """

    @staticmethod
    def _async_client_returning(resp_mock=None, raises=None):
        """Build a context-manager async client whose .get() returns RESP_MOCK
        or raises RAISES (one of the two; not both)."""
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        if raises is not None:
            mock_client.get = AsyncMock(side_effect=raises)
        else:
            mock_client.get = AsyncMock(return_value=resp_mock)
        return mock_client

    @pytest.mark.asyncio
    async def test_returns_tuple_with_profile_and_is_staff(self):
        chat_server = importlib.import_module("mcp_servers.chat_server")
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "data": {
                "id": "1",
                "type": "user",
                "attributes": {
                    "first_name": "Ada",
                    "last_name": "Lovelace",
                    "username": "ada",
                    "email": "ada@example.com",
                    "phone": "",
                    "address": "",
                    "linkedin": "",
                    "github": "",
                    "is_staff": True,
                },
            }
        }
        client = self._async_client_returning(resp_mock=resp)
        with patch.object(chat_server.httpx, "AsyncClient", return_value=client):
            profile, is_staff = await chat_server._fetch_user_context("jh_test")
        assert "First name: Ada" in profile
        assert "Email: ada@example.com" in profile
        # Empty fields are present as "(blank)" — load-bearing for the AW rule.
        assert "Phone: (blank)" in profile
        assert is_staff is True

    @pytest.mark.asyncio
    async def test_non_200_returns_degraded_tuple(self):
        chat_server = importlib.import_module("mcp_servers.chat_server")
        resp = MagicMock()
        resp.status_code = 401
        client = self._async_client_returning(resp_mock=resp)
        with patch.object(chat_server.httpx, "AsyncClient", return_value=client):
            profile, is_staff = await chat_server._fetch_user_context("jh_test")
        assert profile == "Could not load user profile."
        assert is_staff is False

    @pytest.mark.asyncio
    async def test_httpx_error_propagates_to_caller(self):
        chat_server = importlib.import_module("mcp_servers.chat_server")
        client = self._async_client_returning(raises=httpx.ReadTimeout("simulated"))
        with patch.object(chat_server.httpx, "AsyncClient", return_value=client):
            with pytest.raises(httpx.HTTPError):
                await chat_server._fetch_user_context("jh_test")

    @pytest.mark.asyncio
    async def test_hyphenated_is_staff_key_recognized(self):
        """JSON:API serializers sometimes emit `is-staff` instead of
        `is_staff`. Both must round-trip."""
        chat_server = importlib.import_module("mcp_servers.chat_server")
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "data": {
                "id": "1",
                "type": "user",
                "attributes": {
                    "first_name": "Grace",
                    "last_name": "Hopper",
                    "username": "ghopper",
                    "email": "grace@example.com",
                    "is-staff": True,
                },
            }
        }
        client = self._async_client_returning(resp_mock=resp)
        with patch.object(chat_server.httpx, "AsyncClient", return_value=client):
            _, is_staff = await chat_server._fetch_user_context("jh_test")
        assert is_staff is True


class TestChatEndpointEmitsRunErrorOnFetchFailure:
    """When `_fetch_user_context` raises httpx.HTTPError, the /chat
    StreamingResponse must emit a RunErrorEvent — NOT end the SSE stream
    silently with zero events.

    Pre-fix, an httpx.ReadTimeout on /me/ killed the generator before any
    yield, so the frontend saw `done=true` with empty `accumulated` and
    rendered the bare "(no response from agent)" fallback. Regression
    coverage for the parent todo "chat_server httpx.ReadTimeout on api v1
    me — root cause for no-response fallback".
    """

    def test_run_error_event_emitted_on_fetch_timeout(self, monkeypatch):
        from starlette.testclient import TestClient

        chat_server = importlib.import_module("mcp_servers.chat_server")

        async def _raising_fetch(_token):
            raise httpx.ReadTimeout("simulated")

        monkeypatch.setattr(chat_server, "_fetch_user_context", _raising_fetch)

        client = TestClient(chat_server.app)
        response = client.post(
            "/chat",
            json={"message": "hello", "token": "jh_test"},
        )

        assert response.status_code == 200
        body = response.text
        # The user-facing message is the actionable string from event_stream.
        # The frontend reads SSE events as `data: {json}` lines; assert on
        # the substring to stay tolerant of the ag_ui encoder's exact shape.
        assert "try again" in body.lower()
        # No content events — the stream errored out before the agent ran.
        assert "TEXT_MESSAGE_CONTENT" not in body
