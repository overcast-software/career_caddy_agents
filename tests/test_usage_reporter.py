"""Tests for lib.usage_reporter — payload format and error handling."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from lib.usage_reporter import report_usage


class FakeUsage:
    """Mimics pydantic-ai Usage object."""

    def __init__(self, request_tokens=100, response_tokens=50, total_tokens=150, requests=1):
        self.request_tokens = request_tokens
        self.response_tokens = response_tokens
        self.total_tokens = total_tokens
        self.requests = requests


@pytest.mark.asyncio
class TestReportUsage:
    async def test_payload_format(self):
        """Verify the JSON:API payload has underscored keys and correct type."""
        captured = {}

        async def mock_post(url, json=None, headers=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            resp = MagicMock()
            resp.status_code = 201
            return resp

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = mock_post

        with patch("lib.usage_reporter.httpx.AsyncClient", return_value=mock_client):
            await report_usage(
                api_token="test-token",
                agent_name="career_caddy_chat",
                model_name="openai:gpt-4o-mini",
                usage=FakeUsage(request_tokens=1000, response_tokens=500, total_tokens=1500, requests=2),
                trigger="chat",
                pipeline_run_id="abc-123",
                base_url="http://localhost:8000",
            )

        payload = captured["json"]
        assert payload["data"]["type"] == "ai-usage"
        attrs = payload["data"]["attributes"]
        # Keys must be underscored (not dasherized)
        assert "agent_name" in attrs
        assert "model_name" in attrs
        assert "request_tokens" in attrs
        assert "response_tokens" in attrs
        assert "total_tokens" in attrs
        assert "request_count" in attrs
        assert "pipeline_run_id" in attrs
        # Values
        assert attrs["agent_name"] == "career_caddy_chat"
        assert attrs["model_name"] == "openai:gpt-4o-mini"
        assert attrs["request_tokens"] == 1000
        assert attrs["response_tokens"] == 500
        assert attrs["total_tokens"] == 1500
        assert attrs["request_count"] == 2
        assert attrs["trigger"] == "chat"
        assert attrs["pipeline_run_id"] == "abc-123"

    async def test_auth_header(self):
        captured = {}

        async def mock_post(url, json=None, headers=None, timeout=None):
            captured["headers"] = headers
            resp = MagicMock()
            resp.status_code = 201
            return resp

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = mock_post

        with patch("lib.usage_reporter.httpx.AsyncClient", return_value=mock_client):
            await report_usage(
                api_token="my-jwt-token",
                agent_name="test",
                model_name="test",
                usage=FakeUsage(),
                trigger="chat",
            )

        assert captured["headers"]["Authorization"] == "Bearer my-jwt-token"
        assert captured["headers"]["Content-Type"] == "application/vnd.api+json"

    async def test_null_pipeline_run_id(self):
        captured = {}

        async def mock_post(url, json=None, headers=None, timeout=None):
            captured["json"] = json
            resp = MagicMock()
            resp.status_code = 201
            return resp

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = mock_post

        with patch("lib.usage_reporter.httpx.AsyncClient", return_value=mock_client):
            await report_usage(
                api_token="token",
                agent_name="test",
                model_name="test",
                usage=FakeUsage(),
                trigger="pipeline",
            )

        assert captured["json"]["data"]["attributes"]["pipeline_run_id"] is None

    async def test_handles_none_token_fields(self):
        """Usage objects may have None for token fields."""
        captured = {}

        async def mock_post(url, json=None, headers=None, timeout=None):
            captured["json"] = json
            resp = MagicMock()
            resp.status_code = 201
            return resp

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = mock_post

        usage = MagicMock()
        usage.request_tokens = None
        usage.response_tokens = None
        usage.total_tokens = None
        usage.requests = None

        with patch("lib.usage_reporter.httpx.AsyncClient", return_value=mock_client):
            await report_usage(
                api_token="token",
                agent_name="test",
                model_name="test",
                usage=usage,
                trigger="chat",
            )

        attrs = captured["json"]["data"]["attributes"]
        assert attrs["request_tokens"] == 0
        assert attrs["response_tokens"] == 0
        assert attrs["total_tokens"] == 0
        assert attrs["request_count"] == 1

    async def test_error_does_not_raise(self):
        """Usage reporting errors are swallowed — never break agent flow."""
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=ConnectionError("refused"))

        with patch("lib.usage_reporter.httpx.AsyncClient", return_value=mock_client):
            # Should not raise
            await report_usage(
                api_token="token",
                agent_name="test",
                model_name="test",
                usage=FakeUsage(),
                trigger="chat",
            )
