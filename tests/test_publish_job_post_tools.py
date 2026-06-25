"""Tests for the ActivityPub publish/unpublish MCP tools (CC-60).

lib.api_tools.publish_job_post / unpublish_job_post are thin wrappers over
the api's owner-scoped POST /api/v1/job-posts/<id>/publish/ + /unpublish/
@actions. These tests assert the wrappers (1) forward to the right endpoint
with an empty body, (2) return the updated job-post resource, and (3)
surface a 403/404 from the api as a clean error result — all with the HTTP
layer mocked, never hitting a live api (mirrors test_attended_scrape_routing
+ test_api_tools _fake_api).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from lib.api_tools import publish_job_post, unpublish_job_post


def _api(result):
    """A fake ApiClient whose post_data returns the given (payload, error,
    status) tuple and records its call args."""
    api = MagicMock()
    api.post_data = AsyncMock(return_value=result)
    return api


def _job_post(id_, audience):
    return {
        "data": {
            "type": "job-post",
            "id": str(id_),
            "attributes": {"title": "Senior Engineer", "audience": audience},
            "relationships": {
                "company": {"data": {"type": "company", "id": "7"}},
            },
        }
    }


AS2_PUBLIC = "https://www.w3.org/ns/activitystreams#Public"


class TestPublishJobPost:
    @pytest.mark.asyncio
    async def test_forwards_to_publish_endpoint_with_empty_body(self):
        api = _api((_job_post(42, [AS2_PUBLIC]), None, 200))
        out = yaml.safe_load(await publish_job_post(api, 42))
        # Forwarded to the publish action with an empty POST body.
        assert api.post_data.await_args.args == (
            "/api/v1/job-posts/42/publish/", {},
        )
        # Returns the updated (now-public) job-post resource.
        assert out["data"]["id"] == "42"
        assert out["data"]["attributes"]["audience"] == [AS2_PUBLIC]
        assert "error" not in out

    @pytest.mark.asyncio
    async def test_403_from_api_surfaced_as_error(self):
        api = _api((None, "403 - {\"errors\": [{\"detail\": \"Forbidden\"}]}", 403))
        out = yaml.safe_load(await publish_job_post(api, 99))
        assert api.post_data.await_args.args[0] == "/api/v1/job-posts/99/publish/"
        assert "403" in out["error"]
        assert out["status_code"] == 403

    @pytest.mark.asyncio
    async def test_404_from_api_surfaced_as_error(self):
        api = _api((None, "404 - {\"errors\": [{\"detail\": \"Not found\"}]}", 404))
        out = yaml.safe_load(await publish_job_post(api, 12345))
        assert "404" in out["error"]
        assert out["status_code"] == 404


class TestUnpublishJobPost:
    @pytest.mark.asyncio
    async def test_forwards_to_unpublish_endpoint_with_empty_body(self):
        api = _api((_job_post(42, []), None, 200))
        out = yaml.safe_load(await unpublish_job_post(api, 42))
        assert api.post_data.await_args.args == (
            "/api/v1/job-posts/42/unpublish/", {},
        )
        # Returns the updated (now-private) job-post resource.
        assert out["data"]["id"] == "42"
        assert out["data"]["attributes"]["audience"] == []
        assert "error" not in out

    @pytest.mark.asyncio
    async def test_403_from_api_surfaced_as_error(self):
        api = _api((None, "403 - {\"errors\": [{\"detail\": \"Forbidden\"}]}", 403))
        out = yaml.safe_load(await unpublish_job_post(api, 99))
        assert api.post_data.await_args.args[0] == "/api/v1/job-posts/99/unpublish/"
        assert "403" in out["error"]
        assert out["status_code"] == 403

    @pytest.mark.asyncio
    async def test_404_from_api_surfaced_as_error(self):
        api = _api((None, "404 - {\"errors\": [{\"detail\": \"Not found\"}]}", 404))
        out = yaml.safe_load(await unpublish_job_post(api, 12345))
        assert "404" in out["error"]
        assert out["status_code"] == 404
