"""Tests for lib.api_tools.scrape_and_score."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from lib.api_tools import ApiClient, scrape_and_score


def _api_response(data, status_code=200):
    return json.dumps(
        {"success": True, "data": data, "error": None, "status_code": status_code},
        indent=2,
    )


def _api_error(message, status_code):
    return json.dumps(
        {"success": False, "data": None, "error": message, "status_code": status_code},
        indent=2,
    )


class _FakeResponse:
    def __init__(self, body: dict):
        self._body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self._body


def _scrape_body(scrape_id: int, status: str, job_post_id=None):
    rels = {}
    if job_post_id is not None:
        rels["job-post"] = {"data": {"type": "job-post", "id": str(job_post_id)}}
    return {
        "data": {
            "type": "scrape",
            "id": str(scrape_id),
            "attributes": {"status": status, "url": "https://x"},
            "relationships": rels,
        }
    }


@pytest.fixture
def api():
    return ApiClient("http://api:8000", "jh_test")


@pytest.fixture(autouse=True)
def _no_sleep():
    async def _instant(_s):
        return None

    with patch("lib.api_tools.asyncio.sleep", _instant):
        yield


@pytest.mark.asyncio
async def test_happy_path(api):
    create_resp = _api_response({"data": {"type": "scrape", "id": "42"}}, 202)
    score_resp = _api_response(
        {"data": {"type": "score", "id": "7", "attributes": {"status": "pending"}}},
        202,
    )

    with patch.object(ApiClient, "post", new=AsyncMock(side_effect=[create_resp, score_resp])), \
         patch("lib.api_tools._raw_get_scrape", new=AsyncMock(side_effect=[
             _scrape_body(42, "pending"),
             _scrape_body(42, "completed", job_post_id=9),
         ])):
        result = json.loads(await scrape_and_score(api, "https://x"))

    assert result["success"] is True
    data = result["data"]
    assert data["scrape_id"] == 42
    assert data["job_post_id"] == 9
    assert data["score_id"] == 7
    assert data["scores_url"].endswith("/job-posts/9/scores")


@pytest.mark.asyncio
async def test_scrape_failed(api):
    create_resp = _api_response({"data": {"type": "scrape", "id": "3"}}, 202)
    with patch.object(ApiClient, "post", new=AsyncMock(return_value=create_resp)), \
         patch("lib.api_tools._raw_get_scrape", new=AsyncMock(return_value=_scrape_body(3, "failed"))):
        result = json.loads(await scrape_and_score(api, "https://x"))
    assert result["success"] is False
    assert "failed" in result["error"]


@pytest.mark.asyncio
async def test_scrape_timeout(api):
    create_resp = _api_response({"data": {"type": "scrape", "id": "5"}}, 202)
    with patch.object(ApiClient, "post", new=AsyncMock(return_value=create_resp)), \
         patch("lib.api_tools._raw_get_scrape", new=AsyncMock(return_value=_scrape_body(5, "pending"))):
        result = json.loads(await scrape_and_score(api, "https://x", timeout=0.0))
    assert result["success"] is False
    assert "Timed out" in result["error"]


@pytest.mark.asyncio
async def test_late_job_post_linkage(api):
    create_resp = _api_response({"data": {"type": "scrape", "id": "11"}}, 202)
    score_resp = _api_response(
        {"data": {"type": "score", "id": "2", "attributes": {"status": "pending"}}},
        202,
    )
    with patch.object(ApiClient, "post", new=AsyncMock(side_effect=[create_resp, score_resp])), \
         patch("lib.api_tools._raw_get_scrape", new=AsyncMock(side_effect=[
             _scrape_body(11, "completed"),          # no job-post yet
             _scrape_body(11, "completed"),          # still none
             _scrape_body(11, "completed", job_post_id=22),
         ])):
        result = json.loads(await scrape_and_score(api, "https://x"))
    assert result["success"] is True
    assert result["data"]["job_post_id"] == 22


@pytest.mark.asyncio
async def test_hold_fallback_on_501(api):
    create_501 = _api_error("501 - disabled", 501)
    create_ok = _api_response({"data": {"type": "scrape", "id": "17"}}, 201)
    score_resp = _api_response(
        {"data": {"type": "score", "id": "1", "attributes": {"status": "pending"}}},
        202,
    )
    with patch.object(ApiClient, "post", new=AsyncMock(side_effect=[create_501, create_ok, score_resp])), \
         patch("lib.api_tools._raw_get_scrape", new=AsyncMock(return_value=_scrape_body(17, "completed", job_post_id=4))):
        result = json.loads(await scrape_and_score(api, "https://x"))
    assert result["success"] is True
    assert result["data"]["hold_fallback"] is True
    assert "hold-poller" in result["data"]["message"]


@pytest.mark.asyncio
async def test_explicit_resume_id(api):
    create_resp = _api_response({"data": {"type": "scrape", "id": "8"}}, 202)
    score_resp = _api_response(
        {"data": {"type": "score", "id": "99", "attributes": {"status": "pending"}}},
        202,
    )
    post_mock = AsyncMock(side_effect=[create_resp, score_resp])
    with patch.object(ApiClient, "post", new=post_mock), \
         patch("lib.api_tools._raw_get_scrape", new=AsyncMock(return_value=_scrape_body(8, "completed", job_post_id=3))):
        result = json.loads(await scrape_and_score(api, "https://x", resume_id=55))

    assert result["success"] is True
    # Second post call is the score — verify it carried the resume relationship.
    score_call = post_mock.await_args_list[1]
    payload = score_call.args[1] if len(score_call.args) > 1 else score_call.kwargs["payload"]
    rels = payload["data"]["relationships"]
    assert rels["resume"]["data"]["id"] == "55"
    assert rels["job-post"]["data"]["id"] == "3"
