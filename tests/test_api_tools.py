"""Tests for lib.api_tools — ApiClient and tool functions."""

import json
from unittest.mock import MagicMock

import pytest
import httpx

from lib.api_tools import ApiClient, APIResponse, CompanyData, JobPostCreate


# ---------------------------------------------------------------------------
# ApiClient
# ---------------------------------------------------------------------------


class TestApiClient:
    def test_init(self):
        client = ApiClient("http://localhost:8000", "jh_test")
        assert client.base_url == "http://localhost:8000"
        assert client._headers == {"Authorization": "Bearer jh_test", "X-Forwarded-Proto": "https"}

    def test_ok_success(self):
        client = ApiClient("http://test:8000", "jh_x")
        response = MagicMock(spec=httpx.Response)
        response.status_code = 200
        response.json.return_value = {"data": {"id": "1"}}
        result = json.loads(client._ok(response))
        assert result["success"] is True
        assert result["status_code"] == 200
        assert result["data"]["data"]["id"] == "1"

    def test_ok_201(self):
        client = ApiClient("http://test:8000", "jh_x")
        response = MagicMock(spec=httpx.Response)
        response.status_code = 201
        response.json.return_value = {"data": {"id": "2"}}
        result = json.loads(client._ok(response))
        assert result["success"] is True

    def test_ok_error(self):
        client = ApiClient("http://test:8000", "jh_x")
        response = MagicMock(spec=httpx.Response)
        response.status_code = 404
        response.text = "Not found"
        result = json.loads(client._ok(response))
        assert result["success"] is False
        assert "404" in result["error"]

    def test_ok_truncates_long_errors(self):
        client = ApiClient("http://test:8000", "jh_x")
        response = MagicMock(spec=httpx.Response)
        response.status_code = 500
        response.text = "x" * 1000
        result = json.loads(client._ok(response))
        assert len(result["error"]) < 600


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class TestModels:
    def test_api_response_success(self):
        r = APIResponse(success=True, data={"foo": "bar"}, status_code=200)
        assert r.success
        assert r.data["foo"] == "bar"

    def test_api_response_error(self):
        r = APIResponse(success=False, error="something broke")
        assert not r.success
        assert r.error == "something broke"

    def test_company_data_validation(self):
        c = CompanyData(name="Acme Corp")
        assert c.name == "Acme Corp"
        assert c.website is None

    def test_company_data_rejects_empty_name(self):
        with pytest.raises(Exception):
            CompanyData(name="")

    def test_job_post_create_salary_validation(self):
        jp = JobPostCreate(title="Dev", company_id=1, salary_min=50000, salary_max=100000)
        assert jp.salary_min == 50000

    def test_job_post_create_rejects_negative_salary(self):
        with pytest.raises(Exception):
            JobPostCreate(title="Dev", company_id=1, salary_min=-1)

    def test_job_post_create_rejects_zero_company(self):
        with pytest.raises(Exception):
            JobPostCreate(title="Dev", company_id=0)
