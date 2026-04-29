"""Tests for lib.api_tools — ApiClient and tool functions."""

from unittest.mock import MagicMock

import pytest
import httpx
import yaml

from lib.api_tools import (
    ApiClient,
    APIResponse,
    CompanyData,
    JobPostCreate,
    TOOL_SHAPES,
    _relationships_to_counts,
    _respond,
    _slim_record,
)


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
        response.json.return_value = {"data": {"type": "company", "id": "1"}}
        result = yaml.safe_load(client._ok(response))
        # No outer envelope: the JSON:API body sits at the top.
        assert "success" not in result
        assert "status_code" not in result
        assert result["data"]["id"] == "1"

    def test_ok_201(self):
        client = ApiClient("http://test:8000", "jh_x")
        response = MagicMock(spec=httpx.Response)
        response.status_code = 201
        response.json.return_value = {"data": {"type": "company", "id": "2"}}
        result = yaml.safe_load(client._ok(response))
        assert result["data"]["id"] == "2"

    def test_ok_error(self):
        client = ApiClient("http://test:8000", "jh_x")
        response = MagicMock(spec=httpx.Response)
        response.status_code = 404
        response.text = "Not found"
        result = yaml.safe_load(client._ok(response))
        assert "404" in result["error"]
        assert result["status_code"] == 404

    def test_ok_truncates_long_errors(self):
        client = ApiClient("http://test:8000", "jh_x")
        response = MagicMock(spec=httpx.Response)
        response.status_code = 500
        response.text = "x" * 1000
        result = yaml.safe_load(client._ok(response))
        assert len(result["error"]) < 600

    def test_parse_returns_payload_on_2xx(self):
        client = ApiClient("http://test:8000", "jh_x")
        response = MagicMock(spec=httpx.Response)
        response.status_code = 200
        response.json.return_value = {"data": {"id": "9"}}
        payload, error, status = client._parse(response)
        assert payload == {"data": {"id": "9"}}
        assert error is None
        assert status == 200

    def test_parse_returns_error_on_non_2xx(self):
        client = ApiClient("http://test:8000", "jh_x")
        response = MagicMock(spec=httpx.Response)
        response.status_code = 404
        response.text = "Not found"
        payload, error, status = client._parse(response)
        assert payload is None
        assert "404" in error
        assert status == 404


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


# ---------------------------------------------------------------------------
# Slim-response helpers (PR #1 — present but unwired)
# ---------------------------------------------------------------------------


class TestRespond:
    def test_yaml_no_envelope(self):
        out = _respond({"foo": "bar"})
        loaded = yaml.safe_load(out)
        assert loaded == {"foo": "bar"}
        assert "success" not in loaded
        assert "status_code" not in loaded

    def test_preserves_insertion_order(self):
        # YAML dump should keep id/title before bulk fields so the agent
        # gets the identifying bits at the top of the response.
        out = _respond({"id": "5", "title": "Dev", "description": "a" * 200})
        first_key = out.split(":", 1)[0]
        assert first_key == "id"

    def test_error_key_only(self):
        out = _respond(None, error="boom")
        loaded = yaml.safe_load(out)
        assert loaded == {"error": "boom"}

    def test_error_status_attached_when_4xx(self):
        out = _respond(None, error="not found", status_code=404)
        loaded = yaml.safe_load(out)
        assert loaded == {"error": "not found", "status_code": 404}

    def test_error_status_dropped_when_2xx(self):
        out = _respond(None, error="weird", status_code=200)
        loaded = yaml.safe_load(out)
        assert "status_code" not in loaded


class TestRelationshipsToCounts:
    def test_array_becomes_int(self):
        record = {
            "type": "user", "id": "2",
            "relationships": {
                "scores": {"data": [{"type": "score", "id": "1"}, {"type": "score", "id": "2"}]},
            },
        }
        _relationships_to_counts(record)
        assert record["relationships"]["scores"] == 2

    def test_singleton_becomes_id(self):
        record = {
            "type": "job-post", "id": "10",
            "relationships": {
                "company": {"data": {"type": "company", "id": "78"}},
            },
        }
        _relationships_to_counts(record)
        assert record["relationships"]["company"] == "78"

    def test_null_becomes_none(self):
        record = {
            "type": "job-post", "id": "10",
            "relationships": {"company": {"data": None}},
        }
        _relationships_to_counts(record)
        assert record["relationships"]["company"] is None

    def test_no_relationships_key_is_noop(self):
        record = {"type": "user", "id": "1"}
        _relationships_to_counts(record)
        assert "relationships" not in record


class TestSlimRecord:
    def test_attrs_filter(self):
        record = {
            "type": "user", "id": "2",
            "attributes": {"username": "dough", "email": "x", "phone": "y", "address": "z"},
        }
        _slim_record(record, attrs=["username", "email"])
        assert record["attributes"] == {"username": "dough", "email": "x"}

    def test_attrs_none_keeps_all(self):
        record = {
            "type": "user", "id": "2",
            "attributes": {"username": "dough", "email": "x"},
        }
        _slim_record(record, attrs=None, relationships="omit")
        assert record["attributes"] == {"username": "dough", "email": "x"}

    def test_relationships_omit(self):
        record = {
            "type": "user", "id": "2",
            "relationships": {"scores": {"data": [1, 2, 3]}},
        }
        _slim_record(record, relationships="omit")
        assert "relationships" not in record

    def test_relationships_counts(self):
        record = {
            "type": "user", "id": "2",
            "relationships": {
                "scores": {"data": [{"id": "1"}, {"id": "2"}]},
                "company": {"data": {"id": "78"}},
            },
        }
        _slim_record(record, relationships="counts")
        assert record["relationships"]["scores"] == 2
        assert record["relationships"]["company"] == "78"

    def test_relationships_inline_default_passthrough(self):
        record = {
            "type": "user", "id": "2",
            "relationships": {"scores": {"data": [{"id": "1"}]}},
        }
        _slim_record(record, relationships="inline")
        assert record["relationships"]["scores"]["data"] == [{"id": "1"}]


class TestToolShapesAudit:
    """The audit dict is a maintenance contract — every tool the MCP servers
    expose should have an entry. Missing entries are how slimming silently
    skips a tool, so we lock the inventory in tests."""

    REQUIRED = {
        "get_current_user", "find_company_by_name", "find_job_post_by_link",
        "get_scrape_profile", "get_scrape_graph_trace",
        "get_companies", "search_companies",
        "get_job_posts", "search_job_posts",
        "get_job_applications", "get_applications_for_job_post",
        "get_scrapes", "get_scores", "get_resumes",
        "get_questions", "get_answers",
        "list_scrape_screenshots", "get_scrape_statuses",
        "get_career_data",
        "create_company", "create_job_post_with_company_check",
        "create_job_application", "create_scrape",
        "update_job_post", "update_job_application",
        "update_scrape", "update_scrape_profile",
        "score_job_post", "fetch_scrape_screenshot",
    }

    def test_every_required_tool_has_a_row(self):
        missing = self.REQUIRED - TOOL_SHAPES.keys()
        assert not missing, f"TOOL_SHAPES missing: {sorted(missing)}"

    def test_every_row_has_a_kind(self):
        valid_kinds = {"single", "list", "list_or_single",
                       "aggregate", "passthrough", "binary_meta"}
        for tool, spec in TOOL_SHAPES.items():
            assert spec.get("kind") in valid_kinds, (
                f"{tool} has invalid kind {spec.get('kind')!r}"
            )

    def test_relationships_value_is_a_known_mode(self):
        valid_modes = {"counts", "omit", "inline"}
        for tool, spec in TOOL_SHAPES.items():
            mode = spec.get("relationships")
            if mode is None:
                continue  # passthrough/aggregate/binary_meta don't set this
            assert mode in valid_modes, f"{tool}: relationships={mode!r}"
