"""CC-101 regression: no MCP tool advertises an integer id param for a
NanoID-keyed resource.

CC-77 swapped 9 models (JobPost, Company, Scrape, Score, ScrapeProfile,
Answer, CoverLetter, Question, JobApplication, Resume) to opaque 10-char
NanoID string PKs. CC-87 flipped the int id params on `public_server.py`
only. This suite covers the surfaces CC-87 missed:

  * the prod **chat server** tools (pydantic-ai derives their schema from
    the `lib/api_tools.py` function signatures via `lib/toolsets.py`'s
    `_make_tool_wrapper`, so an int hint there reaches the chat model);
  * the local **career_caddy_server** FastMCP tools;
  * the runtime int()-coercions in `api_tools` that raised on a NanoID even
    once the param hint was a string.

`user_id` stays int (auth.User keeps an integer PK) and pagination / value
params (page, per_page, salary_*, …) stay int.
"""

import asyncio
import inspect
import types
import typing
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from lib import api_tools
from lib.toolsets import TOOL_REGISTRY, _make_tool_wrapper


# Params that legitimately stay integer — they key to nothing NanoID.
_INT_VALUE_PARAMS = {
    "page", "per_page", "page_size",
    "salary_min", "salary_max",
    "max_chars", "max_matches", "max_results",
}


def _annotation_types(annotation) -> set:
    """Flatten an annotation to the set of concrete types it allows,
    walking Optional[...] / X | None unions."""
    if annotation is inspect.Parameter.empty:
        return set()
    origin = typing.get_origin(annotation)
    if origin is typing.Union or origin is getattr(types, "UnionType", object()):
        out: set = set()
        for arg in typing.get_args(annotation):
            out |= _annotation_types(arg)
        return out
    return {annotation}


def _is_id_param(name: str) -> bool:
    """An id-like param addresses a resource by primary key."""
    return name == "id" or name.endswith("_id")


# ---------------------------------------------------------------------------
# Chat server — tools built by lib/toolsets._make_tool_wrapper from api_tools
# ---------------------------------------------------------------------------


class TestChatToolsetNanoIdParams:
    def test_no_id_param_is_integer_typed(self):
        """The regression guard: no id param keyed to a NanoID resource may be
        int-typed, or the chat model strips the NanoID before the call."""
        offenders = []
        for name, fn in TOOL_REGISTRY.items():
            sig = inspect.signature(_make_tool_wrapper(fn))
            for pname, param in sig.parameters.items():
                if pname == "ctx" or not _is_id_param(pname) or pname == "user_id":
                    continue
                if int in _annotation_types(param.annotation):
                    offenders.append((name, pname))
        assert not offenders, (
            "NanoID id params must not be int-typed: " + repr(offenders)
        )

    def test_headline_id_params_are_string(self):
        """Positive coverage that the ticket's named params are str-typed."""
        expected = {
            "score_job_post": ["job_post_id"],
            "update_job_post": ["job_post_id", "company_id"],
            "get_job_posts": ["id"],
            "get_scrapes": ["id"],
            "update_scrape": ["scrape_id"],
            "create_answer": ["question_id"],
            "update_answer": ["answer_id"],
            "show_resume": ["resume_id"],
            "get_duplicate_candidates": ["job_post_id"],
        }
        for name, params in expected.items():
            sig = inspect.signature(_make_tool_wrapper(TOOL_REGISTRY[name]))
            for pname in params:
                types_ = _annotation_types(sig.parameters[pname].annotation)
                assert str in types_ and int not in types_, (name, pname, types_)

    def test_pagination_and_value_params_stay_integer(self):
        for name, fn in TOOL_REGISTRY.items():
            sig = inspect.signature(_make_tool_wrapper(fn))
            for pname, param in sig.parameters.items():
                if pname in _INT_VALUE_PARAMS:
                    types_ = _annotation_types(param.annotation)
                    assert int in types_ and str not in types_, (name, pname, types_)


# ---------------------------------------------------------------------------
# career_caddy_server (local FastMCP) — JSON-schema assertion
# ---------------------------------------------------------------------------


def _schema_types(prop: dict) -> set:
    """Collect every JSON-schema 'type' on a property, walking anyOf/oneOf so
    Optional[...] (type | null) unions are handled."""
    out = set()
    if isinstance(prop.get("type"), str):
        out.add(prop["type"])
    for branch in (prop.get("anyOf") or []) + (prop.get("oneOf") or []):
        if isinstance(branch, dict) and isinstance(branch.get("type"), str):
            out.add(branch["type"])
    return out


class TestCareerCaddyServerNanoIdParams:
    @pytest.fixture(autouse=True)
    def load_tools(self):
        from mcp_servers.career_caddy_server import server
        tools = asyncio.run(server._list_tools())
        self.props = {t.name: t.parameters["properties"] for t in tools}

    def test_no_id_param_is_json_integer(self):
        offenders = []
        for tool_name, params in self.props.items():
            for pname, schema in params.items():
                if not _is_id_param(pname) or pname == "user_id":
                    continue
                stypes = _schema_types(schema)
                if "integer" in stypes or "string" not in stypes:
                    offenders.append((tool_name, pname, sorted(stypes)))
        assert not offenders, (
            "career_caddy_server NanoID id params must be JSON string, not "
            f"integer: {offenders}"
        )


# ---------------------------------------------------------------------------
# Runtime: the int()-coercions that raised on a NanoID even with a str hint
# ---------------------------------------------------------------------------


class TestApiToolsNanoIdRuntime:
    @pytest.mark.asyncio
    async def test_create_job_post_resolves_existing_nanoid_company(self):
        """The existing-company branch used `int(company.id)` — a ValueError
        on a NanoID. The company's NanoID must now ride the relationship."""
        api = MagicMock()
        api.get_data = AsyncMock(side_effect=[
            ({"data": []}, None, 200),  # dup check: no existing link
            (
                {"data": [{"type": "company", "id": "Co8nanoidX",
                           "attributes": {"name": "Acme"}}]},
                None, 200,
            ),  # company search: a hit
        ])
        api.post_data = AsyncMock(return_value=(
            {"data": {"type": "job-post", "id": "Jp9nanoid00",
                      "attributes": {"title": "Dev"}}},
            None, 201,
        ))
        out = yaml.safe_load(await api_tools.create_job_post_with_company_check(
            api, title="Dev", company_name="Acme", url="https://x/job/1",
        ))
        assert "error" not in out
        body = api.post_data.await_args.args[1]
        company_ref = body["data"]["relationships"]["company"]["data"]
        assert company_ref["id"] == "Co8nanoidX"

    @pytest.mark.asyncio
    async def test_create_job_application_forwards_nanoid(self):
        api = MagicMock()
        api.post_data = AsyncMock(return_value=(
            {"data": {"type": "job-application", "id": "Ja1nanoid00",
                      "attributes": {"status": "applied"}}},
            None, 201,
        ))
        out = yaml.safe_load(
            await api_tools.create_job_application(api, "Jp9nanoid00")
        )
        assert "error" not in out
        body = api.post_data.await_args.args[1]
        jp_ref = body["data"]["relationships"]["job-post"]["data"]
        assert jp_ref["id"] == "Jp9nanoid00"

    @pytest.mark.asyncio
    async def test_create_job_application_rejects_empty_id(self):
        """The old `<= 0` numeric guard raised TypeError on a str; the new
        guard rejects only empty/missing ids and never posts."""
        api = MagicMock()
        api.post_data = AsyncMock()
        out = yaml.safe_load(await api_tools.create_job_application(api, ""))
        assert "Invalid job_post_id" in out["error"]
        api.post_data.assert_not_awaited()
