"""Tests for mcp_servers.public_server — tool registration and security."""

import asyncio
import inspect

import pytest
import yaml


class TestPublicServerTools:
    @pytest.fixture(autouse=True)
    def load_server(self):
        from mcp_servers.public_server import server
        self.server = server
        # `_list_tools()` is the internal that BYPASSES middleware — what
        # we want for the "is every tool registered?" assertion. The
        # request-handler `list_tools()` goes through the
        # StaffOnlyToolFilter middleware and would return the filtered
        # non-staff view; see TestStaffOnlyToolFilter below for that
        # surface's behavior.
        self.tools = asyncio.run(server._list_tools())
        self.tool_names = {t.name for t in self.tools}

    def test_tool_count(self):
        # 27→28 when find_duplicate_candidates was added — composite pre-POST
        # check (title + link or company) over existing primitives.
        # 28→31 when the scrape-profile-enhancer tools landed
        # (inspect_scrape_html, test_url_rewrite, find_selectors_for_text).
        # 31→33 when the ActivityPub publish/unpublish tools landed
        # (publish_job_post, unpublish_job_post — CC-60).
        # Asserts the REGISTERED total, not the filtered surface a
        # non-staff client sees (see TestStaffOnlyToolFilter for that).
        assert len(self.tools) == 33

    def test_has_all_expected_tools(self):
        expected = {
            "create_company", "find_company_by_name", "search_companies", "get_companies",
            "create_job_post_with_company_check", "find_job_post_by_link",
            "search_job_posts", "get_job_posts", "update_job_post",
            "publish_job_post", "unpublish_job_post",
            "get_duplicate_candidates", "find_duplicate_candidates",
            "create_job_application", "get_job_applications",
            "get_applications_for_job_post", "update_job_application",
            "get_career_data", "get_current_user",
            "create_scrape", "get_scrapes", "update_scrape",
            "list_scrape_screenshots", "fetch_scrape_screenshot",
            "get_scrape_graph_trace", "get_scrape_statuses",
            "get_scrape_profile", "update_scrape_profile",
            "score_job_post", "get_scores",
            # scrape-profile-enhancer support (BS4 trim + selector test
            # + url_rewrites dry-run, all backed by lib/scrape_inspector)
            "inspect_scrape_html", "test_url_rewrite", "find_selectors_for_text",
        }
        assert self.tool_names == expected

    def test_no_browser_tools(self):
        browser_names = {n for n in self.tool_names if "browser" in n or "scrape_page" in n or "navigate" in n}
        assert not browser_names

    def test_no_email_tools(self):
        email_names = {n for n in self.tool_names if "email" in n or "tag" in n or "notmuch" in n}
        assert not email_names


class TestStaffOnlyToolFilter:
    """The StaffOnlyToolFilter middleware hides three enhancer tools
    from non-staff `tools/list` responses.

    The api enforces authorization on every endpoint these tools call,
    so the filter is a UX / prompt-context concern — regular users'
    LLMs shouldn't burn tokens on tool definitions they can't usefully
    invoke. These tests pin the visibility contract; the api-side
    authorization is covered by api/job_hunting/tests.
    """

    @pytest.fixture(autouse=True)
    def load_server(self):
        from mcp_servers.public_server import server
        from mcp_servers.staff_tool_filter import STAFF_ONLY_TOOLS
        self.server = server
        self.staff_only = STAFF_ONLY_TOOLS

    def _list_via_middleware(self, *, is_staff: bool) -> set[str]:
        """Run the request-handler `list_tools()` with a stubbed
        access token. Returns the visible tool name set.
        """
        from unittest.mock import patch
        from fastmcp.server.auth import AccessToken
        access = AccessToken(
            token="jh_stub", client_id="stub",
            scopes=["read", "write"] + (["staff"] if is_staff else []),
            claims={"user_id": "stub", "is_staff": is_staff},
        )
        # Patch in BOTH the middleware module's import site (where the
        # filter calls `get_access_token`) AND any other read sites.
        with patch(
            "mcp_servers.staff_tool_filter.get_access_token",
            return_value=access,
        ):
            tools = asyncio.run(self.server.list_tools())
        return {t.name for t in tools}

    def test_non_staff_session_hides_enhancer_tools(self):
        visible = self._list_via_middleware(is_staff=False)
        assert not (visible & self.staff_only), (
            "non-staff client must not see the enhancer tools in tools/list"
        )

    def test_staff_session_sees_enhancer_tools(self):
        visible = self._list_via_middleware(is_staff=True)
        assert self.staff_only <= visible, (
            "staff client must see all three enhancer tools"
        )

    def test_filtered_count_drops_by_exactly_three(self):
        non_staff = self._list_via_middleware(is_staff=False)
        staff = self._list_via_middleware(is_staff=True)
        assert len(staff) - len(non_staff) == len(self.staff_only)

    def test_unauthenticated_session_treated_as_non_staff(self):
        """No AccessToken in scope → defaults to non-staff. The fallback
        is intentional: an unauthenticated probe shouldn't see the
        staff-only surface, even though api gates would catch any
        actual attempt to invoke."""
        # The base `server.list_tools()` (used by TestPublicServerTools
        # above through _list_tools) goes through middleware with no
        # active session — middleware should hide the enhancer tools.
        tools = asyncio.run(self.server.list_tools())
        visible = {t.name for t in tools}
        assert not (visible & self.staff_only)


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


class TestApiKeyTokenVerifierResilience:
    """The verifier must NEVER leak exceptions as uvicorn 500. Any
    crash (network failure, OOM mid-call, malformed upstream response)
    has to degrade to a clean 401 by returning None.

    See cc todo Inbox/Bug/mcp public_server returns 500 from uvicorn
    on initialize with valid bearer — this test pins the contract that
    incident exposed.
    """

    def _verifier(self):
        from mcp_servers.public_server import ApiKeyTokenVerifier
        return ApiKeyTokenVerifier(api_base_url="http://test-api")

    def test_rejects_non_jh_prefix_without_calling_upstream(self):
        """Cheap sanity check — short-circuit path returns None and
        never touches httpx."""
        result = asyncio.run(self._verifier().verify_token("not_a_jh_token"))
        assert result is None

    def test_upstream_network_failure_returns_none_not_raises(self):
        from unittest.mock import patch
        import httpx

        class _ExplodingClient:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return False
            async def get(self, *args, **kwargs):
                raise httpx.ConnectError("simulated upstream down")

        with patch("mcp_servers.public_server.httpx.AsyncClient", _ExplodingClient):
            result = asyncio.run(self._verifier().verify_token("jh_test_token"))
        assert result is None

    def test_malformed_upstream_json_returns_none_not_raises(self):
        from unittest.mock import patch

        class _BadJsonResponse:
            status_code = 200
            def json(self):
                raise ValueError("not json")

        class _BadJsonClient:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return False
            async def get(self, *args, **kwargs):
                return _BadJsonResponse()

        with patch("mcp_servers.public_server.httpx.AsyncClient", _BadJsonClient):
            result = asyncio.run(self._verifier().verify_token("jh_test_token"))
        assert result is None


class TestApiKeyTokenVerifierCache:
    """A successful verification is cached for _VERIFY_CACHE_TTL_S so the
    verifier does NOT re-hit /me/ on every MCP request. Without the cache
    a client that connects (one verify) then immediately calls tools/list
    (another verify) turned any transient /me/ failure on the second call
    into a bogus 'invalid_token' 401 that dropped the session.

    Failures (non-200 or exception) are never cached, so a transient
    upstream error retries on the next call rather than sticking.
    """

    def _verifier(self):
        from mcp_servers.public_server import ApiKeyTokenVerifier
        return ApiKeyTokenVerifier(api_base_url="http://test-api")

    def _ok_client_factory(self, calls: list):
        """An httpx.AsyncClient stand-in whose GET returns a valid /me/
        200 and records each call into `calls`."""
        class _OkResponse:
            status_code = 200
            def json(self):
                return {"data": {"id": "42", "attributes": {"is_staff": False}}}

        class _OkClient:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return False
            async def get(self, *args, **kwargs):
                calls.append(1)
                return _OkResponse()

        return _OkClient

    def test_second_verify_within_ttl_hits_me_only_once(self):
        from unittest.mock import patch
        calls: list = []
        verifier = self._verifier()
        with patch(
            "mcp_servers.public_server.httpx.AsyncClient",
            self._ok_client_factory(calls),
        ):
            first = asyncio.run(verifier.verify_token("jh_cached"))
            second = asyncio.run(verifier.verify_token("jh_cached"))
        assert first is not None
        # Same cached object returned, /me/ called exactly once.
        assert second is first
        assert len(calls) == 1

    def test_non_200_is_not_cached(self):
        from unittest.mock import patch
        calls: list = []

        class _Resp401:
            status_code = 401
            def json(self):
                return {}

        class _Client:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return False
            async def get(self, *args, **kwargs):
                calls.append(1)
                return _Resp401()

        verifier = self._verifier()
        with patch("mcp_servers.public_server.httpx.AsyncClient", _Client):
            r1 = asyncio.run(verifier.verify_token("jh_bad"))
            r2 = asyncio.run(verifier.verify_token("jh_bad"))
        assert r1 is None
        assert r2 is None
        # Failure must NOT be cached → both calls re-hit /me/.
        assert len(calls) == 2

    def test_exception_is_not_cached(self):
        from unittest.mock import patch
        import httpx
        calls: list = []

        class _ExplodingClient:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return False
            async def get(self, *args, **kwargs):
                calls.append(1)
                raise httpx.ConnectError("simulated upstream down")

        verifier = self._verifier()
        with patch(
            "mcp_servers.public_server.httpx.AsyncClient", _ExplodingClient
        ):
            r1 = asyncio.run(verifier.verify_token("jh_flap"))
            r2 = asyncio.run(verifier.verify_token("jh_flap"))
        assert r1 is None
        assert r2 is None
        assert len(calls) == 2

    def test_expired_cache_entry_revalidates(self):
        from unittest.mock import patch
        calls: list = []
        clock = {"t": 1000.0}

        def _fake_monotonic():
            return clock["t"]

        verifier = self._verifier()
        with patch(
            "mcp_servers.public_server.httpx.AsyncClient",
            self._ok_client_factory(calls),
        ), patch(
            "mcp_servers.public_server.time.monotonic", _fake_monotonic
        ):
            asyncio.run(verifier.verify_token("jh_exp"))
            # Advance past the 300s TTL so the entry is stale.
            clock["t"] += 301
            asyncio.run(verifier.verify_token("jh_exp"))
        # Expired entry → second verify re-hits /me/.
        assert len(calls) == 2


# Every staff/CRUD tool param that addresses a NanoID-keyed resource.
# CC-77 swapped these models' integer PKs to opaque 10-char NanoID strings
# (Company, JobPost, JobApplication, Scrape, ScrapeProfile, Score). The MCP
# client validates outbound args against each tool's JSON schema, so an
# `int`-typed hint here makes the client strip a NanoID string like
# "JHTQQNggMp" before the call leaves the client (args collapse to empty) —
# the CC-87 regression. These params MUST resolve to JSON-schema "string".
_NANOID_ID_PARAMS = {
    "get_companies": ["id"],
    "get_duplicate_candidates": ["job_post_id"],
    "search_job_posts": ["company_id"],
    "get_job_posts": ["id"],
    "update_job_post": ["job_post_id", "company_id"],
    "publish_job_post": ["job_post_id"],
    "unpublish_job_post": ["job_post_id"],
    "create_job_application": ["job_post_id"],
    "get_job_applications": ["id"],
    "get_applications_for_job_post": ["job_post_id"],
    "update_job_application": ["application_id", "company_id"],
    "create_scrape": ["job_post_id", "company_id"],
    "get_scrapes": ["id"],
    "update_scrape": ["scrape_id"],
    "list_scrape_screenshots": ["scrape_id"],
    "get_scrape_graph_trace": ["scrape_id"],
    "get_scrape_statuses": ["scrape_id"],
    "fetch_scrape_screenshot": ["scrape_id"],
    "update_scrape_profile": ["profile_id"],
    "score_job_post": ["job_post_id"],
    "get_scores": ["id", "job_post_id"],
    "inspect_scrape_html": ["scrape_id"],
    "find_selectors_for_text": ["scrape_id"],
}

# Params that legitimately stayed `int` — pagination / value fields keyed to
# nothing NanoID. Pinned so a future swap can't over-broaden int → str.
_INT_PARAMS = {
    "get_scrapes": ["page", "per_page"],
    "get_job_posts": ["page", "per_page"],
    "inspect_scrape_html": ["max_chars", "max_matches"],
    "create_job_post_with_company_check": ["salary_min", "salary_max"],
    "find_selectors_for_text": ["max_results"],
}


def _schema_types(prop: dict) -> set:
    """Collect every JSON-schema 'type' declared on a property, walking
    anyOf/oneOf so Optional[...] (type | null) unions are handled."""
    types = set()
    if isinstance(prop.get("type"), str):
        types.add(prop["type"])
    for branch in (prop.get("anyOf") or []) + (prop.get("oneOf") or []):
        if isinstance(branch, dict) and isinstance(branch.get("type"), str):
            types.add(branch["type"])
    return types


class TestNanoIdParamSchema:
    """CC-87 regression: id params keyed to NanoID resources must surface as
    JSON-schema `string`, never `integer`. If they revert to `integer`, the
    MCP client strips the NanoID before the request is sent and every
    inspect-or-mutate-by-id op against prod breaks (the scrape-profile
    enhancer / sharpen / manual recon)."""

    @pytest.fixture(autouse=True)
    def load_tools(self):
        from mcp_servers.public_server import server
        # `_list_tools()` bypasses the staff filter so the enhancer tools
        # (inspect_scrape_html / find_selectors_for_text) are present.
        tools = asyncio.run(server._list_tools())
        self.props = {t.name: t.parameters["properties"] for t in tools}

    def test_nanoid_id_params_are_string_typed(self):
        offenders = []
        for tool_name, params in _NANOID_ID_PARAMS.items():
            for param in params:
                types = _schema_types(self.props[tool_name][param])
                if "string" not in types or "integer" in types:
                    offenders.append((tool_name, param, sorted(types)))
        assert not offenders, (
            "NanoID id params must be JSON-schema string, not integer: "
            f"{offenders}"
        )

    def test_pagination_and_value_params_stay_integer(self):
        # Guards against an over-broad str swap that would break paging /
        # salary / cap inputs the LLM passes as real integers.
        for tool_name, params in _INT_PARAMS.items():
            for param in params:
                types = _schema_types(self.props[tool_name][param])
                assert "integer" in types, (
                    f"{tool_name}.{param} should stay integer, got {sorted(types)}"
                )
                assert "string" not in types, (
                    f"{tool_name}.{param} should not be string, got {sorted(types)}"
                )


class TestNanoIdRoundTrip:
    """CC-87: a NanoID-shaped id reaches the api client unchanged — it is not
    stripped or coerced en route. Mocks the api client (mirrors
    test_publish_job_post_tools / test_api_tools _fake_api). The three tools
    the ticket calls out — update_scrape_profile (write), inspect_scrape_html
    + find_selectors_for_text (enhancer reads) — must carry the NanoID into
    the outbound URL / body."""

    NANOID = "JHTQQNggMp"

    def test_update_scrape_profile_forwards_nanoid_in_url_and_body(self):
        from unittest.mock import AsyncMock, MagicMock, patch
        import mcp_servers.public_server as ps

        fake = MagicMock()
        fake.patch_data = AsyncMock(return_value=(
            {"data": {"type": "scrape-profile", "id": self.NANOID,
                      "attributes": {}}},
            None, 200,
        ))
        with patch.object(ps, "_api", return_value=fake):
            asyncio.run(ps.update_scrape_profile(
                profile_id=self.NANOID,
                preferred_tier="anthropic:claude-haiku",
            ))
        path, body = fake.patch_data.await_args.args
        assert self.NANOID in path, path
        assert body["data"]["id"] == self.NANOID

    def test_inspect_scrape_html_forwards_nanoid_in_url(self):
        from unittest.mock import AsyncMock, MagicMock, patch
        import mcp_servers.public_server as ps

        fake = MagicMock()
        fake.get_data = AsyncMock(return_value=(
            {"data": {"attributes": {
                "html": "<html><body><h1>About the job</h1></body></html>",
                "status": "completed",
            }}},
            None, 200,
        ))
        with patch.object(ps, "_api", return_value=fake):
            out = asyncio.run(ps.inspect_scrape_html(scrape_id=self.NANOID))
        path = fake.get_data.await_args.args[0]
        assert self.NANOID in path, path
        # The NanoID was accepted, not rejected as a bad id.
        assert "error" not in (yaml.safe_load(out) or {})

    def test_find_selectors_for_text_forwards_nanoid_in_url(self):
        from unittest.mock import AsyncMock, MagicMock, patch
        import mcp_servers.public_server as ps

        fake = MagicMock()
        fake.get_data = AsyncMock(return_value=(
            {"data": {"attributes": {
                "html": (
                    "<html><body>"
                    "<h2 data-testid='jd'>About the job</h2>"
                    "</body></html>"
                ),
                "status": "completed",
            }}},
            None, 200,
        ))
        with patch.object(ps, "_api", return_value=fake):
            asyncio.run(ps.find_selectors_for_text(
                scrape_id=self.NANOID, text="About the job",
            ))
        path = fake.get_data.await_args.args[0]
        assert self.NANOID in path, path
