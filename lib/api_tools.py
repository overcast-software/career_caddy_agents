"""
Shared Career Caddy API tool implementations.

Both career_caddy_server.py (local, CC_API_TOKEN) and public_server.py
(public, per-client auth) import from here. The ApiClient is instantiated
with a base_url and a token — the caller decides where the token comes from.
"""

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.parse import urljoin

import httpx
import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class APIResponse(BaseModel):
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None
    status_code: Optional[int] = None


class CompanyData(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    website: Optional[str] = None
    industry: Optional[str] = Field(None, max_length=100)
    size: Optional[str] = None
    location: Optional[str] = Field(None, max_length=100)


class JobPostCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    company_id: int = Field(..., gt=0)
    location: Optional[str] = Field(None, max_length=100)
    salary_min: Optional[int] = Field(None, ge=0)
    salary_max: Optional[int] = Field(None, ge=0)
    employment_type: Optional[str] = None
    remote_ok: bool = False
    link: Optional[str] = None
    posted_date: Optional[str] = None


_APPLICATION_SORT_FIELDS = Literal[
    "id", "applied_at", "status", "job_post_id", "company_id", "notes"
]


# ---------------------------------------------------------------------------
# ApiClient — thin async HTTP wrapper
# ---------------------------------------------------------------------------


_TYPE_TO_ROUTE = {
    "job-post": "job-posts",
    "job-application": "job-applications",
    "company": "companies",
    "score": "scores",
    "resume": "resumes",
    "cover-letter": "cover-letters",
    "question": "questions",
    "answer": "answers",
    "summary": "summaries",
    "scrape": "scrapes",
}


def _inject_frontend_urls(data: dict) -> dict:
    """Add _frontend_url and strip API links so the LLM uses frontend paths."""
    def _tag(resource):
        if isinstance(resource, dict) and "type" in resource and "id" in resource:
            route = _TYPE_TO_ROUTE.get(resource["type"])
            if route:
                resource["_frontend_url"] = f"/{route}/{resource['id']}"
            # Remove API links/relationships to prevent the model from
            # constructing URLs like https://api/v1/... instead of /companies/78
            resource.pop("links", None)
            resource.pop("relationships", None)
        return resource

    if isinstance(data.get("data"), list):
        for item in data["data"]:
            _tag(item)
    elif isinstance(data.get("data"), dict):
        _tag(data["data"])
    return data


# ---------------------------------------------------------------------------
# Slim-response helpers (PR #1: present but unwired)
# ---------------------------------------------------------------------------
#
# The MCP layer used to return JSON:API verbatim, double-wrapped in
# {"success", "data", "status_code"} and pretty-printed JSON. ~95% of a
# typical response was relationship arrays the agent never reads.
#
# These helpers give each tool a small kit for shaping its own response:
#
#   _respond(payload)              → YAML serializer; no outer envelope.
#   _relationships_to_counts(rec)  → array → int per relationship.
#   _slim_record(rec, ...)         → keep only listed attrs / relationship mode.
#
# Per-tool shape decisions live in TOOL_SHAPES below, NOT in the agent.
# Tools read their own row when composing the response.


# Tool authors decide what each tool returns. Kept as a single dict so the
# audit is one PR-reviewable diff. Agents never see this.
TOOL_SHAPES: dict[str, dict[str, Any]] = {
    # --- Read: single-resource ---
    "get_current_user": {
        "kind": "single",
        "attrs": [
            "username", "email", "first_name", "last_name",
            "is_staff", "is_active", "is_guest", "auto_score",
            "linkedin", "github", "address", "links", "onboarding",
        ],
        "relationships": "counts",
        "notes": "Drop scores/job-applications id arrays; keep links blob.",
    },
    "find_company_by_name": {
        "kind": "single",
        "attrs": None,
        "relationships": "counts",
    },
    "find_job_post_by_link": {
        "kind": "single",
        "attrs": None,
        "relationships": "counts",
        "notes": "Keep duplicate_of_id (it's an attr, not a relationship).",
    },
    "get_scrape_profile": {
        "kind": "list_or_single",
        "attrs": None,
        "relationships": "counts",
        "notes": "Config-shaped, naturally small. No attr trim.",
    },
    "get_scrape_graph_trace": {
        "kind": "passthrough",
        "notes": "Already slim. Just YAML-serialize and drop wrapper.",
    },

    # --- Read: list (table) ---
    "get_companies": {
        "kind": "list_or_single",
        "list_attrs": ["name"],
        "single_attrs": None,
        "relationships": "counts",
    },
    "search_companies": {
        "kind": "list",
        "list_attrs": ["name"],
        "relationships": "counts",
    },
    "get_job_posts": {
        "kind": "list_or_single",
        # Company shows up via relationships=counts (belongsTo → id).
        "list_attrs": ["title", "posting_status", "created_at", "duplicate_of_id"],
        "single_attrs": None,
        "relationships": "counts",
    },
    "search_job_posts": {
        "kind": "list",
        "list_attrs": ["title", "posting_status", "created_at", "duplicate_of_id"],
        "relationships": "counts",
    },
    "get_job_applications": {
        "kind": "list_or_single",
        "list_attrs": ["status", "reason_code", "applied_at"],
        "single_attrs": None,
        "relationships": "counts",
    },
    "get_applications_for_job_post": {
        "kind": "list",
        "list_attrs": ["status", "reason_code", "applied_at"],
        "relationships": "counts",
    },
    "get_scrapes": {
        "kind": "list_or_single",
        "list_attrs": ["url", "status", "scraped_at"],
        "single_attrs": None,
        "relationships": "counts",
    },
    "get_scores": {
        "kind": "list_or_single",
        # `top_score` etc. are belongsTo on score; relationships=counts
        # surfaces job_post id.
        "list_attrs": ["score", "status", "created_at"],
        "single_attrs": None,
        "relationships": "counts",
    },
    "get_resumes": {
        "kind": "list_or_single",
        "list_attrs": ["name", "title", "created_at", "favorite"],
        "single_attrs": None,
        "relationships": "counts",
    },
    "get_questions": {
        "kind": "list_or_single",
        "list_attrs": ["content", "created_at"],
        "single_attrs": None,
        "relationships": "counts",
    },
    "get_answers": {
        "kind": "list_or_single",
        "list_attrs": ["content", "created_at", "favorite"],
        "single_attrs": None,
        "relationships": "counts",
    },
    "list_scrape_screenshots": {
        "kind": "passthrough",
        "notes": "Already slim filename list. Just YAML + drop wrapper.",
    },
    "get_scrape_statuses": {
        "kind": "passthrough",
        "notes": "Already slim. Just YAML + drop wrapper.",
    },

    # --- Read: aggregate ---
    "get_career_data": {
        "kind": "aggregate",
        "section_attrs": {
            "resume": ["name", "created_at"],
            "skill": ["name", "category", "level"],
            "experience": ["company", "title", "started_at", "ended_at"],
            "education": ["institution", "degree", "field", "ended_at"],
            "certification": ["name", "issuer", "issued_at"],
            "cover_letter": ["title", "created_at"],
        },
        "notes": "Trim every nested record; keep nesting structure.",
    },

    # --- Write: create ---
    "create_company": {
        "kind": "single",
        "attrs": None,
        "relationships": "counts",
    },
    "create_job_post_with_company_check": {
        "kind": "single",
        "attrs": None,
        "relationships": "counts",
    },
    "create_job_application": {
        "kind": "single",
        "attrs": None,
        "relationships": "counts",
    },
    "create_scrape": {
        "kind": "single",
        "attrs": None,
        "relationships": "counts",
    },

    # --- Write: update ---
    "update_job_post": {
        "kind": "single",
        "attrs": None,
        "relationships": "counts",
    },
    "update_job_application": {
        "kind": "single",
        "attrs": None,
        "relationships": "counts",
    },
    "update_scrape": {
        "kind": "single",
        "attrs": None,
        "relationships": "counts",
    },
    "update_scrape_profile": {
        "kind": "single",
        "attrs": None,
        "relationships": "counts",
    },

    # --- Action ---
    "score_job_post": {
        "kind": "passthrough",
        "notes": "202 + pending status. Already small.",
    },
    "fetch_scrape_screenshot": {
        "kind": "binary_meta",
        "notes": "Don't dump the bytes. Return filename + size only.",
    },
}


def _respond(payload: Any, *, error: Optional[str] = None,
             status_code: Optional[int] = None) -> str:
    """Serialize a tool response as YAML.

    No outer {success, data, status_code} envelope; success is "no top-level
    error key." Errors include the HTTP status only when non-2xx. Insertion
    order is preserved (sort_keys=False) so the LLM sees identifying fields
    first.
    """
    if error is not None:
        out: dict[str, Any] = {"error": error}
        if status_code is not None and status_code >= 400:
            out["status_code"] = status_code
        return yaml.safe_dump(out, sort_keys=False, default_flow_style=False)
    return yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)


def _relationships_to_counts(record: dict) -> dict:
    """Replace each `relationships.<rel>.data` array with its integer count.

    Mutates and returns. Keeps relationship keys so the agent knows which
    relationships exist; just strips the `[{type:..., id:...}, ...]` payload
    that's never the cheapest way to fetch related records anyway.
    """
    rels = record.get("relationships")
    if not isinstance(rels, dict):
        return record
    for name, blob in list(rels.items()):
        if not isinstance(blob, dict):
            continue
        data = blob.get("data")
        if isinstance(data, list):
            rels[name] = len(data)
        elif isinstance(data, dict):
            # singular relationship (belongsTo) — replace with the id only,
            # the agent can refetch via the typed get_* tool if it needs more.
            rels[name] = data.get("id")
        else:
            rels[name] = None
    return record


async def _shaped_get(
    api: "ApiClient",
    path: str,
    *,
    shape: dict,
    params: Optional[dict] = None,
    is_single: Optional[bool] = None,
) -> str:
    """Fetch + slim + serialize. Tool one-liner glue."""
    payload, error, status = await api.get_data(path, params=params)
    if error is not None:
        return _respond(None, error=error, status_code=status)
    _slim_payload(payload, shape=shape, is_single=is_single)
    return _respond(payload)


async def _shaped_post(
    api: "ApiClient",
    path: str,
    body: dict,
    *,
    shape: dict,
) -> str:
    """POST + slim single-record response + serialize."""
    payload, error, status = await api.post_data(path, body)
    if error is not None:
        return _respond(None, error=error, status_code=status)
    _slim_payload(payload, shape=shape, is_single=True)
    return _respond(payload)


async def _shaped_patch(
    api: "ApiClient",
    path: str,
    body: dict,
    *,
    shape: dict,
) -> str:
    """PATCH + slim single-record response + serialize."""
    payload, error, status = await api.patch_data(path, body)
    if error is not None:
        return _respond(None, error=error, status_code=status)
    _slim_payload(payload, shape=shape, is_single=True)
    return _respond(payload)


def _slim_payload(payload: Optional[dict], *, shape: dict,
                  is_single: Optional[bool] = None) -> Optional[dict]:
    """Apply a TOOL_SHAPES row to a JSON:API response payload.

    - kind="passthrough": return payload unchanged.
    - payload.data is a list: treat each record with shape["list_attrs"].
    - payload.data is a dict: treat as single record with shape["single_attrs"]
      (falling back to shape["attrs"] if not set).

    Mutates and returns the payload so callers can chain into _respond.
    Tools that branch list vs single by url (.../id/ vs .../) can pass
    `is_single` explicitly; the default reads off payload.data's type.
    """
    if not isinstance(payload, dict):
        return payload
    if shape.get("kind") == "passthrough":
        return payload

    rels_mode = shape.get("relationships", "counts")
    data = payload.get("data")

    if is_single is None:
        is_single = isinstance(data, dict)

    if isinstance(data, list):
        list_attrs = shape.get("list_attrs", shape.get("attrs"))
        for record in data:
            if isinstance(record, dict):
                _slim_record(record, attrs=list_attrs, relationships=rels_mode)
    elif isinstance(data, dict):
        single_attrs = shape.get("single_attrs", shape.get("attrs"))
        _slim_record(data, attrs=single_attrs, relationships=rels_mode)

    # JSON:API compound docs sometimes inflate response with `included`.
    # We don't surface relationship arrays anyway, so drop included to
    # avoid leaking record bodies the agent never asked for.
    payload.pop("included", None)
    return payload


def _slim_record(record: dict, *,
                 attrs: Optional[list[str]] = None,
                 relationships: str = "counts") -> dict:
    """Trim a JSON:API resource record to its slim shape.

    Args:
        record: a JSON:API resource dict ({type, id, attributes, ...}).
        attrs: if provided, keep only these attribute keys; None keeps all.
        relationships: "counts" | "omit" | "inline"
            counts → arrays become ints, belongsTo become ids.
            omit   → drop the relationships key entirely.
            inline → leave as-is (escape hatch; default for un-audited tools).

    Mutates and returns the record so callers can chain.
    """
    if attrs is not None and isinstance(record.get("attributes"), dict):
        record["attributes"] = {
            k: v for k, v in record["attributes"].items() if k in attrs
        }
    if relationships == "omit":
        record.pop("relationships", None)
    elif relationships == "counts":
        _relationships_to_counts(record)
    # inline: no-op
    return record


class ApiClient:
    """HTTP client that forwards a token to the Career Caddy API."""

    def __init__(self, base_url: str, token: str, timeout: int = 120):
        self.base_url = base_url
        self.timeout = timeout
        self._headers = {
            "Authorization": f"Bearer {token}",
            "X-Forwarded-Proto": "https",
        }

    def _parse(self, response: httpx.Response) -> tuple[Optional[dict], Optional[str], int]:
        """Parse an httpx Response into (payload, error, status_code).

        On 2xx: payload is the JSON body (post-_inject_frontend_urls), error
        is None.
        On non-2xx: payload is None, error is a short string.

        Used by both the agent-facing _ok (which serializes to YAML) and the
        internal *_data methods that need to inspect the response without
        round-tripping through a string format.
        """
        if response.status_code in (200, 201, 202):
            body = response.json()
            _inject_frontend_urls(body)
            return body, None, response.status_code
        text = response.text[:500] if len(response.text) > 500 else response.text
        return None, f"{response.status_code} - {text}", response.status_code

    def _ok(self, response: httpx.Response) -> str:
        """Agent-facing serializer. Returns a YAML string with no outer
        envelope; errors land as a top-level `error:` key."""
        payload, error, status = self._parse(response)
        if error is not None:
            return _respond(None, error=error, status_code=status)
        return _respond(payload)

    async def get(self, path: str, params: dict | None = None) -> str:
        async with httpx.AsyncClient(follow_redirects=True, timeout=self.timeout, trust_env=False) as client:
            resp = await client.get(
                urljoin(self.base_url, path),
                headers=self._headers,
                params=params,
            )
            return self._ok(resp)

    async def get_data(self, path: str, params: dict | None = None) -> tuple[Optional[dict], Optional[str], int]:
        """Internal-use GET that returns parsed (payload, error, status).

        Tools that need to inspect the response (e.g. find duplicate
        before creating, chain a child request) should use this instead
        of json.loads-ing the agent-facing YAML output.
        """
        async with httpx.AsyncClient(follow_redirects=True, timeout=self.timeout, trust_env=False) as client:
            resp = await client.get(
                urljoin(self.base_url, path),
                headers=self._headers,
                params=params,
            )
            return self._parse(resp)

    async def get_text(self, path: str, params: dict | None = None) -> str:
        """GET an endpoint that returns a non-JSON body (e.g. text/markdown).

        Returns the raw body on 2xx; on error returns a YAML error string
        matching the agent-facing shape so callers can branch the same way.
        """
        async with httpx.AsyncClient(follow_redirects=True, timeout=self.timeout, trust_env=False) as client:
            resp = await client.get(
                urljoin(self.base_url, path),
                headers=self._headers,
                params=params,
            )
        if resp.status_code in (200, 201, 202):
            return resp.text
        text = resp.text[:500] if len(resp.text) > 500 else resp.text
        return _respond(
            None,
            error=f"{resp.status_code} - {text}",
            status_code=resp.status_code,
        )

    async def post(self, path: str, payload: dict) -> str:
        async with httpx.AsyncClient(follow_redirects=True, timeout=self.timeout, trust_env=False) as client:
            resp = await client.post(
                urljoin(self.base_url, path),
                headers=self._headers,
                json=payload,
            )
            return self._ok(resp)

    async def post_data(self, path: str, payload: dict) -> tuple[Optional[dict], Optional[str], int]:
        """Internal-use POST that returns parsed (payload, error, status)."""
        async with httpx.AsyncClient(follow_redirects=True, timeout=self.timeout, trust_env=False) as client:
            resp = await client.post(
                urljoin(self.base_url, path),
                headers=self._headers,
                json=payload,
            )
            return self._parse(resp)

    async def patch_data(self, path: str, payload: dict) -> tuple[Optional[dict], Optional[str], int]:
        """Internal-use PATCH that returns parsed (payload, error, status)."""
        async with httpx.AsyncClient(follow_redirects=True, timeout=self.timeout, trust_env=False) as client:
            resp = await client.patch(
                urljoin(self.base_url, path),
                headers=self._headers,
                json=payload,
            )
            return self._parse(resp)

    async def patch(self, path: str, payload: dict) -> str:
        async with httpx.AsyncClient(follow_redirects=True, timeout=self.timeout, trust_env=False) as client:
            resp = await client.patch(
                urljoin(self.base_url, path),
                headers=self._headers,
                json=payload,
            )
            return self._ok(resp)

    async def post_file(self, path: str, file_path: Path, field: str = "file") -> str:
        """POST a file as multipart/form-data."""
        async with httpx.AsyncClient(follow_redirects=True, timeout=self.timeout, trust_env=False) as client:
            with open(file_path, "rb") as f:
                resp = await client.post(
                    urljoin(self.base_url, path),
                    headers={
                        "Authorization": self._headers["Authorization"],
                        "X-Forwarded-Proto": "https",
                    },
                    files={field: (file_path.name, f, "image/png")},
                )
            return self._ok(resp)


# ---------------------------------------------------------------------------
# Tool implementations — all take an ApiClient as first argument
# ---------------------------------------------------------------------------


async def create_company(
    api: ApiClient,
    name: str,
    description: Optional[str] = None,
    website: Optional[str] = None,
    industry: Optional[str] = None,
    size: Optional[str] = None,
    location: Optional[str] = None,
) -> str:
    """Create a new company."""
    try:
        data = CompanyData(
            name=name,
            description=description,
            website=website,
            industry=industry,
            size=size,
            location=location,
        )
        body = {
            "data": {
                "type": "company",
                "attributes": data.model_dump(exclude_none=True),
            }
        }
        return await _shaped_post(
            api, "/api/v1/companies/", body, shape=TOOL_SHAPES["create_company"],
        )
    except ValueError as e:
        return _respond(None, error=f"Validation error: {e}")


async def find_company_by_name(api: ApiClient, company_name: str) -> str:
    """Find a company by name (case-insensitive search)."""
    payload, error, status = await api.get_data(
        "/api/v1/companies/", params={"filter[query]": company_name}
    )
    if error is not None:
        return _respond(None, error=error, status_code=status)
    companies = (payload or {}).get("data", []) or []
    if not companies:
        return _respond(
            None,
            error=f"No companies found matching '{company_name}'",
            status_code=404,
        )
    # Slim each company per audit (relationships → counts).
    shape = TOOL_SHAPES["find_company_by_name"]
    rels_mode = shape.get("relationships", "counts")
    for c in companies:
        if isinstance(c, dict):
            _slim_record(c, attrs=shape.get("attrs"), relationships=rels_mode)
    return _respond({"companies": companies, "count": len(companies)})


async def search_companies(
    api: ApiClient,
    query: Optional[str] = None,
    page_size: Optional[int] = None,
) -> str:
    """Search companies by name or display_name (case-insensitive OR match)."""
    params = {}
    if query is not None:
        params["filter[query]"] = query
    if page_size is not None:
        params["page[size]"] = page_size
    return await _shaped_get(
        api, "/api/v1/companies/", shape=TOOL_SHAPES["search_companies"], params=params,
    )


async def get_companies(api: ApiClient, id: Optional[int] = None) -> str:
    """Fetch companies. Pass id to retrieve a single company; omit for the full list."""
    shape = TOOL_SHAPES["get_companies"]
    if id is not None:
        return await _shaped_get(
            api, f"/api/v1/companies/{id}/", shape=shape, is_single=True,
        )
    return await _shaped_get(api, "/api/v1/companies/", shape=shape, is_single=False)


async def find_job_post_by_link(api: ApiClient, link: str) -> str:
    """Find a job post by its original posting URL."""
    return await _shaped_get(
        api,
        "/api/v1/job-posts/",
        shape=TOOL_SHAPES["find_job_post_by_link"],
        params={"filter[link]": link},
    )


async def create_job_post_with_company_check(
    api: ApiClient,
    title: str,
    company_name: Optional[str] = None,
    description: Optional[str] = None,
    location: Optional[str] = None,
    salary_min: Optional[int] = None,
    salary_max: Optional[int] = None,
    employment_type: Optional[str] = None,
    remote_ok: bool = False,
    url: Optional[str] = None,
    link: Optional[str] = None,
    posted_date: Optional[str] = None,
    company_description: Optional[str] = None,
    company_website: Optional[str] = None,
    company_industry: Optional[str] = None,
    company_size: Optional[str] = None,
    company_location: Optional[str] = None,
    source: str = "chat",
) -> str:
    """Create a job post, creating the company first if it doesn't exist.

    `source` tags provenance on the JobPost — defaults to 'chat' since this
    helper is primarily called from the career_caddy_agent tool. Backend
    uses the field for the sankey report's source-attribution view.
    """
    job_url = url or link
    if not company_name:
        return _respond(
            None, error="company_name is required to create a job post"
        )

    _PLACEHOLDER_NAMES = {"unknown", "n/a", "na", "none", "tbd", "not specified", ""}
    if company_name.strip().lower() in _PLACEHOLDER_NAMES:
        return _respond(
            None,
            error=(
                f"'{company_name}' is not an acceptable company name. "
                "Infer the company from: (1) the recruiter's company, "
                "(2) the email sender domain, (3) the job posting URL domain. "
                "If you cannot determine the company, ask the user."
            ),
        )

    try:
        # Check for duplicate by URL
        if job_url:
            existing_payload, _, _ = await api.get_data(
                "/api/v1/job-posts/", params={"filter[link]": job_url}
            )
            posts = (existing_payload or {}).get("data", []) or []
            if posts:
                existing_id = posts[0].get("id")
                return _respond({
                    "error": f"Duplicate: job post with this link already exists (id={existing_id})",
                    "status_code": 409,
                    "duplicate": True,
                    "existing_id": existing_id,
                })

        # Search for existing company
        company_search_payload, _, _ = await api.get_data(
            "/api/v1/companies/", params={"filter[query]": company_name}
        )
        company_id = None
        existing_companies = (company_search_payload or {}).get("data", []) or []
        if existing_companies:
            company_id = int(existing_companies[0].get("id"))

        # Create company if not found
        if company_id is None:
            company_data = CompanyData(
                name=company_name,
                description=company_description,
                website=company_website,
                industry=company_industry,
                size=company_size,
                location=company_location,
            )
            create_payload, create_err, _ = await api.post_data(
                "/api/v1/companies/",
                {"data": {"type": "company", "attributes": company_data.model_dump(exclude_none=True)}},
            )
            if create_err is not None:
                return _respond(None, error=f"Failed to create company: {create_err}")
            new_company = (create_payload or {}).get("data", {}) or {}
            company_id = int(new_company.get("id"))

        # Create the job post
        job_data = JobPostCreate(
            title=title,
            description=description,
            company_id=company_id,
            location=location,
            salary_min=salary_min,
            salary_max=salary_max,
            employment_type=employment_type,
            remote_ok=remote_ok,
            link=job_url,
            posted_date=posted_date,
        )
        attributes = job_data.model_dump(exclude={"company_id"}, exclude_none=True)
        attributes["source"] = source
        body = {
            "data": {
                "type": "job-post",
                "attributes": attributes,
                "relationships": {
                    "company": {"data": {"type": "company", "id": str(company_id)}}
                },
            }
        }
        return await _shaped_post(
            api, "/api/v1/job-posts/", body,
            shape=TOOL_SHAPES["create_job_post_with_company_check"],
        )

    except Exception as e:
        return _respond(None, error=f"Error creating job post with company check: {e}")


async def search_job_posts(
    api: ApiClient,
    query: Optional[str] = None,
    title: Optional[str] = None,
    company: Optional[str] = None,
    company_id: Optional[int] = None,
    sort: Optional[str] = None,
    page_size: Optional[int] = None,
) -> str:
    """Search job posts by keyword, title, company name, or company ID."""
    params = {}
    if query is not None:
        params["filter[query]"] = query
    if title is not None:
        params["filter[title]"] = title
    if company is not None:
        params["filter[company]"] = company
    if company_id is not None:
        params["filter[company_id]"] = company_id
    if sort is not None:
        params["sort"] = sort
    if page_size is not None:
        params["page[size]"] = page_size
    return await _shaped_get(
        api, "/api/v1/job-posts/", shape=TOOL_SHAPES["search_job_posts"], params=params,
    )


async def get_job_posts(
    api: ApiClient,
    id: Optional[int] = None,
    sort: Optional[str] = None,
    order: Optional[str] = None,
    page: Optional[int] = None,
    per_page: Optional[int] = None,
) -> str:
    """Fetch job posts. Pass id for a single post; omit for a paginated list."""
    shape = TOOL_SHAPES["get_job_posts"]
    if id is not None:
        return await _shaped_get(
            api, f"/api/v1/job-posts/{id}/", shape=shape, is_single=True,
        )
    params = {}
    if sort is not None:
        params["sort"] = sort
    if order is not None:
        params["order"] = order
    if page is not None:
        params["page"] = page
    if per_page is not None:
        params["per_page"] = per_page
    return await _shaped_get(
        api, "/api/v1/job-posts/", shape=shape, params=params, is_single=False,
    )


async def update_job_post(
    api: ApiClient,
    job_post_id: int,
    title: Optional[str] = None,
    description: Optional[str] = None,
    location: Optional[str] = None,
    salary_min: Optional[int] = None,
    salary_max: Optional[int] = None,
    employment_type: Optional[str] = None,
    remote_ok: Optional[bool] = None,
    link: Optional[str] = None,
    posted_date: Optional[str] = None,
    company_id: Optional[int] = None,
) -> str:
    """Update an existing job post's attributes or company relationship."""
    attributes = {}
    if title is not None:
        attributes["title"] = title
    if description is not None:
        attributes["description"] = description
    if location is not None:
        attributes["location"] = location
    if salary_min is not None:
        attributes["salary_min"] = salary_min
    if salary_max is not None:
        attributes["salary_max"] = salary_max
    if employment_type is not None:
        attributes["employment_type"] = employment_type
    if remote_ok is not None:
        attributes["remote_ok"] = remote_ok
    if link is not None:
        attributes["link"] = link
    if posted_date is not None:
        attributes["posted_date"] = posted_date

    if not attributes and company_id is None:
        return _respond(None, error="No fields provided to update")

    body: dict = {
        "data": {
            "type": "job-post",
            "id": str(job_post_id),
            "attributes": attributes,
        }
    }
    if company_id is not None:
        body["data"]["relationships"] = {
            "company": {"data": {"type": "company", "id": str(company_id)}}
        }
    return await _shaped_patch(
        api, f"/api/v1/job-posts/{job_post_id}/", body,
        shape=TOOL_SHAPES["update_job_post"],
    )


async def create_job_application(
    api: ApiClient,
    job_post_id: int,
    status: str = "applied",
    notes: Optional[str] = None,
    applied_at: Optional[str] = None,
) -> str:
    """Create a new job application linked to an existing job post."""
    if job_post_id <= 0:
        return _respond(
            None,
            error=f"Invalid job_post_id={job_post_id}. Look up the real ID first.",
        )

    attributes: dict = {"status": status}
    if notes is not None:
        attributes["notes"] = notes
    if applied_at is not None:
        attributes["applied_at"] = applied_at

    body = {
        "data": {
            "type": "job-application",
            "attributes": attributes,
            "relationships": {
                "job-post": {"data": {"type": "job-post", "id": str(job_post_id)}}
            },
        }
    }
    return await _shaped_post(
        api, "/api/v1/job-applications/", body,
        shape=TOOL_SHAPES["create_job_application"],
    )


async def get_job_applications(
    api: ApiClient,
    id: Optional[int] = None,
    sort: Optional[_APPLICATION_SORT_FIELDS] = None,
    order: Optional[Literal["asc", "desc"]] = None,
    page: Optional[int] = None,
    per_page: Optional[int] = None,
) -> str:
    """Fetch job applications. Pass id for a single application; omit for a list."""
    shape = TOOL_SHAPES["get_job_applications"]
    if id is not None:
        return await _shaped_get(
            api, f"/api/v1/job-applications/{id}/", shape=shape, is_single=True,
        )
    params: dict = {}
    if sort is not None:
        params["sort"] = sort
    if order is not None:
        params["order"] = order
    if page is not None:
        params["page"] = page
    if per_page is not None:
        params["per_page"] = per_page
    return await _shaped_get(
        api, "/api/v1/job-applications/", shape=shape, params=params, is_single=False,
    )


async def get_applications_for_job_post(api: ApiClient, job_post_id: int) -> str:
    """Fetch all job applications linked to a specific job post."""
    return await _shaped_get(
        api,
        f"/api/v1/job-posts/{job_post_id}/job-applications/",
        shape=TOOL_SHAPES["get_applications_for_job_post"],
    )


async def update_job_application(
    api: ApiClient,
    application_id: int,
    status: Optional[str] = None,
    notes: Optional[str] = None,
    applied_at: Optional[str] = None,
    company_id: Optional[int] = None,
) -> str:
    """Update a job application's status, notes, or company association."""
    attributes = {}
    if status is not None:
        attributes["status"] = status
    if notes is not None:
        attributes["notes"] = notes
    if applied_at is not None:
        attributes["applied_at"] = applied_at

    if not attributes and company_id is None:
        return _respond(None, error="No fields provided to update")

    body: dict = {
        "data": {
            "type": "job-application",
            "id": str(application_id),
            "attributes": attributes,
        }
    }
    if company_id is not None:
        body["data"]["relationships"] = {
            "company": {"data": {"type": "company", "id": str(company_id)}}
        }
    return await _shaped_patch(
        api, f"/api/v1/job-applications/{application_id}/", body,
        shape=TOOL_SHAPES["update_job_application"],
    )


async def get_career_data(api: ApiClient) -> str:
    """Fetch the user's personal career profile."""
    payload, error, status = await api.get_data("/api/v1/career-data/")
    if error is not None:
        return _respond(None, error=error, status_code=status)
    # career-data is a flat dict of nested arrays per resource type
    # ({"resume": [...], "skill": [...], ...}). Trim every nested record
    # to the per-section attr list in TOOL_SHAPES.
    section_attrs = TOOL_SHAPES["get_career_data"]["section_attrs"]
    if isinstance(payload, dict):
        for section, attrs in section_attrs.items():
            rows = payload.get(section)
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict):
                        # career-data records are flat (no JSON:API wrapper),
                        # so filter the dict in place.
                        for k in list(row.keys()):
                            if k not in attrs and k != "id":
                                row.pop(k, None)
    return _respond(payload)


async def get_resumes(
    api: ApiClient,
    id: Optional[int] = None,
    favorite: Optional[bool] = None,
    page: Optional[int] = None,
    per_page: Optional[int] = None,
) -> str:
    """Fetch resumes. Pass id for a single resume; omit for a paginated list.

    Use this tool to count or list ALL resumes — do not infer resume counts
    from career data, which may only include favorites.
    """
    shape = TOOL_SHAPES["get_resumes"]
    if id is not None:
        return await _shaped_get(
            api, f"/api/v1/resumes/{id}/", shape=shape, is_single=True,
        )
    params: dict = {}
    if favorite is not None:
        params["favorite"] = str(favorite).lower()
    if page is not None:
        params["page"] = page
    if per_page is not None:
        params["per_page"] = per_page
    return await _shaped_get(
        api, "/api/v1/resumes/", shape=shape, params=params, is_single=False,
    )


# ---------------------------------------------------------------------------
# New tools (scrapes + scores)
# ---------------------------------------------------------------------------


async def create_scrape(
    api: ApiClient,
    url: str,
    job_post_id: Optional[int] = None,
    company_id: Optional[int] = None,
    status: Optional[str] = None,
) -> str:
    """Create a scrape record. Omit status (or pass 'pending') to start scraping immediately; pass 'hold' to queue for later."""
    attributes = {"url": url}
    if status:
        attributes["status"] = status
    relationships = {}
    if job_post_id is not None:
        relationships["job-post"] = {
            "data": {"type": "job-post", "id": str(job_post_id)}
        }
    if company_id is not None:
        relationships["company"] = {
            "data": {"type": "company", "id": str(company_id)}
        }

    body: dict = {"data": {"type": "scrape", "attributes": attributes}}
    if relationships:
        body["data"]["relationships"] = relationships

    return await _shaped_post(
        api, "/api/v1/scrapes/", body, shape=TOOL_SHAPES["create_scrape"],
    )


async def get_scrapes(
    api: ApiClient,
    id: Optional[int] = None,
    sort: Optional[str] = None,
    page: Optional[int] = None,
    per_page: Optional[int] = None,
    status: Optional[str] = None,
    has_score: Optional[bool] = None,
) -> str:
    """Fetch scrape records. Pass id for a single scrape; omit for a paginated list. Use sort='-id' for most recent first, per_page=1 for just the latest. Filter by status with status='hold'. Pass has_score=False to scope to scrapes whose linked JobPost has no Score yet (drives the auto-score daemon)."""
    shape = TOOL_SHAPES["get_scrapes"]
    if id is not None:
        return await _shaped_get(
            api, f"/api/v1/scrapes/{id}/", shape=shape, is_single=True,
        )
    params: dict = {}
    if sort is not None:
        params["sort"] = sort
    if page is not None:
        params["page"] = page
    if per_page is not None:
        params["per_page"] = per_page
    if status is not None:
        params["filter[status]"] = status
    if has_score is not None:
        params["filter[has_score]"] = "true" if has_score else "false"
    return await _shaped_get(
        api, "/api/v1/scrapes/", shape=shape, params=params, is_single=False,
    )


async def update_scrape(
    api: ApiClient,
    scrape_id: int,
    status: Optional[str] = None,
    job_content: Optional[str] = None,
    url: Optional[str] = None,
    note: Optional[str] = None,
) -> str:
    """Update a scrape record's status, content, or URL."""
    attributes = {}
    if status is not None:
        attributes["status"] = status
    if job_content is not None:
        attributes["job_content"] = job_content
    if url is not None:
        attributes["url"] = url
    if note is not None:
        attributes["note"] = note

    if not attributes:
        return _respond(None, error="No fields provided to update")

    body = {
        "data": {
            "type": "scrape",
            "id": str(scrape_id),
            "attributes": attributes,
        }
    }
    return await _shaped_patch(
        api, f"/api/v1/scrapes/{scrape_id}/", body,
        shape=TOOL_SHAPES["update_scrape"],
    )


async def upload_screenshot(api: ApiClient, scrape_id: int, file_path: Path) -> str:
    """Upload a screenshot PNG to the API for a scrape."""
    return await api.post_file(
        f"/api/v1/scrapes/{scrape_id}/screenshots/", file_path
    )


async def list_screenshots(api: ApiClient, scrape_id: int) -> str:
    """List screenshot filenames for a scrape. Staff-only endpoint."""
    return await api.get(f"/api/v1/scrapes/{scrape_id}/screenshots/")


async def get_scrape_graph_trace(api: ApiClient, scrape_id: int) -> str:
    """Fetch the pydantic-graph node trace for a scrape. Owner-or-staff
    gated endpoint. Returns ordered transitions (graph_node + payload +
    note + timestamp) and meta.chain walking the source_scrape parents
    so a tracker URL and its canonical child render as one path."""
    return await api.get(f"/api/v1/scrapes/{scrape_id}/graph-trace/")


async def get_scrape_statuses(api: ApiClient, scrape_id: int) -> str:
    """Fetch the full ScrapeStatus history for a scrape — every row, not
    just rows with a graph_node set. Owner-or-staff gated; the rows
    can carry exception text and internal-only diagnostic detail in
    `note` and `graph_payload`."""
    return await api.get(f"/api/v1/scrapes/{scrape_id}/scrape-statuses/")


async def fetch_screenshot_bytes(api: ApiClient, scrape_id: int, filename: str) -> bytes:
    """Download a screenshot PNG's raw bytes. Staff-only endpoint.

    Returns raw PNG bytes on success; raises on HTTP error. Used by the MCP
    tool wrapper to base64-encode for transport to MCP clients.
    """
    async with httpx.AsyncClient(follow_redirects=True, timeout=api.timeout, trust_env=False) as client:
        resp = await client.get(
            urljoin(api.base_url, f"/api/v1/scrapes/{scrape_id}/screenshots/{filename}"),
            headers=api._headers,
        )
    resp.raise_for_status()
    return resp.content


async def get_scrape_profile(api: ApiClient, hostname: str) -> str:
    """Fetch the scrape profile for a hostname. Returns profile data or empty."""
    return await _shaped_get(
        api,
        "/api/v1/scrape-profiles/",
        shape=TOOL_SHAPES["get_scrape_profile"],
        params={"filter[hostname]": hostname},
    )


async def update_scrape_profile(api: ApiClient, profile_id: int, **attrs) -> str:
    """Update a scrape profile's editable fields (css_selectors, extraction_hints, etc.)."""
    json_attrs = {}
    for key, value in attrs.items():
        json_attrs[key.replace("_", "-")] = value
    body = {
        "data": {
            "type": "scrape-profile",
            "id": str(profile_id),
            "attributes": json_attrs,
        }
    }
    return await _shaped_patch(
        api, f"/api/v1/scrape-profiles/{profile_id}/", body,
        shape=TOOL_SHAPES["update_scrape_profile"],
    )


async def get_questions(
    api: ApiClient,
    id: Optional[int] = None,
    company_id: Optional[int] = None,
    job_post_id: Optional[int] = None,
    page: Optional[int] = None,
    per_page: Optional[int] = None,
) -> str:
    """Fetch interview questions. Pass id for a single question; omit for a paginated list. Filter by company_id or job_post_id."""
    shape = TOOL_SHAPES["get_questions"]
    if id is not None:
        return await _shaped_get(
            api, f"/api/v1/questions/{id}/", shape=shape, is_single=True,
        )
    params: dict = {}
    if company_id is not None:
        params["filter[company_id]"] = company_id
    if job_post_id is not None:
        params["filter[job_post_id]"] = job_post_id
    if page is not None:
        params["page"] = page
    if per_page is not None:
        params["per_page"] = per_page
    return await _shaped_get(
        api, "/api/v1/questions/", shape=shape, params=params, is_single=False,
    )


async def create_question(
    api: ApiClient,
    content: str,
    company_id: Optional[int] = None,
    job_post_id: Optional[int] = None,
    job_application_id: Optional[int] = None,
) -> str:
    """Create an interview question. Supply at least one of company_id, job_post_id, or job_application_id so the question is scoped — unscoped questions are hard to find later. Always check get_questions first to avoid duplicates."""
    relationships: dict = {}
    if company_id is not None:
        relationships["company"] = {
            "data": {"type": "company", "id": str(company_id)}
        }
    if job_post_id is not None:
        relationships["jobPost"] = {
            "data": {"type": "job-post", "id": str(job_post_id)}
        }
    if job_application_id is not None:
        relationships["jobApplication"] = {
            "data": {"type": "job-application", "id": str(job_application_id)}
        }
    payload: dict = {
        "data": {
            "type": "question",
            "attributes": {"content": content},
        }
    }
    if relationships:
        payload["data"]["relationships"] = relationships
    return await api.post("/api/v1/questions/", payload)


async def get_answers(
    api: ApiClient,
    id: Optional[int] = None,
    question_id: Optional[int] = None,
    favorite: Optional[bool] = None,
    page: Optional[int] = None,
    per_page: Optional[int] = None,
) -> str:
    """Fetch answers to interview questions. Pass id for a single answer; omit for a paginated list. Filter by question_id or favorite status."""
    shape = TOOL_SHAPES["get_answers"]
    if id is not None:
        return await _shaped_get(
            api, f"/api/v1/answers/{id}/", shape=shape, is_single=True,
        )
    params: dict = {}
    if question_id is not None:
        params["filter[question_id]"] = question_id
    if favorite is not None:
        params["favorite"] = str(favorite).lower()
    if page is not None:
        params["page"] = page
    if per_page is not None:
        params["per_page"] = per_page
    return await _shaped_get(
        api, "/api/v1/answers/", shape=shape, params=params, is_single=False,
    )


async def create_answer(
    api: ApiClient,
    question_id: int,
    content: str,
    ai_assist: bool = False,
    prompt: Optional[str] = None,
) -> str:
    """Create an answer for an interview question. Set ai_assist=true to let the backend AI generate the content (content can be empty in that case). Optionally pass a prompt to guide AI generation."""
    attributes: dict = {}
    if content:
        attributes["content"] = content
    if ai_assist:
        attributes["ai_assist"] = True
    if prompt:
        attributes["prompt"] = prompt

    payload = {
        "data": {
            "type": "answer",
            "attributes": attributes,
            "relationships": {
                "question": {"data": {"type": "question", "id": str(question_id)}}
            },
        }
    }
    return await api.post("/api/v1/answers/", payload)


async def update_answer(
    api: ApiClient,
    answer_id: int,
    content: Optional[str] = None,
    favorite: Optional[bool] = None,
) -> str:
    """Update an existing answer's content or favorite status."""
    attributes = {}
    if content is not None:
        attributes["content"] = content
    if favorite is not None:
        attributes["favorite"] = favorite
    payload = {
        "data": {
            "type": "answer",
            "id": str(answer_id),
            "attributes": attributes,
        }
    }
    return await api.patch(f"/api/v1/answers/{answer_id}/", payload)


async def score_job_post(api: ApiClient, job_post_id: int) -> str:
    """Score a job post against the user's career data."""
    payload = {
        "data": {
            "type": "score",
            "attributes": {},
            "relationships": {
                "job-post": {"data": {"type": "job-post", "id": str(job_post_id)}}
            },
        }
    }
    return await api.post("/api/v1/scores/", payload)


async def get_scores(
    api: ApiClient,
    id: Optional[int] = None,
    job_post_id: Optional[int] = None,
    page: Optional[int] = None,
    per_page: Optional[int] = None,
) -> str:
    """Fetch scores. Pass id for a single score, or filter by job_post_id."""
    shape = TOOL_SHAPES["get_scores"]
    if id is not None:
        return await _shaped_get(
            api, f"/api/v1/scores/{id}/", shape=shape, is_single=True,
        )
    params: dict = {}
    if job_post_id is not None:
        params["filter[job_post_id]"] = job_post_id
    if page is not None:
        params["page"] = page
    if per_page is not None:
        params["per_page"] = per_page
    return await _shaped_get(
        api, "/api/v1/scores/", shape=shape, params=params, is_single=False,
    )


# ---------------------------------------------------------------------------
# Composite tool: scrape_and_score
# ---------------------------------------------------------------------------


_SCRAPE_TERMINAL = {"completed", "failed"}


async def _raw_get_scrape(api: ApiClient, scrape_id: int) -> dict:
    """GET a scrape and return the raw JSON:API body (relationships intact)."""
    async with httpx.AsyncClient(follow_redirects=True, timeout=api.timeout) as client:
        resp = await client.get(
            urljoin(api.base_url, f"/api/v1/scrapes/{scrape_id}/"),
            headers=api._headers,
        )
        resp.raise_for_status()
        return resp.json()


def _composite_err(message: str, **data) -> str:
    payload = {"error": message}
    if data:
        payload.update(data)
    return _respond(payload)


def _composite_ok(message: str, **data) -> str:
    payload = {"message": message, **data}
    return _respond(payload)


async def scrape_and_score(
    api: ApiClient,
    url: str,
    resume_id: Optional[int] = None,
    poll_interval: float = 3.0,
    timeout: float = 60.0,
) -> str:
    """Scrape a URL into Career Caddy, wait for completion, then score the resulting
    job post. Returns a message with the frontend URL to view scores.

    Flow: POST scrape (pending) → poll until completed/failed → POST score →
    return /job-posts/:id/scores link. Falls back to status=hold if scraping
    is disabled (501); in that case the hold-poller must be running.
    """
    # 1. Create scrape (try pending first)
    create_payload, create_err, create_status = await api.post_data(
        "/api/v1/scrapes/", {"data": {"type": "scrape", "attributes": {"url": url}}}
    )
    hold_fallback = False
    if create_err is not None:
        if create_status == 501:
            hold_fallback = True
            create_payload, create_err, _ = await api.post_data(
                "/api/v1/scrapes/",
                {"data": {"type": "scrape", "attributes": {"url": url, "status": "hold"}}},
            )
            if create_err is not None:
                return _composite_err(f"Failed to create scrape (hold fallback): {create_err}")
        else:
            return _composite_err(f"Failed to create scrape: {create_err}")

    scrape_data = (create_payload or {}).get("data", {}) or {}
    scrape_id = scrape_data.get("id")
    if scrape_id is None:
        return _composite_err("Scrape created but no id returned", response=create_payload)
    scrape_id = int(scrape_id)

    # 2. Poll for terminal status
    deadline = time.monotonic() + timeout
    scrape_body: dict = {}
    last_status: Optional[str] = None
    while True:
        try:
            scrape_body = await _raw_get_scrape(api, scrape_id)
        except httpx.HTTPError as exc:
            return _composite_err(f"Error polling scrape {scrape_id}: {exc}", scrape_id=scrape_id)

        attrs = scrape_body.get("data", {}).get("attributes", {}) or {}
        last_status = attrs.get("status")
        if last_status in _SCRAPE_TERMINAL:
            break
        if time.monotonic() >= deadline:
            return _composite_err(
                f"Timed out after {timeout:.0f}s waiting for scrape {scrape_id}; "
                f"last status={last_status}.",
                scrape_id=scrape_id,
                last_status=last_status,
                hold_fallback=hold_fallback,
            )
        await asyncio.sleep(poll_interval)

    if last_status == "failed":
        return _composite_err(
            f"Scrape {scrape_id} failed.",
            scrape_id=scrape_id,
            last_status=last_status,
        )

    # 3. Resolve job_post_id from relationships (may trail status flip briefly)
    def _extract_job_post_id(body: dict) -> Optional[int]:
        rels = body.get("data", {}).get("relationships", {}) or {}
        jp = (rels.get("job-post") or {}).get("data")
        if jp and jp.get("id"):
            return int(jp["id"])
        return None

    job_post_id = _extract_job_post_id(scrape_body)
    link_deadline = time.monotonic() + 15.0
    while job_post_id is None and time.monotonic() < link_deadline:
        await asyncio.sleep(poll_interval)
        try:
            scrape_body = await _raw_get_scrape(api, scrape_id)
        except httpx.HTTPError:
            break
        job_post_id = _extract_job_post_id(scrape_body)

    if job_post_id is None:
        return _composite_err(
            f"Scrape {scrape_id} completed but no job-post was linked yet. "
            "Try the 'parse' action on the scrape to extract the job post.",
            scrape_id=scrape_id,
        )

    # 4. Create the score
    if resume_id is not None:
        score_post = {
            "data": {
                "type": "score",
                "attributes": {},
                "relationships": {
                    "job-post": {"data": {"type": "job-post", "id": str(job_post_id)}},
                    "resume": {"data": {"type": "resume", "id": str(resume_id)}},
                },
            }
        }
    else:
        score_post = {
            "data": {
                "type": "score",
                "attributes": {},
                "relationships": {
                    "job-post": {"data": {"type": "job-post", "id": str(job_post_id)}}
                },
            }
        }
    score_payload, score_err, _ = await api.post_data("/api/v1/scores/", score_post)
    if score_err is not None:
        return _composite_err(
            f"Scrape completed but score creation failed: {score_err}",
            scrape_id=scrape_id,
            job_post_id=job_post_id,
        )

    score_data = (score_payload or {}).get("data", {}) or {}
    score_id = int(score_data["id"]) if score_data.get("id") else None
    score_status = (score_data.get("attributes") or {}).get("status")

    # 5. Build frontend URL + return
    frontend_base = os.environ.get("CC_FRONTEND_URL", "http://localhost:4200").rstrip("/")
    scores_url = f"{frontend_base}/job-posts/{job_post_id}/scores"

    message_parts = [
        f"Scrape {scrape_id} completed; scored job post {job_post_id}.",
        f"Score is {score_status or 'pending'} (will update as the backend finishes).",
        f"Open {scores_url} to view results.",
    ]
    if hold_fallback:
        message_parts.insert(
            0,
            "Note: scraping runs via hold-queue (the hold-poller must be running).",
        )

    return _composite_ok(
        " ".join(message_parts),
        scrape_id=scrape_id,
        job_post_id=job_post_id,
        score_id=score_id,
        score_status=score_status,
        scores_url=scores_url,
        hold_fallback=hold_fallback,
    )


# ---------------------------------------------------------------------------
# Agent Wizard tools — resume + cover-letter show/edit, resume import,
# and onboarding state writes.
# ---------------------------------------------------------------------------


async def show_resume(api: ApiClient, resume_id: int) -> str:
    """Return a markdown-rendered view of a resume (token-efficient vs. JSON).

    Use this after an import to narrate the extracted contents back to the user
    for review. The response body is plain text/markdown.
    """
    return await api.get_text(f"/api/v1/resumes/{resume_id}/markdown/")


async def edit_resume(
    api: ApiClient,
    resume_id: int,
    title: Optional[str] = None,
    name: Optional[str] = None,
    notes: Optional[str] = None,
    favorite: Optional[bool] = None,
) -> str:
    """Update a resume's TOP-LEVEL fields only (title, name, notes, favorite).

    This tool does NOT touch linked records: experiences, educations,
    certifications, projects, descriptions, skills. If the user asks to
    change a job title under a specific experience (e.g. "change my
    title at Robert Half International"), do NOT call this tool — that
    would incorrectly overwrite the resume's top-level title. Instead,
    navigate the user to the resume page (/resumes/{id}) and have them
    edit the experience directly. Chat is deliberately not a surface for
    editing deep resume structure; that lives in the form UI.

    When a request is ambiguous ("change my title" could mean the resume's
    label OR a job title inside an experience), ASK FOR CLARIFICATION
    before calling this tool."""
    attributes: dict = {}
    if title is not None:
        attributes["title"] = title
    if name is not None:
        attributes["name"] = name
    if notes is not None:
        attributes["notes"] = notes
    if favorite is not None:
        attributes["favorite"] = favorite
    if not attributes:
        return _respond(
            None, error="edit_resume requires at least one field to update"
        )
    payload = {
        "data": {
            "type": "resume",
            "id": str(resume_id),
            "attributes": attributes,
        }
    }
    return await api.patch(f"/api/v1/resumes/{resume_id}/", payload)


async def show_cover_letter(api: ApiClient, cover_letter_id: int) -> str:
    """Return a markdown-rendered view of a cover letter."""
    return await api.get_text(
        f"/api/v1/cover-letters/{cover_letter_id}/markdown/"
    )


async def edit_cover_letter(
    api: ApiClient,
    cover_letter_id: int,
    content: Optional[str] = None,
    favorite: Optional[bool] = None,
    status: Optional[str] = None,
) -> str:
    """Update a cover letter's content, favorite, or status."""
    attributes: dict = {}
    if content is not None:
        attributes["content"] = content
    if favorite is not None:
        attributes["favorite"] = favorite
    if status is not None:
        attributes["status"] = status
    if not attributes:
        return _respond(
            None, error="edit_cover_letter requires at least one field to update"
        )
    payload = {
        "data": {
            "type": "cover-letter",
            "id": str(cover_letter_id),
            "attributes": attributes,
        }
    }
    return await api.patch(
        f"/api/v1/cover-letters/{cover_letter_id}/", payload
    )


async def import_resume_from_url(
    api: ApiClient, url: str, resume_name: Optional[str] = None
) -> str:
    """Download a resume (DOCX or PDF) from a URL and hand it off to the ingest pipeline.

    Returns immediately after the 202 — callers should poll with get_resumes(id=...)
    until `status == "completed"`. For browser-driven uploads the user goes through
    the normal UI; this tool is for programmatic handoffs and URL-based flows.
    """
    try:
        async with httpx.AsyncClient(
            follow_redirects=True, timeout=api.timeout
        ) as client:
            resp = await client.get(url)
        if resp.status_code != 200:
            return _respond(
                None,
                error=f"Download failed: {resp.status_code}",
                status_code=resp.status_code,
            )
        filename = resume_name or url.rsplit("/", 1)[-1] or "resume.docx"
        async with httpx.AsyncClient(
            follow_redirects=True, timeout=api.timeout
        ) as client:
            upload_resp = await client.post(
                urljoin(api.base_url, "/api/v1/resumes/ingest/"),
                headers={
                    "Authorization": api._headers["Authorization"],
                    "X-Forwarded-Proto": "https",
                },
                files={
                    "file": (
                        filename,
                        resp.content,
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )
                },
            )
        return api._ok(upload_resp)
    except Exception as e:
        return _respond(None, error=f"import_resume_from_url: {e}")


async def reconcile_onboarding(api: ApiClient) -> str:
    """Recompute the authenticated user's onboarding state from their real data.

    Use this BEFORE reporting onboarding progress to the user. The stored
    onboarding blob defaults to all-false when a user predates the feature,
    so a naive read says "nothing is set up" even for long-time users.
    Reconcile first, then answer questions like "how is my onboarding?" or
    "what should I do next?" from the returned blob.

    Preserves the subjective fields the data cannot answer for
    (=wizard_enabled=, =resume_reviewed=) and overwrites the rest based on
    actual resume / job post / score / cover letter / profile-basics state.
    """
    return await api.post("/api/v1/onboarding/reconcile/", {})


async def edit_profile_onboarding(api: ApiClient, patch: dict) -> str:
    """Merge a partial dict into the authenticated user's profile.onboarding blob.

    Only keys in the canonical onboarding shape are accepted server-side; unknown
    keys are ignored. Typical usage:
        edit_profile_onboarding({"resume_reviewed": true})
        edit_profile_onboarding({"wizard_enabled": false})
    """
    if not isinstance(patch, dict) or not patch:
        return _respond(
            None, error="edit_profile_onboarding requires a non-empty dict"
        )
    me_payload, me_err, me_status = await api.get_data("/api/v1/me/")
    if me_err is not None:
        return _respond(None, error=me_err, status_code=me_status)
    user_id = (me_payload or {}).get("data", {}).get("id")
    if not user_id:
        return _respond(None, error="Could not resolve authenticated user id")
    payload = {
        "data": {
            "type": "user",
            "id": str(user_id),
            "attributes": {"onboarding": patch},
        }
    }
    return await api.patch(f"/api/v1/users/{user_id}/", payload)
