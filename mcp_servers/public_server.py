"""
Career Caddy Public MCP Server — authenticated proxy to the Career Caddy API.

Deployed at mcp.careercaddy.online. Exposes career-data tools only (no email,
no browser). Each client authenticates with their own jh_* API key, which is
forwarded to the Django API on every request.

    Connect at: https://mcp.careercaddy.online/mcp
    Auth:       Authorization: Bearer jh_xxxxx

Security invariants:
    - This file MUST NOT import email_server, browser_server, gateway, or lib/browser/*
    - No CC_API_TOKEN env var — all auth comes from clients
    - No secrets.yml or mail directory access
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Literal, Optional

import httpx
from fastmcp import FastMCP
from fastmcp.server.auth import AccessToken, TokenVerifier
from fastmcp.server.dependencies import get_access_token

# Add project root so lib imports work
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from lib.api_tools import ApiClient  # noqa: E402
from lib import api_tools  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_BASE_URL = os.environ.get("CC_API_BASE_URL", "http://localhost:8000")


# ---------------------------------------------------------------------------
# Auth: API key pass-through via TokenVerifier
# ---------------------------------------------------------------------------


class ApiKeyTokenVerifier(TokenVerifier):
    """Validates jh_* API keys by calling the Career Caddy API's /me/ endpoint.

    On success, stores the raw token and user profile in the AccessToken so
    tool functions can forward the token on every downstream API call.
    """

    def __init__(self, api_base_url: str, **kwargs):
        super().__init__(**kwargs)
        self.api_base_url = api_base_url

    async def verify_token(self, token: str) -> AccessToken | None:
        if not token.startswith("jh_"):
            return None

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.api_base_url}/api/v1/me/",
                headers={"Authorization": f"Bearer {token}"},
            )
            if resp.status_code != 200:
                return None

            user_data = resp.json().get("data", resp.json())
            user_id = user_data.get("id", "unknown")
            logger.info("Authenticated user_id=%s via API key", user_id)

            return AccessToken(
                token=token,
                client_id=str(user_id),
                scopes=["read", "write"],
                claims={"user_id": user_id, "user": user_data},
            )


# ---------------------------------------------------------------------------
# Server setup
# ---------------------------------------------------------------------------

verifier = ApiKeyTokenVerifier(api_base_url=API_BASE_URL)
server = FastMCP(
    "career-caddy-public",
    auth=verifier,
    instructions=(
        "You are Career Caddy, a job hunt management assistant. "
        "At the start of every conversation, call get_current_user to learn "
        "who you are acting for. Always address the user by their first name. "
        "Use the available tools to look up real data — never guess."
    ),
)


def _api() -> ApiClient:
    """Build an ApiClient using the authenticated client's token."""
    access = get_access_token()
    if access is None:
        raise RuntimeError("No authenticated session")
    return ApiClient(API_BASE_URL, access.token)


# ---------------------------------------------------------------------------
# Tool: get_current_user (public-server only, not in api_tools)
# ---------------------------------------------------------------------------


@server.tool()
async def get_current_user() -> str:
    """Returns the authenticated user's profile. Use this to know who you are acting as."""
    access = get_access_token()
    if access and access.claims.get("user"):
        return json.dumps(access.claims["user"], indent=2)
    return await _api().get("/api/v1/me/")


# ---------------------------------------------------------------------------
# Companies
# ---------------------------------------------------------------------------


@server.tool()
async def create_company(
    name: str,
    description: Optional[str] = None,
    website: Optional[str] = None,
    industry: Optional[str] = None,
    size: Optional[str] = None,
    location: Optional[str] = None,
) -> str:
    """Create a new company. Measures are taken to avoid duplicate companies."""
    return await api_tools.create_company(
        _api(), name, description, website, industry, size, location
    )


@server.tool()
async def find_company_by_name(company_name: str) -> str:
    """Find a company by name (case-insensitive search)."""
    return await api_tools.find_company_by_name(_api(), company_name)


@server.tool()
async def search_companies(
    query: Optional[str] = None,
    page_size: Optional[int] = None,
) -> str:
    """Search companies by name or display_name (case-insensitive OR match)."""
    return await api_tools.search_companies(_api(), query, page_size)


@server.tool()
async def get_companies(id: Optional[int] = None) -> str:
    """Fetch companies. Pass id to retrieve a single company; omit for the full list."""
    return await api_tools.get_companies(_api(), id)


# ---------------------------------------------------------------------------
# Job Posts
# ---------------------------------------------------------------------------


@server.tool()
async def create_job_post_with_company_check(
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
) -> str:
    """Create a job post, creating the company first if it doesn't exist.

    This is the primary tool for adding jobs. It checks for duplicate URLs and
    resolves or creates the company by name (handles 'Foobar' vs 'Foobar Inc.').
    """
    return await api_tools.create_job_post_with_company_check(
        _api(),
        title, company_name, description, location,
        salary_min, salary_max, employment_type, remote_ok,
        url, link, posted_date,
        company_description, company_website, company_industry,
        company_size, company_location,
    )


@server.tool()
async def find_job_post_by_link(link: str) -> str:
    """Find a job post by its original posting URL.
    A 200 response with data: [] means no job post exists for that link."""
    return await api_tools.find_job_post_by_link(_api(), link)


@server.tool()
async def search_job_posts(
    query: Optional[str] = None,
    title: Optional[str] = None,
    company: Optional[str] = None,
    company_id: Optional[int] = None,
    sort: Optional[str] = None,
    page_size: Optional[int] = None,
) -> str:
    """Search job posts by keyword, title, company name, or company ID.

    Args:
        query: Free-text search across title, description, and company name.
        title: Filter by title only (case-insensitive contains).
        company: Filter by company name only (case-insensitive contains).
        company_id: Filter by exact company ID.
        sort: Sort field, e.g. '-created_at' (prefix '-' for descending).
        page_size: Number of results to return.
    """
    return await api_tools.search_job_posts(
        _api(), query, title, company, company_id, sort, page_size
    )


@server.tool()
async def get_job_posts(
    id: Optional[int] = None,
    sort: Optional[str] = None,
    order: Optional[str] = None,
    page: Optional[int] = None,
    per_page: Optional[int] = None,
) -> str:
    """Fetch job posts. Pass id to retrieve a single post; omit for a paginated list."""
    return await api_tools.get_job_posts(_api(), id, sort, order, page, per_page)


@server.tool()
async def update_job_post(
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
    """Update an existing job post's attributes or company relationship.

    All fields are optional — only provided fields are updated.
    To change the company, pass company_id (use find_company_by_name to look it up).
    """
    return await api_tools.update_job_post(
        _api(), job_post_id, title, description, location,
        salary_min, salary_max, employment_type, remote_ok,
        link, posted_date, company_id,
    )


# ---------------------------------------------------------------------------
# Job Applications
# ---------------------------------------------------------------------------


@server.tool()
async def create_job_application(
    job_post_id: int,
    status: str = "applied",
    notes: Optional[str] = None,
    applied_at: Optional[str] = None,
) -> str:
    """Create a new job application linked to an existing job post.

    job_post_id is the integer ID of the job post.
    status should be one of: applied, interviewing, offered, rejected, withdrawn.
    applied_at: ISO date string (e.g. '2026-03-23').
    """
    return await api_tools.create_job_application(
        _api(), job_post_id, status, notes, applied_at
    )


@server.tool()
async def get_job_applications(
    id: Optional[int] = None,
    sort: Optional[api_tools._APPLICATION_SORT_FIELDS] = None,
    order: Optional[Literal["asc", "desc"]] = None,
    page: Optional[int] = None,
    per_page: Optional[int] = None,
) -> str:
    """Fetch job applications. Pass id for a single application; omit for a list.

    Sort by: id, applied_at, status, job_post_id, company_id, notes.
    Do NOT use 'created_at' — it is not a valid sort field.
    """
    return await api_tools.get_job_applications(
        _api(), id, sort, order, page, per_page
    )


@server.tool()
async def get_applications_for_job_post(job_post_id: int) -> str:
    """Fetch all job applications linked to a specific job post.

    Use this to find the application ID when you need to update an existing application.
    Returning data: [] means no applications exist for that job post.
    """
    return await api_tools.get_applications_for_job_post(_api(), job_post_id)


@server.tool()
async def update_job_application(
    application_id: int,
    status: Optional[str] = None,
    notes: Optional[str] = None,
    applied_at: Optional[str] = None,
    company_id: Optional[int] = None,
) -> str:
    """Update a job application's status, notes, or company association.

    application_id is the application's own ID, NOT the job post ID.
    """
    return await api_tools.update_job_application(
        _api(), application_id, status, notes, applied_at, company_id
    )


# ---------------------------------------------------------------------------
# Career Data
# ---------------------------------------------------------------------------


@server.tool(
    description="Fetch the user's personal career profile: resume, skills, experience, "
    "education, certifications, and cover letters. Use this to score jobs or "
    "answer questions about the user's background. This is NOT job posts."
)
async def get_career_data() -> str:
    return await api_tools.get_career_data(_api())


# ---------------------------------------------------------------------------
# Scrapes
# ---------------------------------------------------------------------------


@server.tool()
async def create_scrape(
    url: str,
    job_post_id: Optional[int] = None,
    company_id: Optional[int] = None,
) -> str:
    """Create a scrape record with status='hold' for later processing.

    Use this to queue a URL for scraping by a separate process. The scrape is
    NOT dispatched immediately — it sits in 'hold' status until picked up.
    """
    return await api_tools.create_scrape(_api(), url, job_post_id, company_id)


@server.tool()
async def get_scrapes(
    id: Optional[int] = None,
    sort: Optional[str] = None,
    page: Optional[int] = None,
    per_page: Optional[int] = None,
) -> str:
    """Fetch scrape records. Pass id to retrieve a single scrape; omit for a paginated list."""
    return await api_tools.get_scrapes(_api(), id, sort, page, per_page)


@server.tool()
async def update_scrape(
    scrape_id: int,
    status: Optional[str] = None,
    job_content: Optional[str] = None,
    url: Optional[str] = None,
) -> str:
    """Update a scrape record's status, content, or URL.

    Common status transitions: hold -> pending, hold -> completed (with job_content).
    """
    return await api_tools.update_scrape(_api(), scrape_id, status, job_content, url)


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------


@server.tool()
async def score_job_post(job_post_id: int) -> str:
    """Score a job post against the user's career data.

    Scores against the user's full career data (all favorite resumes,
    cover letters, answers). No resume selection needed.

    Returns 202 with status='pending'. The API scores asynchronously —
    poll get_scores to check for completion.
    """
    return await api_tools.score_job_post(_api(), job_post_id)


@server.tool()
async def get_scores(
    id: Optional[int] = None,
    job_post_id: Optional[int] = None,
    page: Optional[int] = None,
    per_page: Optional[int] = None,
) -> str:
    """Fetch scores. Pass id for a single score, or filter by job_post_id.

    Use this to check scoring results after calling score_job_post.
    Score attributes: score (int 0-100), status (pending/completed/failed), explanation (text).
    """
    return await api_tools.get_scores(_api(), id, job_post_id, page, per_page)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    host = os.environ.get("FASTMCP_HOST", "0.0.0.0")
    port = int(os.environ.get("FASTMCP_PORT", "8030"))

    logger.info("Starting Career Caddy Public MCP Server")
    logger.info("  API backend: %s", API_BASE_URL)
    logger.info("  Listening on: %s:%s", host, port)
    logger.info("  Auth: API key (jh_*) pass-through")

    server.run(transport="streamable-http", host=host, port=port)


if __name__ == "__main__":
    main()
