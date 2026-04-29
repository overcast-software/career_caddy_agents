"""
Career Caddy MCP Server — local/stdio version.

Used by pydantic-ai agents and the local MCP gateway. Authenticates to the
Career Caddy API using the CC_API_TOKEN environment variable.

Tool implementations live in lib.api_tools and are shared with public_server.py.
"""

import sys
from pathlib import Path

# Add project root to Python path for imports to work when called via stdio
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import logging
from typing import Literal, Optional

import logfire
from fastmcp import FastMCP

from lib.api_tools import ApiClient, _APPLICATION_SORT_FIELDS
from lib import api_tools
from browser.credentials import Credentials  # type: ignore
from lib.models.career_caddy import APICredentials  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from lib.logfire_setup import setup_logfire  # noqa: E402

setup_logfire("career_caddy_server")

# Load credentials
try:
    credentials = Credentials.load_credentials()
    logfire.info(f"Loaded credentials for {len(credentials.domains)} domains")
except FileNotFoundError as e:
    logfire.warn(f"No credentials file found: {e}")
    credentials = Credentials(domains={})
except Exception as e:
    logfire.error(f"Error loading credentials: {e}")
    credentials = Credentials(domains={})


server = FastMCP("career-caddy-server")


def _api() -> ApiClient:
    """Build an ApiClient using CC_API_TOKEN from the environment."""
    creds = APICredentials()
    return ApiClient(creds.base_url, creds.api_token)


# ---------------------------------------------------------------------------
# Companies
# ---------------------------------------------------------------------------


@server.tool()
async def create_company(
    name: str,
    description: Optional[str] = """
    This is the company for which the job post is for.
    Most resources are related to this, measures are taken to keep duplicate company's.
    """,
    website: Optional[str] = None,
    industry: Optional[str] = None,
    size: Optional[str] = None,
    location: Optional[str] = None,
) -> str:
    """Create a new company."""
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
    """Create a job post, creating the company first if it doesn't exist."""
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
    If the response is 200 and data is [] that means there is no job-post for that given link."""
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
    """Search job posts by keyword, title, company name, or company ID."""
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
    """Update an existing job post's attributes or company relationship."""
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
    applied_at: ISO date string (e.g. '2026-03-23'). Do NOT use 'applied_date'.
    """
    return await api_tools.create_job_application(
        _api(), job_post_id, status, notes, applied_at
    )


@server.tool()
async def get_applications_for_job_post(job_post_id: int) -> str:
    """Fetch all job applications linked to a specific job post.

    Use this to find the application ID when you need to update an existing application.
    Returning data: [] means that there are no job applications for that job post.
    """
    return await api_tools.get_applications_for_job_post(_api(), job_post_id)


@server.tool()
async def get_job_applications(
    id: Optional[int] = None,
    sort: Optional[_APPLICATION_SORT_FIELDS] = None,
    order: Optional[Literal["asc", "desc"]] = None,
    page: Optional[int] = None,
    per_page: Optional[int] = None,
) -> str:
    """Fetch job applications. Pass id to retrieve a single application; omit for a list.

    Sort by: id, applied_at, status, job_post_id, company_id, notes.
    Do NOT use 'created_at' — it is not a valid field.
    """
    return await api_tools.get_job_applications(
        _api(), id, sort, order, page, per_page
    )


@server.tool()
async def update_job_application(
    application_id: int,
    status: Optional[str] = None,
    notes: Optional[str] = None,
    applied_at: Optional[str] = None,
    company_id: Optional[int] = None,
) -> str:
    """Update a job application's status, notes, or company association.

    application_id is the application's own ID, NOT the job post ID or company ID.
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


if __name__ == "__main__":
    logfire.info("Starting Career Caddy MCP Server...")
    try:
        server.run()
    except Exception as e:
        logfire.error(f"Failed to start MCP server: {e}")
        sys.exit(1)
