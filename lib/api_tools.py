"""
Shared Career Caddy API tool implementations.

Both career_caddy_server.py (local, CC_API_TOKEN) and public_server.py
(public, per-client auth) import from here. The ApiClient is instantiated
with a base_url and a token — the caller decides where the token comes from.
"""

import json
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import urljoin

import httpx
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


class ApiClient:
    """HTTP client that forwards a token to the Career Caddy API."""

    def __init__(self, base_url: str, token: str, timeout: int = 120):
        self.base_url = base_url
        self.timeout = timeout
        self._headers = {
            "Authorization": f"Bearer {token}",
            "X-Forwarded-Proto": "https",
        }

    def _ok(self, response: httpx.Response) -> str:
        if response.status_code in (200, 201, 202):
            body = response.json()
            _inject_frontend_urls(body)
            result = APIResponse(
                success=True, data=body, status_code=response.status_code
            )
        else:
            text = response.text[:500] if len(response.text) > 500 else response.text
            result = APIResponse(
                success=False,
                error=f"{response.status_code} - {text}",
                status_code=response.status_code,
            )
        return json.dumps(result.model_dump(), indent=2)

    async def get(self, path: str, params: dict | None = None) -> str:
        async with httpx.AsyncClient(follow_redirects=True, timeout=self.timeout) as client:
            resp = await client.get(
                urljoin(self.base_url, path),
                headers=self._headers,
                params=params,
            )
            return self._ok(resp)

    async def post(self, path: str, payload: dict) -> str:
        async with httpx.AsyncClient(follow_redirects=True, timeout=self.timeout) as client:
            resp = await client.post(
                urljoin(self.base_url, path),
                headers=self._headers,
                json=payload,
            )
            return self._ok(resp)

    async def patch(self, path: str, payload: dict) -> str:
        async with httpx.AsyncClient(follow_redirects=True, timeout=self.timeout) as client:
            resp = await client.patch(
                urljoin(self.base_url, path),
                headers=self._headers,
                json=payload,
            )
            return self._ok(resp)

    async def post_file(self, path: str, file_path: Path, field: str = "file") -> str:
        """POST a file as multipart/form-data."""
        async with httpx.AsyncClient(follow_redirects=True, timeout=self.timeout) as client:
            with open(file_path, "rb") as f:
                resp = await client.post(
                    urljoin(self.base_url, path),
                    headers={"Authorization": self._headers["Authorization"]},
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
        payload = {
            "data": {
                "type": "company",
                "attributes": data.model_dump(exclude_none=True),
            }
        }
        return await api.post("/api/v1/companies/", payload)
    except ValueError as e:
        return json.dumps(
            APIResponse(success=False, error=f"Validation error: {e}").model_dump(),
            indent=2,
        )


async def find_company_by_name(api: ApiClient, company_name: str) -> str:
    """Find a company by name (case-insensitive search)."""
    result = await api.get("/api/v1/companies/", params={"filter[query]": company_name})
    data = json.loads(result)
    if data.get("success"):
        companies = data.get("data", {}).get("data", [])
        if companies:
            return json.dumps(
                APIResponse(
                    success=True,
                    data={"companies": companies, "count": len(companies)},
                    status_code=200,
                ).model_dump(),
                indent=2,
            )
        return json.dumps(
            APIResponse(
                success=False,
                error=f"No companies found matching '{company_name}'",
                status_code=404,
            ).model_dump(),
            indent=2,
        )
    return result


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
    return await api.get("/api/v1/companies/", params=params)


async def get_companies(api: ApiClient, id: Optional[int] = None) -> str:
    """Fetch companies. Pass id to retrieve a single company; omit for the full list."""
    if id is not None:
        return await api.get(f"/api/v1/companies/{id}/")
    return await api.get("/api/v1/companies/")


async def find_job_post_by_link(api: ApiClient, link: str) -> str:
    """Find a job post by its original posting URL."""
    return await api.get("/api/v1/job-posts/", params={"filter[link]": link})


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
) -> str:
    """Create a job post, creating the company first if it doesn't exist."""
    job_url = url or link
    if not company_name:
        return json.dumps(
            APIResponse(
                success=False, error="company_name is required to create a job post"
            ).model_dump(),
            indent=2,
        )

    _PLACEHOLDER_NAMES = {"unknown", "n/a", "na", "none", "tbd", "not specified", ""}
    if company_name.strip().lower() in _PLACEHOLDER_NAMES:
        return json.dumps(
            APIResponse(
                success=False,
                error=(
                    f"'{company_name}' is not an acceptable company name. "
                    "Infer the company from: (1) the recruiter's company, "
                    "(2) the email sender domain, (3) the job posting URL domain. "
                    "If you cannot determine the company, ask the user."
                ),
            ).model_dump(),
            indent=2,
        )

    try:
        # Check for duplicate by URL
        if job_url:
            existing = json.loads(await find_job_post_by_link(api, job_url))
            if existing.get("success"):
                posts = existing.get("data", {}).get("data", [])
                if posts:
                    existing_id = posts[0].get("id")
                    return json.dumps(
                        APIResponse(
                            success=False,
                            error=f"Duplicate: job post with this link already exists (id={existing_id})",
                            status_code=409,
                            data={"duplicate": True, "existing_id": existing_id},
                        ).model_dump(),
                        indent=2,
                    )

        # Search for existing company
        company_search = json.loads(await find_company_by_name(api, company_name))
        company_id = None
        if company_search.get("success"):
            companies = company_search.get("data", {}).get("companies", [])
            if companies:
                company_id = int(companies[0].get("id"))

        # Create company if not found
        if company_id is None:
            create_result = json.loads(
                await create_company(
                    api,
                    name=company_name,
                    description=company_description,
                    website=company_website,
                    industry=company_industry,
                    size=company_size,
                    location=company_location,
                )
            )
            if create_result.get("success"):
                company_id = int(
                    create_result.get("data", {}).get("data", {}).get("id")
                )
            else:
                return json.dumps(
                    APIResponse(
                        success=False,
                        error=f"Failed to create company: {create_result.get('error')}",
                    ).model_dump(),
                    indent=2,
                )

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
        payload = {
            "data": {
                "type": "job-post",
                "attributes": attributes,
                "relationships": {
                    "company": {"data": {"type": "company", "id": str(company_id)}}
                },
            }
        }
        return await api.post("/api/v1/job-posts/", payload)

    except Exception as e:
        return json.dumps(
            APIResponse(
                success=False,
                error=f"Error creating job post with company check: {e}",
            ).model_dump(),
            indent=2,
        )


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
    return await api.get("/api/v1/job-posts/", params=params)


async def get_job_posts(
    api: ApiClient,
    id: Optional[int] = None,
    sort: Optional[str] = None,
    order: Optional[str] = None,
    page: Optional[int] = None,
    per_page: Optional[int] = None,
) -> str:
    """Fetch job posts. Pass id for a single post; omit for a paginated list."""
    if id is not None:
        return await api.get(f"/api/v1/job-posts/{id}/")
    params = {}
    if sort is not None:
        params["sort"] = sort
    if order is not None:
        params["order"] = order
    if page is not None:
        params["page"] = page
    if per_page is not None:
        params["per_page"] = per_page
    return await api.get("/api/v1/job-posts/", params=params)


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
        return json.dumps(
            APIResponse(success=False, error="No fields provided to update").model_dump(),
            indent=2,
        )

    payload: dict = {
        "data": {
            "type": "job-post",
            "id": str(job_post_id),
            "attributes": attributes,
        }
    }
    if company_id is not None:
        payload["data"]["relationships"] = {
            "company": {"data": {"type": "company", "id": str(company_id)}}
        }
    return await api.patch(f"/api/v1/job-posts/{job_post_id}/", payload)


async def create_job_application(
    api: ApiClient,
    job_post_id: int,
    status: str = "applied",
    notes: Optional[str] = None,
    applied_at: Optional[str] = None,
) -> str:
    """Create a new job application linked to an existing job post."""
    if job_post_id <= 0:
        return json.dumps(
            APIResponse(
                success=False,
                error=f"Invalid job_post_id={job_post_id}. Look up the real ID first.",
            ).model_dump(),
            indent=2,
        )

    attributes: dict = {"status": status}
    if notes is not None:
        attributes["notes"] = notes
    if applied_at is not None:
        attributes["applied_at"] = applied_at

    payload = {
        "data": {
            "type": "job-application",
            "attributes": attributes,
            "relationships": {
                "job-post": {"data": {"type": "job-post", "id": str(job_post_id)}}
            },
        }
    }
    return await api.post("/api/v1/job-applications/", payload)


async def get_job_applications(
    api: ApiClient,
    id: Optional[int] = None,
    sort: Optional[_APPLICATION_SORT_FIELDS] = None,
    order: Optional[Literal["asc", "desc"]] = None,
    page: Optional[int] = None,
    per_page: Optional[int] = None,
) -> str:
    """Fetch job applications. Pass id for a single application; omit for a list."""
    if id is not None:
        return await api.get(f"/api/v1/job-applications/{id}/")
    params = {}
    if sort is not None:
        params["sort"] = sort
    if order is not None:
        params["order"] = order
    if page is not None:
        params["page"] = page
    if per_page is not None:
        params["per_page"] = per_page
    return await api.get("/api/v1/job-applications/", params=params)


async def get_applications_for_job_post(api: ApiClient, job_post_id: int) -> str:
    """Fetch all job applications linked to a specific job post."""
    return await api.get(f"/api/v1/job-posts/{job_post_id}/job-applications/")


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
        return json.dumps(
            APIResponse(success=False, error="No fields provided to update").model_dump(),
            indent=2,
        )

    payload: dict = {
        "data": {
            "type": "job-application",
            "id": str(application_id),
            "attributes": attributes,
        }
    }
    if company_id is not None:
        payload["data"]["relationships"] = {
            "company": {"data": {"type": "company", "id": str(company_id)}}
        }
    return await api.patch(f"/api/v1/job-applications/{application_id}/", payload)


async def get_career_data(api: ApiClient) -> str:
    """Fetch the user's personal career profile."""
    return await api.get("/api/v1/career-data/")


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
    if id is not None:
        return await api.get(f"/api/v1/resumes/{id}/")
    params = {}
    if favorite is not None:
        params["favorite"] = str(favorite).lower()
    if page is not None:
        params["page"] = page
    if per_page is not None:
        params["per_page"] = per_page
    return await api.get("/api/v1/resumes/", params=params)


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

    payload: dict = {"data": {"type": "scrape", "attributes": attributes}}
    if relationships:
        payload["data"]["relationships"] = relationships

    return await api.post("/api/v1/scrapes/", payload)


async def get_scrapes(
    api: ApiClient,
    id: Optional[int] = None,
    sort: Optional[str] = None,
    page: Optional[int] = None,
    per_page: Optional[int] = None,
    status: Optional[str] = None,
) -> str:
    """Fetch scrape records. Pass id for a single scrape; omit for a paginated list. Use sort='-id' for most recent first, per_page=1 for just the latest. Filter by status with status='hold'."""
    if id is not None:
        return await api.get(f"/api/v1/scrapes/{id}/")
    params = {}
    if sort is not None:
        params["sort"] = sort
    if page is not None:
        params["page"] = page
    if per_page is not None:
        params["per_page"] = per_page
    if status is not None:
        params["filter[status]"] = status
    return await api.get("/api/v1/scrapes/", params=params)


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
        return json.dumps(
            APIResponse(success=False, error="No fields provided to update").model_dump(),
            indent=2,
        )

    payload = {
        "data": {
            "type": "scrape",
            "id": str(scrape_id),
            "attributes": attributes,
        }
    }
    return await api.patch(f"/api/v1/scrapes/{scrape_id}/", payload)


async def upload_screenshot(api: ApiClient, scrape_id: int, file_path: Path) -> str:
    """Upload a screenshot PNG to the API for a scrape."""
    return await api.post_file(
        f"/api/v1/scrapes/{scrape_id}/screenshots/", file_path
    )


async def get_scrape_profile(api: ApiClient, hostname: str) -> str:
    """Fetch the scrape profile for a hostname. Returns profile data or empty."""
    return await api.get("/api/v1/scrape-profiles/", params={"filter[hostname]": hostname})


async def update_scrape_profile(api: ApiClient, profile_id: int, **attrs) -> str:
    """Update a scrape profile's editable fields (css_selectors, extraction_hints, etc.)."""
    json_attrs = {}
    for key, value in attrs.items():
        json_attrs[key.replace("_", "-")] = value
    payload = {
        "data": {
            "type": "scrape-profile",
            "id": str(profile_id),
            "attributes": json_attrs,
        }
    }
    return await api.patch(f"/api/v1/scrape-profiles/{profile_id}/", payload)


async def get_questions(
    api: ApiClient,
    id: Optional[int] = None,
    company_id: Optional[int] = None,
    job_post_id: Optional[int] = None,
    page: Optional[int] = None,
    per_page: Optional[int] = None,
) -> str:
    """Fetch interview questions. Pass id for a single question; omit for a paginated list. Filter by company_id or job_post_id."""
    if id is not None:
        return await api.get(f"/api/v1/questions/{id}/")
    params = {}
    if company_id is not None:
        params["filter[company_id]"] = company_id
    if job_post_id is not None:
        params["filter[job_post_id]"] = job_post_id
    if page is not None:
        params["page"] = page
    if per_page is not None:
        params["per_page"] = per_page
    return await api.get("/api/v1/questions/", params=params)


async def get_answers(
    api: ApiClient,
    id: Optional[int] = None,
    question_id: Optional[int] = None,
    favorite: Optional[bool] = None,
    page: Optional[int] = None,
    per_page: Optional[int] = None,
) -> str:
    """Fetch answers to interview questions. Pass id for a single answer; omit for a paginated list. Filter by question_id or favorite status."""
    if id is not None:
        return await api.get(f"/api/v1/answers/{id}/")
    params = {}
    if question_id is not None:
        params["filter[question_id]"] = question_id
    if favorite is not None:
        params["favorite"] = str(favorite).lower()
    if page is not None:
        params["page"] = page
    if per_page is not None:
        params["per_page"] = per_page
    return await api.get("/api/v1/answers/", params=params)


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
    if id is not None:
        return await api.get(f"/api/v1/scores/{id}/")
    params = {}
    if job_post_id is not None:
        params["filter[job_post_id]"] = job_post_id
    if page is not None:
        params["page"] = page
    if per_page is not None:
        params["per_page"] = per_page
    return await api.get("/api/v1/scores/", params=params)
