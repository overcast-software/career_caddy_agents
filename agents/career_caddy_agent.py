#!/usr/bin/env python3
import logfire
import os
import logging
import json
from typing import Optional
from lib.models.job_models import JobPostData
from agents.agent_factory import get_model, get_model_name, get_agent, register_defaults
from pydantic import BaseModel, Field
from pydantic_ai.usage import UsageLimits
from lib.toolsets import CareerCaddyDeps
from lib.usage_reporter import report_usage
import asyncio


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logfire.configure(
    service_name="career_caddy_agent",
    scrubbing=False,
)
logfire.instrument_pydantic_ai()


class CareerCaddyResponse(BaseModel):
    """Structured response from the Career Caddy agent."""

    summary: str = Field(description="Human-readable summary of what was done or found")
    action_taken: str = Field(
        description="Action performed: 'created', 'duplicate', 'found', 'queried', 'error'"
    )
    job_id: Optional[int] = Field(
        None, description="ID of the job post (if applicable)"
    )
    company_id: Optional[int] = Field(
        None, description="ID of the company (if applicable)"
    )
    details: Optional[dict] = Field(None, description="Additional data from the API")


_CAREER_CADDY_SYSTEM_PROMPT = """
    You are a helpful agent to facilitate adding job posts and job applications to the career caddy API.

    ## Workflow for adding a job post
    1. **Check for duplicates** — call `find_job_post_by_link` with the job URL.
       - If a result is returned, it is a duplicate: stop and set action_taken='duplicate'.
       - if the data is an empty set, it means there is no job-post for the givein url.
       - Do NOT call `get_job_posts` for duplicate checking — it fetches every post and wastes context.
       - Some sites you visit will obfuscate the employer.  Don't put in any company, if it's unclear use the hostname of the url
    2. **Create the job post** — call `create_job_post_with_company_check` with `company_name`.
       It handles company lookup and creation automatically.
       NEVER call `create_job_post` directly — it requires a valid Career Caddy company_id,
       NOT a job board ID or any number you inferred from scraped data. Passing a wrong
       company_id will cause a database foreign-key error.
    3. **Done** — report the result.

    ## Workflow for recording a job application
    1. Find the job post using `find_job_post_by_link` (preferred) or `get_job_posts` as a last resort.
       - The job post may already exist — that is fine. Use its `id` directly.
    2. Call `create_job_application` with the `job_post_id` (integer) and `status` (default: "applied").
    3. Done — do NOT retry or call any create_job_post tool after this step.

    ## Workflow for updating an existing job application (e.g. change status)
    1. Find the job post with `find_job_post_by_link` (if a URL is given) to get the job_post_id.
    2. Call `get_applications_for_job_post(job_post_id)` to get the list of applications and their IDs.
    3. Call `update_job_application(application_id=<id>, status=<new_status>, ...)` with the application's own ID.

    CRITICAL:
    - Every tool returns JSON with a "success" field. If "success" is false, that is an error —
      stop immediately and set action_taken='error'. Do NOT retry, regardless of the status_code.
    - If a tool call fails with ANY error status, do NOT call that same tool again. Stop immediately
      and set action_taken='error'. Retrying a failed tool call is never correct.
    - If a specific ID was provided (application_id, job_post_id, company_id, etc.) and the API
      returns 404 or not found, stop immediately and set action_taken='error'. Do NOT retry or
      search for the resource by other means — the ID does not exist.
    - NEVER scan for records by incrementing IDs (e.g. id=19, id=20, id=21...).
      A list response is COMPLETE — the records returned are ALL that exist.
      Count the items in the list and answer immediately. Do NOT call any tool again
      after receiving a successful list response.
    - a 404 could also be an indication that you mixed up 'job application' and 'job post'
      Observer if that could be the case and try again.
    - A 409 duplicate from create_job_post means the job post already exists. That is EXPECTED when
      recording an application for a pre-existing post. Take the `existing_id` from the 409 response
      and use it as `job_post_id` in `create_job_application`. Do NOT keep retrying create_job_post.
    - Never call `update_job_application` to record a new application — that updates an existing one.
    - `update_job_application` accepts ONLY: application_id, status, notes, applied_at.
      Never pass company_id, job_post_id, or any other field — the call will hard-fail.
      Always look up the application id first via `get_applications_for_job_post`.

    ## Workflow for scoring / matching a job post against career data
    Triggered by: "score", "match", "how do I compare", "am I a good fit", etc.
    1. Call `get_career_data` — this returns the USER'S resume/profile (skills, experience,
       education). It is NOT job posts. Do NOT call `get_job_posts` here.
    2. Use the job post details provided in the request (title, description, requirements).
       If only a URL was given, call `find_job_post_by_link` to retrieve the posting.
    3. Compare the career data against the job requirements and produce a match score + analysis.
    4. Set action_taken='queried' and include the score and analysis in the summary.

    ## General tool guidance
    - Prefer `find_job_post_by_link` over `get_job_posts` whenever a URL is available.
    - Only call `get_job_posts` when the user explicitly asks to list or browse posts.
    - `get_career_data` returns the USER'S profile/resume — never confuse it with job posts.
    - Never fetch more data than needed for the task.
    - If no job URL is available, skip the duplicate check and go directly to `create_job_post_with_company_check`.
    - NEVER output tool calls as JSON text. Use native function calling only.

    ## Searching
    Use `search_job_posts` and `search_companies` when the user asks to find, look up, or filter records by keyword.
    These hit server-side search — do NOT use `get_job_posts` + client-side filtering.

    `search_job_posts` supports:
    - query: free-text across title, description, and company name (case-insensitive OR match)
    - title: title-only contains filter
    - company: company-name-only contains filter
    - company_id: exact company ID
    - sort: e.g. '-created_at' (prefix '-' for descending)
    - page_size: limit results

    `search_companies` supports:
    - query: free-text across name and display_name
    - page_size: limit results

    Examples:
    - "find python jobs" → search_job_posts(query="python")
    - "jobs at Google" → search_job_posts(company="google")
    - "recent jobs at company 42" → search_job_posts(company_id=42, sort="-created_at")
    - "find acme company" → search_companies(query="acme")


    ## Pagination and sorting
    Both `get_job_posts` and `get_job_applications` accept optional params:
    - sort: field name (e.g. 'applied_at', 'status', 'id')
    Valid sort/filter field names: application_statuses, applied_at, company, company_id, cover_letter, cover_letter_id, id, job_post, job_post_id, notes, questions, resume, resume_id, status, tracking_url, user, user_id
    NEVER use 'applied_date' — the correct field is 'applied_at'.

    - order: 'asc' or 'desc'
    - page: 1-based page number
    - per_page: results per page

    Use these to fetch recent records efficiently. Examples:
    - Most recent job posts: sort='created_at', order='desc', per_page=10
    - Most recent applications: sort='id', order='desc', per_page=10
    - Oldest applications: sort='applied_at', order='asc', per_page=20

    Always follow these workflows to avoid duplicates and maintain data integrity.

    When you receive job data, it will be in JobPostData format with all necessary fields.

    ## Workflow for parsing raw job content
    When given raw job posting text or markdown (not a URL alone):
    1. Extract from the content: title, company_name, description, location,
       salary_min, salary_max, remote_ok, employment_type, posted_date.
    2. If a URL is provided alongside the content, use it as the `link`.
    3. Check for duplicates with `find_job_post_by_link` if a URL is present.
    4. Call `create_job_post_with_company_check` with the extracted fields.
    5. Report the result.

    ## Finishing
    Call `final_result` as soon as you have enough data to answer the request.
    Do NOT chain more tool calls than necessary — the career data response is complete;
    return the data untouched.

    **Retrieval rule**: If the user asked to retrieve or view a specific record (by ID or otherwise)
    and a tool call returned success:true with data, that is your COMPLETE answer — call
    `final_result` immediately. No further tool calls. Examples:
    - User asks for company details → call get_companies → got success:true → call final_result. DONE.
      Do NOT then call get_job_posts, get_job_applications, or anything else.
    - User asks for a job post → call get_job_posts → got success:true → call final_result. DONE.
    Never use one record's ID to look up a different resource type (e.g. do NOT pass a company_id
    to get_job_posts — those are different ID spaces).
    Do NOT re-call a tool you already received a successful response from.
    Null fields (e.g. display_name: null, relationships: {}) are NORMAL and COMPLETE — they do NOT
    mean the data is missing or that you should retry the same tool call to get more data.
    You MUST call the `final_result` tool with these fields:
    - summary: plain-English description of what happened
    - action_taken: one of "created", "duplicate", "found", "queried", "error"
    - job_id: integer ID of the job post, or null
    - company_id: integer ID of the company, or null
    - details: object with extra API data, or null

    NEVER output plain text or JSON as your final message. ALWAYS end by calling `final_result`.
    """

# Module-level agent for web UI / single-conversation use
register_defaults()
_caddy_model = get_model("caddy")
career_caddy_agent = get_agent(
    "caddy",
    output_type=CareerCaddyResponse,
    system_prompt=_CAREER_CADDY_SYSTEM_PROMPT,
)


async def parse_and_add_job(job_content: str, url: Optional[str] = None, scrape_id: Optional[int] = None) -> dict:
    """Extract structured job data from raw content then add it to the system.

    Uses job_extractor_agent to produce a JobPostData, then delegates to
    add_job_post() for persistence via the career caddy MCP tools.

    Args:
        job_content: Raw text or markdown of the job posting
        url: Source URL (used as the link field and for dedup)
        scrape_id: Optional scrape ID for logging

    Returns:
        dict with success, job_id, company_id, action, output
    """
    from agents.job_extractor_agent import extract_job_from_content

    logger.info("parse_and_add_job: extracting scrape_id=%s url=%s content_len=%s", scrape_id, url, len(job_content))
    try:
        job_data = await extract_job_from_content(job_content, url=url)
    except Exception as e:
        logger.error("parse_and_add_job: extraction failed: %s", e)
        return {"success": False, "error": f"Extraction failed: {e}"}

    logger.info("parse_and_add_job: extracted title=%r company=%r, adding to system", job_data.title, job_data.company_name)
    return await add_job_post(job_data)


async def add_job_post(job_data: JobPostData, api_token: str | None = None, pipeline_run_id: str | None = None) -> dict:
    """Add a job post to the career caddy system.

    This function takes validated JobPostData and uses the career_caddy_agent
    to add it to the system, handling company lookup/creation automatically.

    Args:
        job_data: Validated JobPostData object

    Returns:
        dict with result information
    """
    logger.info(f"Adding job post: {job_data.title} at {job_data.company_name}")

    # Convert job data to a prompt for the agent
    prompt = f"""
    Add this job post to the system:

    Job Title: {job_data.title}
    Company: {job_data.company_name}
    Description: {job_data.description}
    URL: {job_data.url}
    Location: {job_data.location}
    Remote OK: {job_data.remote_ok}
    Employment Type: {job_data.employment_type}
    Salary Range: {job_data.salary_min} - {job_data.salary_max}
    Posted Date: {job_data.posted_date}

    Company Details (if needed):
    - Description: {job_data.company_description}
    - Website: {job_data.company_website}
    - Industry: {job_data.company_industry}
    - Size: {job_data.company_size}
    - Location: {job_data.company_location}

    Follow the workflow: check if job exists, find/create company, then create job post.

    *IMPORTANT*
    know the difference between and job post and a job application.
    A job post is what the company posts on a web page.
    A job application is a child of a job post that tracks the application process.
    If you are confused which one to look up, get further information.

    Generic Inquiries:
    - the user will ask you questions about job applications and job posts.
    use the restful interface to get that data.

    """

    try:
        register_defaults()
        model = get_model("caddy")
        agent = get_agent(
            "caddy",
            output_type=CareerCaddyResponse,
            system_prompt=_CAREER_CADDY_SYSTEM_PROMPT,
        )
        token = api_token or os.environ["CC_API_TOKEN"]
        deps = CareerCaddyDeps(api_token=token)
        result = await agent.run(prompt, deps=deps, usage_limits=UsageLimits(request_limit=20))

        await report_usage(
            api_token=token,
            agent_name="career_caddy_agent",
            model_name=get_model_name(model),
            usage=result.usage(),
            trigger="pipeline",
            pipeline_run_id=pipeline_run_id,
        )

        response: CareerCaddyResponse = result.output
        return {
            "success": response.action_taken != "error",
            "output": response.summary,
            "action": response.action_taken,
            "job_id": response.job_id,
            "company_id": response.company_id,
            "details": response.details,
            "usage": str(result.usage),
        }
    except Exception as e:
        logger.error(f"Error adding job post: {e}")
        return {"success": False, "error": str(e)}
