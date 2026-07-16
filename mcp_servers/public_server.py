"""
Career Caddy Public MCP Server — authenticated proxy to the Career Caddy API.

Deployed at careercaddy.online (/mcp). Exposes career-data tools only (no email,
no browser). Each client authenticates with their own jh_* API key, which is
forwarded to the Django API on every request.

    Connect at: https://careercaddy.online/mcp
    Auth:       Authorization: Bearer jh_xxxxx

Security invariants:
    - This file MUST NOT import email_server, browser_server, gateway, or lib/browser/*
    - No CC_API_TOKEN env var — all auth comes from clients
    - No secrets.yml or mail directory access
"""

import logging
import os
import sys
import time
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
from lib.logfire_setup import setup_logfire  # noqa: E402

setup_logfire("public_mcp_server")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_BASE_URL = os.environ.get("CC_API_BASE_URL", "http://localhost:8000")

# How long a successful token verification is reused before re-hitting
# /me/. jh_* keys are long-lived, so a 5-min cache means a revocation
# propagates within <=5 min — acceptable — and it makes the MCP resilient
# to api flap / rate-limits: one good verify at connect is reused across
# the burst of tools/list + tool calls that follow, instead of one extra
# /me/ round-trip per MCP request (the failure mode where a transient
# /me/ non-200 surfaced to clients as a bogus "invalid_token" 401).
_VERIFY_CACHE_TTL_S = 300


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
        # token -> (AccessToken, expiry_monotonic). Only successful
        # verifications are cached; failures are never stored so a
        # transient /me/ error retries on the next call instead of
        # sticking. Concurrent cache misses double-calling /me/ are
        # harmless (idempotent GET), so no lock is needed.
        self._cache: dict[str, tuple[AccessToken, float]] = {}

    def _prune_expired(self, now: float) -> None:
        """Drop expired cache entries to bound memory growth."""
        expired = [tok for tok, (_, exp) in self._cache.items() if now >= exp]
        for tok in expired:
            self._cache.pop(tok, None)

    async def verify_token(self, token: str) -> AccessToken | None:
        if not token.startswith("jh_"):
            return None

        # Reuse a recent successful verification instead of re-hitting
        # /me/ on every MCP request. Without this, a client that connects
        # (one verify) and immediately calls tools/list (another verify)
        # turns any api flap on the second call into a spurious
        # "invalid_token" 401 that drops the whole session.
        now = time.monotonic()
        self._prune_expired(now)
        cached = self._cache.get(token)
        if cached is not None:
            access_token, expiry = cached
            if now < expiry:
                return access_token
            # Expired — fall through to re-verify against /me/.
            self._cache.pop(token, None)

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self.api_base_url}/api/v1/me/",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "X-Forwarded-Proto": "https",
                    },
                )
                if resp.status_code != 200:
                    logger.warning(
                        "Token verify upstream non-200: status=%s url=%s",
                        resp.status_code,
                        f"{self.api_base_url}/api/v1/me/",
                    )
                    return None

                user_data = resp.json().get("data", resp.json())
                user_id = user_data.get("id", "unknown")
                # is_staff lives in the JSON:API resource's attributes; flat
                # dicts (fallback) put it at the top level. The middleware
                # filter that hides enhancer tools from non-staff sessions
                # consults `claims["is_staff"]` directly so per-call code
                # doesn't have to walk into attributes.
                attrs = user_data.get("attributes") if isinstance(user_data, dict) else None
                is_staff = bool(
                    (attrs or user_data or {}).get("is_staff")
                    if isinstance(user_data, dict)
                    else False
                )
                logger.info(
                    "Authenticated user_id=%s is_staff=%s via API key",
                    user_id, is_staff,
                )

                # `scopes=["staff"]` mirrors the claim so future hooks can
                # use either surface; FastMCP scope filtering is the more
                # canonical channel even though we read claims today.
                scopes = ["read", "write"] + (["staff"] if is_staff else [])
                access = AccessToken(
                    token=token,
                    client_id=str(user_id),
                    scopes=scopes,
                    claims={
                        "user_id": user_id,
                        "user": user_data,
                        "is_staff": is_staff,
                    },
                )
                # Cache only on success; expiry is monotonic so it is
                # immune to wall-clock jumps.
                self._cache[token] = (
                    access, time.monotonic() + _VERIFY_CACHE_TTL_S,
                )
                return access
        except Exception:
            # Any verifier exception (httpx network failure, JSON parse,
            # OOM-driven worker death mid-call, etc.) MUST NOT leak as
            # uvicorn 500 — that pattern silently masks the underlying
            # bug and shows up as "MCP broken" to clients. Logging here
            # surfaces the real cause; returning None yields a clean 401
            # with a proper WWW-Authenticate challenge so the client
            # retries instead of giving up.
            logger.exception("Verifier crashed; returning None → 401")
            return None


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

# Hide the scrape-profile-enhancer tools (inspect_scrape_html /
# test_url_rewrite / find_selectors_for_text) from non-staff clients'
# `tools/list` responses. The api enforces authorization on every
# underlying endpoint these tools call, so this filter is a UX /
# prompt-context concern — regular users' LLMs shouldn't burn tokens
# on tool definitions they can't usefully invoke.
from mcp_servers.staff_tool_filter import StaffOnlyToolFilter  # noqa: E402
server.add_middleware(StaffOnlyToolFilter())


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
    from lib.api_tools import TOOL_SHAPES, _respond, _slim_payload
    shape = TOOL_SHAPES["get_current_user"]
    attrs_keep = shape.get("attrs") or []

    access = get_access_token()
    if access and access.claims.get("user"):
        # JWT claims is a flat dict, not JSON:API. Filter to audit attrs.
        user = dict(access.claims["user"])
        slim = {k: v for k, v in user.items() if k in attrs_keep or k == "id"}
        return _respond(slim)

    payload, error, status = await _api().get_data("/api/v1/me/")
    if error is not None:
        return _respond(None, error=error, status_code=status)
    _slim_payload(payload, shape=shape, is_single=True)
    return _respond(payload)


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
async def get_companies(id: Optional[str] = None) -> str:
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
async def get_duplicate_candidates(job_post_id: str) -> str:
    """List likely-duplicate JobPosts for a given post.

    Returns up to 10 peer posts the system suspects represent the
    same role, ordered confidence-desc, recent-first. Each candidate
    has match_signals drawn from:
      - canonical_link (high confidence, exact URL match after
        tracking-param strip + host rewrites)
      - fingerprint (high confidence, same company + normalized
        title + location)
      - title_similarity (medium, same company + one title is a
        prefix/suffix of the other — catches suffix-drift cases
        fingerprint hashing can't)
    Empty list when nothing looks duplicate. Use to verify a stub
    or freshly-extracted post before treating it as unique."""
    return await api_tools.get_duplicate_candidates(_api(), job_post_id)


@server.tool()
async def find_duplicate_candidates(
    title: str,
    company: Optional[str] = None,
    link: Optional[str] = None,
) -> str:
    """Pre-POST duplicate check — does an incoming posting already exist?

    Use BEFORE creating a job post, when you have the incoming title plus
    at least one of: a `link` or a `company` name. Composite over the
    primitives `find_job_post_by_link` + `find_company_by_name` +
    `search_job_posts`; no api endpoint of its own.

    Strategy:
      - link, if given → exact match → confidence='high', signal='link'.
      - company, if given → resolve to company id, list its job posts,
        compare titles locally:
          * exact iexact match → confidence='high', signal='title_exact'.
          * prefix/suffix overlap → confidence='medium',
            signal='title_similarity'.

    Returns {candidates: [...], count}. Each candidate is
    {id, title, company_name, match_signals, confidence, frontend_url},
    same shape as the by-id /duplicate-candidates/ endpoint.

    If neither link nor company is provided, returns an error — title
    alone is too low-signal."""
    return await api_tools.find_duplicate_candidates(_api(), title, company, link)


@server.tool()
async def search_job_posts(
    query: Optional[str] = None,
    title: Optional[str] = None,
    company: Optional[str] = None,
    company_id: Optional[str] = None,
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
    id: Optional[str] = None,
    sort: Optional[str] = None,
    order: Optional[str] = None,
    page: Optional[int] = None,
    per_page: Optional[int] = None,
) -> str:
    """Fetch job posts. Pass id to retrieve a single post; omit for a paginated list."""
    return await api_tools.get_job_posts(_api(), id, sort, order, page, per_page)


@server.tool()
async def update_job_post(
    job_post_id: str,
    title: Optional[str] = None,
    description: Optional[str] = None,
    location: Optional[str] = None,
    salary_min: Optional[int] = None,
    salary_max: Optional[int] = None,
    employment_type: Optional[str] = None,
    remote_ok: Optional[bool] = None,
    link: Optional[str] = None,
    posted_date: Optional[str] = None,
    company_id: Optional[str] = None,
    source: Optional[str] = None,
) -> str:
    """Update an existing job post's attributes or company relationship.

    All fields are optional — only provided fields are updated.
    To change the company, pass company_id (use find_company_by_name to look it up).
    `source` rewrites the JobPost provenance label (email / scrape / manual / paste / chat / email_direct).
    """
    return await api_tools.update_job_post(
        _api(), job_post_id, title, description, location,
        salary_min, salary_max, employment_type, remote_ok,
        link, posted_date, company_id, source,
    )


@server.tool()
async def publish_job_post(job_post_id: str) -> str:
    """Publish a job post to the fediverse (ActivityPub). Owner-only.

    Marks the post public by adding the AS2 Public URI to its audience; the
    private->public transition fans out a Create to your followers.
    Idempotent — publishing an already-public post does nothing (no
    duplicate Create). A 403 (you don't own this post) or 404 (no such post)
    is returned as an error.
    """
    return await api_tools.publish_job_post(_api(), job_post_id)


@server.tool()
async def unpublish_job_post(job_post_id: str) -> str:
    """Unpublish a job post from the fediverse (ActivityPub). Owner-only.

    Removes the AS2 Public URI from the post's audience, flipping it back to
    private. No Withdraw is emitted (V1). Idempotent — unpublishing an
    already-private post does nothing. A 403 (you don't own this post) or
    404 (no such post) is returned as an error.
    """
    return await api_tools.unpublish_job_post(_api(), job_post_id)


# ---------------------------------------------------------------------------
# Job Applications
# ---------------------------------------------------------------------------


@server.tool()
async def create_job_application(
    job_post_id: str,
    status: str = "applied",
    notes: Optional[str] = None,
    applied_at: Optional[str] = None,
) -> str:
    """Create a new job application linked to an existing job post.

    job_post_id is the ID of the job post.
    status should be one of: applied, interviewing, offered, rejected, withdrawn.
    applied_at: ISO date string (e.g. '2026-03-23').
    """
    return await api_tools.create_job_application(
        _api(), job_post_id, status, notes, applied_at
    )


@server.tool()
async def get_job_applications(
    id: Optional[str] = None,
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
async def get_applications_for_job_post(job_post_id: str) -> str:
    """Fetch all job applications linked to a specific job post.

    Use this to find the application ID when you need to update an existing application.
    Returning data: [] means no applications exist for that job post.
    """
    return await api_tools.get_applications_for_job_post(_api(), job_post_id)


@server.tool()
async def update_job_application(
    application_id: str,
    status: Optional[str] = None,
    notes: Optional[str] = None,
    applied_at: Optional[str] = None,
    company_id: Optional[str] = None,
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
    job_post_id: Optional[str] = None,
    company_id: Optional[str] = None,
) -> str:
    """Create a scrape record with status='hold' for later processing.

    Use this to queue a URL for scraping by a separate process. The scrape is
    NOT dispatched immediately — it sits in 'hold' status until picked up.
    """
    return await api_tools.create_scrape(_api(), url, job_post_id, company_id, status="hold")


@server.tool()
async def get_scrapes(
    id: Optional[str] = None,
    sort: Optional[str] = None,
    page: Optional[int] = None,
    per_page: Optional[int] = None,
    status: Optional[str] = None,
) -> str:
    """Fetch scrape records. Pass id to retrieve a single scrape; omit for a paginated list.

    Filter by status (e.g. 'failed', 'completed', 'hold') and sort with e.g. '-id'.
    """
    return await api_tools.get_scrapes(_api(), id, sort, page, per_page, status=status)


@server.tool()
async def update_scrape(
    scrape_id: str,
    status: Optional[str] = None,
    job_content: Optional[str] = None,
    url: Optional[str] = None,
) -> str:
    """Update a scrape record's status, content, or URL.

    Common status transitions: hold -> pending, hold -> completed (with job_content).
    """
    return await api_tools.update_scrape(_api(), scrape_id, status, job_content, url)


@server.tool()
async def list_scrape_screenshots(scrape_id: str) -> str:
    """List screenshot filenames captured for a scrape. Staff-only.

    Returns JSON with a list of filenames that can be passed to
    fetch_scrape_screenshot to retrieve the PNG bytes.
    """
    return await api_tools.list_screenshots(_api(), scrape_id)


@server.tool()
async def get_scrape_graph_trace(scrape_id: str) -> str:
    """Fetch the pydantic-graph node trace for a scrape. Owner-or-staff.

    Returns ordered transitions: scrape_id, graph_node, graph_payload,
    note, created_at — plus meta.chain walking the source_scrape
    parents so a tracker URL + its canonical child render as one path.
    Use this to diagnose why a scrape ended in `failed` / `error` /
    `ExtractFail` / `ObstacleFail` — the terminating node + its
    payload usually has the reason.
    """
    return await api_tools.get_scrape_graph_trace(_api(), scrape_id)


@server.tool()
async def get_scrape_statuses(scrape_id: str) -> str:
    """Fetch the full status history for a scrape. Owner-or-staff.

    Returns every ScrapeStatus row (not just rows with a graph_node),
    in JSON:API resource shape. Includes exception text and other
    internal-only diagnostic detail in `note` / `graph_payload`. Use
    when get_scrape_graph_trace returns nothing (pre-graph or
    pre-cutover scrapes) to recover whatever the legacy poller wrote.
    """
    return await api_tools.get_scrape_statuses(_api(), scrape_id)


@server.tool()
async def fetch_scrape_screenshot(scrape_id: str, filename: str) -> str:
    """Download a scrape screenshot as a base64-encoded PNG. Staff-only.

    The caller should base64-decode the result to get raw PNG bytes, e.g. to
    pass into a vision model as BinaryContent(media_type="image/png").
    """
    import base64
    from lib.api_tools import _respond
    data = await api_tools.fetch_screenshot_bytes(_api(), scrape_id, filename)
    return _respond({
        "scrape_id": scrape_id,
        "filename": filename,
        "media_type": "image/png",
        "size_bytes": len(data),
        "data_base64": base64.b64encode(data).decode("ascii"),
    })


@server.tool()
async def get_scrape_profile(hostname: str) -> str:
    """Fetch the scrape profile for a hostname. Returns the JSON:API payload."""
    return await api_tools.get_scrape_profile(_api(), hostname)


@server.tool()
async def update_scrape_profile(
    profile_id: str,
    css_selectors: Optional[dict] = None,
    extraction_hints: Optional[str] = None,
    page_structure: Optional[str] = None,
    preferred_tier: Optional[str] = None,
    enabled: Optional[bool] = None,
    apply_resolver_config: Optional[dict] = None,
    extension_selectors: Optional[dict] = None,
    url_rewrites: Optional[list] = None,
) -> str:
    """Update a ScrapeProfile's editable fields.

    ScrapeProfile carries several JSONB blobs by design so per-host tuning
    can land at runtime without a Django migration. Pass only the fields you
    want to update — others are left untouched.

    JSONB fields and their consumers:
      - css_selectors:         hold-poller (browser scraping)
      - apply_resolver_config: server-side ResolveApplyUrl graph node
      - extension_selectors:   ccsender browser extension
      - url_rewrites:          canonicalize_link (dedup canonical forms)
      - extraction_hints:      Tier1/2/3 LLM extractors (free-text)
    """
    attrs: dict = {}
    if css_selectors is not None:
        attrs["css_selectors"] = css_selectors
    if extraction_hints is not None:
        attrs["extraction_hints"] = extraction_hints
    if page_structure is not None:
        attrs["page_structure"] = page_structure
    if preferred_tier is not None:
        attrs["preferred_tier"] = preferred_tier
    if enabled is not None:
        attrs["enabled"] = enabled
    if apply_resolver_config is not None:
        attrs["apply_resolver_config"] = apply_resolver_config
    if extension_selectors is not None:
        attrs["extension_selectors"] = extension_selectors
    if url_rewrites is not None:
        attrs["url_rewrites"] = url_rewrites
    if not attrs:
        from lib.api_tools import _respond
        return _respond(None, error="No fields provided to update")
    return await api_tools.update_scrape_profile(_api(), profile_id, **attrs)


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------


@server.tool()
async def score_job_post(job_post_id: str) -> str:
    """Score a job post against the user's career data.

    Scores against the user's full career data (all favorite resumes,
    cover letters, answers). No resume selection needed.

    Returns 202 with status='pending'. The API scores asynchronously —
    poll get_scores to check for completion.
    """
    return await api_tools.score_job_post(_api(), job_post_id)


@server.tool()
async def get_scores(
    id: Optional[str] = None,
    job_post_id: Optional[str] = None,
    page: Optional[int] = None,
    per_page: Optional[int] = None,
) -> str:
    """Fetch scores. Pass id for a single score, or filter by job_post_id.

    Use this to check scoring results after calling score_job_post.
    Score attributes: score (int 0-100), status (pending/completed/failed), explanation (text).
    """
    return await api_tools.get_scores(_api(), id, job_post_id, page, per_page)


# ---------------------------------------------------------------------------
# Tools: scrape-profile-enhancer support
# ---------------------------------------------------------------------------
# These three tools exist so the scrape-profile-enhancer agent can fill out
# ScrapeProfile fields (ready_selector, apply_link_selectors, url_rewrites)
# from real captured HTML without burning context on raw 200KB blobs or
# making selector decisions blind. Each delegates the actual parsing to
# `lib/scrape_inspector.py` so the logic is unit-testable end-to-end
# without spinning up MCP.


async def _fetch_scrape_html(scrape_id: str) -> tuple[Optional[str], Optional[str]]:
    """Return (html, error). Caller decides how to surface the error."""
    payload, error, status = await _api().get_data(
        f"/api/v1/scrapes/{scrape_id}/"
    )
    if error is not None:
        return None, error
    data = (payload or {}).get("data") or {}
    attrs = data.get("attributes") or {}
    html = attrs.get("html")
    if not html:
        return None, (
            f"scrape {scrape_id} has no html (status="
            f"{attrs.get('status')!r}); html is only persisted by the "
            "browser-driven scrape path, not by paste / email ingest"
        )
    return html, None


@server.tool()
async def inspect_scrape_html(
    scrape_id: str,
    selector: Optional[str] = None,
    mode: Optional[Literal["trim", "skeleton", "selector"]] = None,
    max_chars: Optional[int] = None,
    max_matches: Optional[int] = None,
) -> str:
    """Inspect a scrape's captured HTML with server-side BS4 trim.

    The scrape-profile-enhancer's first stop when designing selectors
    against a host. Server-side trim drops `<script>` / `<style>` /
    comments / tracking pixels and inline event handlers so the
    output is hundreds of times smaller than the raw `scrape.html`
    blob while preserving every attribute the enhancer might anchor
    a selector to (id, class, data-testid, aria-*, role).

    Modes (`mode` defaults to `selector` when `selector` is passed,
    `trim` otherwise):
      - `trim`     — full trimmed HTML, capped at `max_chars` (≤40k).
      - `skeleton` — tag/class/id tree with text stripped; best for
                     first-pass orientation on huge pages.
      - `selector` — run `selector` against the trimmed HTML and
                     return matches with an outline path + text
                     snippet + attrs; cap at `max_matches` (≤100).

    Uses standard CSS3 selectors (BS4 `.select()`). Playwright-engine
    pseudo-selectors like `h2:has-text("...")` will return an error —
    use `find_selectors_for_text` for that flow.
    """
    from lib.api_tools import _respond
    from lib.scrape_inspector import (
        extract_skeleton, query_selector, trim_html,
    )

    html, error = await _fetch_scrape_html(scrape_id)
    if error is not None or html is None:
        return _respond(None, error=error, status_code=404)

    effective_mode = mode or ("selector" if selector else "trim")
    chars_cap = max_chars or 40_000

    if effective_mode == "skeleton":
        return _respond({
            "scrape_id": scrape_id,
            "mode": "skeleton",
            "html_size_bytes": len(html),
            "skeleton": extract_skeleton(html, limit_chars=chars_cap),
        })
    if effective_mode == "selector":
        if not selector:
            return _respond(
                None,
                error="mode='selector' requires a `selector` argument",
                status_code=400,
            )
        try:
            result = query_selector(
                html, selector, max_matches=max_matches or 25,
            )
        except Exception as exc:
            return _respond(
                None,
                error=f"selector failed to parse: {exc}",
                status_code=400,
            )
        result["scrape_id"] = scrape_id
        result["html_size_bytes"] = len(html)
        return _respond(result)
    return _respond({
        "scrape_id": scrape_id,
        "mode": "trim",
        "html_size_bytes": len(html),
        "trimmed_html": trim_html(html, limit_chars=chars_cap),
    })


@server.tool()
async def test_url_rewrite(
    url: str,
    hostname: Optional[str] = None,
) -> str:
    """Dry-run a host's ScrapeProfile `url_rewrites` against a URL.

    Mirrors what `canonicalize_link` does at JobPost.save() time and
    what the scrape-graph does at Navigate time. Use it to validate a
    new `url_rewrites` rule before committing it via
    `update_scrape_profile`, OR to debug why an incoming URL is (or
    isn't) collapsing onto an existing canonical_link.

    Resolves the profile by hostname (derived from `url` when not
    explicitly passed; `www.` stripped). Top-level `url_rewrites` is
    preferred; falls back to the legacy nested `css_selectors.url_rewrites`
    location for profiles that haven't been migrated.

    Returns:
      {"input": <url>, "hostname": <host>, "rule_count": N,
       "rewritten": <new_url>, "changed": bool, "matched_rule": {...}}
    """
    from lib.api_tools import _respond
    from lib.scrape_inspector import derive_hostname

    host = (hostname or derive_hostname(url) or "").lower()
    if not host:
        return _respond(
            None,
            error="could not derive a hostname from `url` and none was passed",
            status_code=400,
        )

    payload, error, status = await _api().get_data(
        f"/api/v1/scrape-profiles/?filter[hostname]={host}"
    )
    if error is not None:
        return _respond(None, error=error, status_code=status)

    records = (payload or {}).get("data") or []
    if not records:
        return _respond({
            "input": url, "hostname": host,
            "rule_count": 0, "rewritten": url, "changed": False,
            "note": f"no ScrapeProfile exists for {host}",
        })
    attrs = (records[0] or {}).get("attributes") or {}
    rules = attrs.get("url_rewrites")
    if not rules and isinstance(attrs.get("css_selectors"), dict):
        rules = attrs["css_selectors"].get("url_rewrites")
    rules = rules or []

    # Try each rule in order; first one that changes the URL wins.
    # Mirrors `apply_url_rewrites` exactly so test results match prod.
    import re
    rewritten = url
    matched: Optional[dict] = None
    for idx, rule in enumerate(rules):
        if not isinstance(rule, dict):
            continue
        pattern = rule.get("match")
        replacement = rule.get("rewrite")
        if not pattern or replacement is None:
            continue
        try:
            new_url, n = re.subn(pattern, replacement, url)
        except re.error:
            continue
        if n > 0 and new_url != url:
            rewritten = new_url
            matched = {"index": idx, "match": pattern, "rewrite": replacement}
            break

    return _respond({
        "input": url,
        "hostname": host,
        "rule_count": len(rules),
        "rewritten": rewritten,
        "changed": rewritten != url,
        "matched_rule": matched,
    })


@server.tool()
async def find_selectors_for_text(
    scrape_id: str,
    text: str,
    max_results: Optional[int] = None,
    case_insensitive: Optional[bool] = None,
) -> str:
    """Propose ranked stable selectors anchoring `text` in a scrape's HTML.

    Direct support for `ready_selector` / `apply_button_selectors`
    design — given the text the enhancer wants to wait for or click
    ("About the job", "Apply now", "Easy Apply"), return CSS selectors
    that match it, ordered by stability heuristic:

      data-testid > role > aria-label > stable id > single semantic
      class > multi-class composite > bare tag

    Hashed-looking ids/classes (`__a1b2c3`, Tailwind atomic classes,
    css-in-js artifacts) are filtered out — they churn on every deploy
    and don't anchor reliable selectors. The returned candidates use
    plain CSS3 syntax so they round-trip through `inspect_scrape_html(...,
    mode='selector')` for verification before the enhancer commits them
    to the profile via `update_scrape_profile`.
    """
    from lib.api_tools import _respond
    from lib.scrape_inspector import (
        find_selectors_for_text as _find,
    )

    html, error = await _fetch_scrape_html(scrape_id)
    if error is not None or html is None:
        return _respond(None, error=error, status_code=404)
    result = _find(
        html,
        text,
        max_results=max_results or 10,
        case_insensitive=True if case_insensitive is None else case_insensitive,
    )
    result["scrape_id"] = scrape_id
    return _respond(result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _probe_upstream_api() -> None:
    """Fail fast if the upstream API is unreachable or misrouted.

    Catches the class of bug where the container starts but every token
    verify silently returns a non-200 (e.g. SSL redirect, DisallowedHost,
    wrong CC_API_BASE_URL). Crashlooping with a clear reason beats a
    running container that 401s every request.
    """
    url = f"{API_BASE_URL}/api/v1/healthcheck/"
    try:
        resp = httpx.get(url, headers={"X-Forwarded-Proto": "https"}, timeout=5.0)
    except httpx.HTTPError as exc:
        logger.error("Upstream probe failed: %s (url=%s)", exc, url)
        sys.exit(1)
    if resp.status_code != 200:
        logger.error(
            "Upstream probe non-200: status=%s url=%s body=%s",
            resp.status_code, url, resp.text[:200],
        )
        sys.exit(1)
    logger.info("Upstream probe ok: %s", url)


def main():
    host = os.environ.get("FASTMCP_HOST", "0.0.0.0")
    port = int(os.environ.get("FASTMCP_PORT", "8030"))

    logger.info("Starting Career Caddy Public MCP Server")
    logger.info("  API backend: %s", API_BASE_URL)
    logger.info("  Listening on: %s:%s", host, port)
    logger.info("  Auth: API key (jh_*) pass-through")

    _probe_upstream_api()
    server.run(transport="streamable-http", host=host, port=port)


if __name__ == "__main__":
    main()
