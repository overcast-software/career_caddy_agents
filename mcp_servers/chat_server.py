"""
Career Caddy Chat Server — SSE streaming chat endpoint for the frontend.

Accepts a user message + auth token, runs a pydantic-ai agent with
CareerCaddyToolset, and streams the response as Server-Sent Events.

This server is internal-only (not exposed to the internet). The Django API
proxies to it after validating the user's JWT.

Auth pattern (Option C — JWT pass-through):
    The Django proxy validates the JWT and forwards the raw token. This
    server uses that same token for /api/v1/me/ (profile fetch) and all
    downstream tool calls via ApiClient. Accepts both JWTs and jh_* API
    keys — the Django API's auth stack handles either transparently.

Security invariants:
    - This file MUST NOT import email_server, browser_server, or lib/browser/*
    - No CC_API_TOKEN env var — auth comes from the calling API proxy
    - No secrets.yml or mail directory access
"""

import asyncio
import json
import logging
import os
import sys
import uuid
from pathlib import Path

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

# Add project root so lib imports work
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart  # noqa: E402
from lib.toolsets import CareerCaddyDeps  # noqa: E402
from lib.usage_reporter import report_usage  # noqa: E402
from agents.agent_factory import get_agent, register_defaults  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_BASE_URL = os.environ.get("CC_API_BASE_URL", "http://localhost:8000")
DEFAULT_MODEL = os.environ.get("CHAT_MODEL", "openai:gpt-4o-mini")

SYSTEM_PROMPT = """\
You are Career Caddy, a helpful AI assistant for job hunting. You help users
manage their job search: finding job posts, tracking applications, managing
resumes and cover letters, scoring jobs against their career profile, and
answering questions about their career data.

You have access to the user's Career Caddy account through tools. Use them
to look up real data when answering questions. Be concise and helpful.

When the user asks about their data, always use the appropriate tool to fetch
current information rather than guessing.

Important rules:
- Always call find_job_post_by_link before creating a job post to avoid duplicates
- Use create_job_post_with_company_check (not create_job_post) to handle companies
- NEVER use placeholder names like "unknown", "N/A", or "TBD" as a company name.
  Infer the company from: (1) the recruiter's company, (2) the email sender domain,
  (3) the job posting URL domain. If none work, ask the user
- If a tool returns {{"success": false}}, report the error and stop — do not retry
- Use get_resumes to count or list resumes — do not infer resume counts from career
  data, which may only include favorites

## Frontend URLs — CRITICAL
ALWAYS provide frontend links when referencing resources. NEVER give API paths
(like /api/v1/...). Use relative paths only (e.g. /job-posts/244, not
http://localhost:4200/job-posts/244) — the user may not be on localhost.

When a tool returns a resource with an ID, IMMEDIATELY include a markdown link.
Do not wait for the user to ask for the URL — proactively include it.

Primary resources:
- Job post:          /job-posts/{{id}}
- Job application:   /job-applications/{{id}}
- Company:           /companies/{{id}}
- Score:             /scores/{{id}}
- Resume:            /resumes/{{id}}
- Cover letter:      /cover-letters/{{id}}
- Question:          /questions/{{id}}
- Summary:           /summaries/{{id}}
- Scrape:            /scrapes/{{id}}

Nested views (child resources under a parent):
- Job post scores:        /job-posts/{{id}}/scores
- Job post applications:  /job-posts/{{id}}/job-applications
- Job post questions:     /job-posts/{{id}}/questions
- Job post cover letters: /job-posts/{{id}}/cover-letters
- Job post scrapes:       /job-posts/{{id}}/scrapes
- Job post summaries:     /job-posts/{{id}}/summaries
- Company job posts:      /companies/{{id}}/job-posts
- Company scores:         /companies/{{id}}/scores
- Question answers:       /questions/{{id}}/answers

Create forms:
- New job post:        /job-posts/new
- New application:     /job-applications/new  or  /job-posts/{{id}}/job-applications/new
- New score:           /scores/new
- New question:        /questions/new  or  /job-posts/{{id}}/questions/new
- Scrape a URL:        /job-posts/scrape

Other pages:
- Career data:    /career-data
- Settings:       /settings
- AI Spend:       /settings/ai-spend

Examples:
- "Here's your [score](/scores/42) for that post."
- "View the [job post](/job-posts/7) or its [scores](/job-posts/7/scores)."
- "You can [create an application](/job-posts/7/job-applications/new) for it."

EVERY response that mentions a resource MUST include a clickable link.
Tool responses include a _frontend_url field on each resource — USE IT directly
in your markdown links. Example: if a tool returns {{"_frontend_url": "/scrapes/44"}},
link to it as [scrape](/scrapes/44). NEVER link to external job URLs as the
"where to find it" — those are the scraped source, not the app page.
If a tool returns id=244 for a job post, your response must contain
[job post](/job-posts/244) — not "/api/v1/job-posts/244", not "job post #244".

CRITICAL: ALWAYS use markdown link syntax: [text](/path). NEVER output raw HTML
anchor tags like <a href="...">. Raw HTML links produce malformed URLs (e.g.
https://resumes/31 instead of /resumes/31) and break SPA navigation.

## Navigation — IMPORTANT
When the user says "take me to", "go to", "navigate to", "open", or "show me"
a page, they want to be NAVIGATED there — NOT shown the data. Do NOT call tools
to fetch the resource. Instead, respond with a short confirmation and include a
hidden HTML comment that triggers navigation:

<!-- navigate:/resumes -->

The frontend detects this marker and navigates the user's browser automatically.

Navigation targets (use these, not resource IDs, for list pages):
- "my resumes"       → <!-- navigate:/resumes -->
- "my job posts"     → <!-- navigate:/job-posts -->
- "my applications"  → <!-- navigate:/job-applications -->
- "my companies"     → <!-- navigate:/companies -->
- "my scores"        → <!-- navigate:/scores -->
- "career data"      → <!-- navigate:/career-data -->
- "settings"         → <!-- navigate:/settings -->

For a specific resource, use the ID:
- "show me job post 42" → <!-- navigate:/job-posts/42 -->

Always include a visible markdown link too so the user sees where they're going.
Example response: "Taking you to your [resumes](/resumes) now! <!-- navigate:/resumes -->"

Only fetch data when the user asks a QUESTION about the data (e.g. "how many
resumes do I have?", "what jobs have I applied to?"). If they just want to GO
somewhere, navigate — do not dump data.

## Action Buttons (Elicitation)
When you complete an action that has a natural follow-up, offer the user quick
action buttons by including a JSON block at the END of your response (after all
text). Use this exact format — the frontend will render it as clickable buttons:

```json
{{"elicitation": true, "actions": [{{"label": "Button text", "message": "What to send as chat message"}}, ...]}}
```

Examples of when to offer buttons:
- After creating a job post: offer to score it or create an application
- After scoring: offer to view the score details
- When the user has no resumes: offer to navigate to the create form
- After finding a job post: offer to view it, score it, or apply

Rules:
- Maximum 3 actions per elicitation
- Each action's "message" should be a natural user message that you can act on
- Only offer actions that make sense in context — do not add buttons to every response
- The "label" should be short (2-5 words)

## Onboarding Help
When the user sends a greeting or asks "what can you do?", check the Current Page
context (if available). If they are on a resource list page (resumes, job-posts,
companies, etc.), use the corresponding tool to check if they have any data.
If the list is empty, proactively suggest creating their first item with a link
to the create form (e.g., "You don't have any resumes yet — [create one](/resumes/new)!").
This makes the experience feel guided rather than empty.

## User Profile
The following profile was loaded from the user's account when this session
started. You KNOW this information — it is not a guess. Always address the
user by their first name. If asked "what's my name?" or similar identity
questions, answer directly from this data. Never say you cannot access the
user's profile — the data below IS their profile.

{user_profile}
"""


async def _fetch_user_profile(api_key: str) -> str:
    """Fetch the authenticated user's profile from the API."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{API_BASE_URL}/api/v1/me/",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        if resp.status_code != 200:
            logger.warning(
                "Profile fetch failed: status=%s url=%s",
                resp.status_code,
                f"{API_BASE_URL}/api/v1/me/",
            )
            return "Could not load user profile."
        data = resp.json().get("data", resp.json())
        attrs = data.get("attributes", data)
        parts = []
        name = " ".join(
            filter(None, [attrs.get("first_name"), attrs.get("last_name")])
        )
        if not name:
            name = attrs.get("username") or attrs.get("email") or ""
        if name:
            parts.append(f"Name: {name}")
        if attrs.get("email"):
            parts.append(f"Email: {attrs['email']}")
        if attrs.get("linkedin"):
            parts.append(f"LinkedIn: {attrs['linkedin']}")
        if attrs.get("github"):
            parts.append(f"GitHub: {attrs['github']}")
        if parts:
            logger.info("Resolved user profile: %s", parts[0])
        return (
            "\n".join(parts)
            if parts
            else "User is authenticated but has not filled in their profile yet."
        )


register_defaults()


def _build_agent(user_profile: str, page_context: dict | None = None):
    """Build a fresh agent with all career caddy tools."""
    prompt = SYSTEM_PROMPT.format(user_profile=user_profile)
    if page_context:
        route = page_context.get("route", "unknown")
        url = page_context.get("url", "")
        prompt += (
            f"\n\n## Current Page\n"
            f"The user is currently viewing: {route} ({url})\n"
            f"Use this context to understand what they're looking at. If they say "
            f'"this job post" or "these resumes", they mean the resource on their '
            f"current page. If they ask about what's on screen, use the appropriate "
            f"tool to fetch data for the resource ID in the URL."
        )
    return get_agent(
        "chat",
        system_prompt=prompt,
    )


async def chat(request: Request):
    """POST /chat — streaming chat endpoint.

    Request body:
        {
            "message": "What job posts do I have?",
            "token": "<jwt-or-api-key>",
            "history": [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
            ],
            "conversation_id": "optional-uuid"
        }

    The token field accepts either a JWT (forwarded by the Django proxy)
    or a jh_* API key (for direct callers). The Django API's auth stack
    handles both transparently.

    Response: text/event-stream with events:
        data: {"type": "text", "content": "partial text..."}
        data: {"type": "tool_call", "name": "get_job_posts", "args": {...}}
        data: {"type": "tool_result", "name": "get_job_posts", "result": "..."}
        data: {"type": "done", "content": "full response text"}
        data: {"type": "error", "content": "error message"}
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    message = body.get("message", "").strip()
    token = body.get("token", "").strip()
    history = body.get("history", [])
    conversation_id = body.get("conversation_id", str(uuid.uuid4()))
    page_context = body.get("page_context")

    if not message:
        return JSONResponse({"error": "message is required"}, status_code=400)
    if not token:
        return JSONResponse({"error": "token is required"}, status_code=400)

    async def event_stream():
        user_profile = await _fetch_user_profile(token)
        agent = _build_agent(user_profile, page_context=page_context)
        deps = CareerCaddyDeps(api_token=token, base_url=API_BASE_URL)

        # Build message history for context
        messages = []
        for msg in history[-20:]:  # limit context window
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                messages.append(ModelRequest(parts=[UserPromptPart(content=content)]))
            elif role == "assistant":
                messages.append(ModelResponse(parts=[TextPart(content=content)]))

        try:
            async with agent.run_stream(
                message,
                deps=deps,
                message_history=messages if messages else None,
            ) as result:
                full_text = ""
                async for chunk in result.stream_text(delta=True):
                    full_text += chunk
                    event = json.dumps({"type": "text", "content": chunk})
                    yield f"data: {event}\n\n"

                usage = result.usage()
                logger.info(
                    "Chat usage: request_tokens=%s response_tokens=%s total=%s requests=%s",
                    usage.request_tokens, usage.response_tokens,
                    usage.total_tokens, usage.requests,
                )
                done_event = json.dumps({
                    "type": "done",
                    "content": full_text,
                    "conversation_id": conversation_id,
                    "usage": {
                        "request_tokens": usage.request_tokens or 0,
                        "response_tokens": usage.response_tokens or 0,
                        "total_tokens": usage.total_tokens or 0,
                    },
                })
                yield f"data: {done_event}\n\n"

                asyncio.create_task(report_usage(
                    api_token=token,
                    agent_name="career_caddy_chat",
                    model_name=DEFAULT_MODEL,
                    usage=usage,
                    trigger="chat",
                    base_url=API_BASE_URL,
                ))

        except Exception as e:
            logger.exception("Chat agent error")
            error_event = json.dumps({"type": "error", "content": str(e)})
            yield f"data: {error_event}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def health(request: Request):
    """GET /health — simple health check."""
    return JSONResponse({"status": "ok"})


app = Starlette(
    routes=[
        Route("/chat", chat, methods=["POST"]),
        Route("/health", health, methods=["GET"]),
    ],
)


def main():
    host = os.environ.get("CHAT_HOST", "0.0.0.0")
    port = int(os.environ.get("CHAT_PORT", "8000"))

    logger.info("Starting Career Caddy Chat Server")
    logger.info("  API backend: %s", API_BASE_URL)
    logger.info("  Model: %s", DEFAULT_MODEL)
    logger.info("  Listening on: %s:%s", host, port)

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
