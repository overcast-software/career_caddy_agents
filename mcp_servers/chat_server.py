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

from pydantic_ai import Agent  # noqa: E402
from lib.toolsets import CareerCaddyToolset, CareerCaddyDeps  # noqa: E402

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
- If a tool returns {{"success": false}}, report the error and stop — do not retry

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


def _build_agent(user_profile: str) -> Agent:
    """Build a fresh agent with all career caddy tools."""
    return Agent(
        DEFAULT_MODEL,
        deps_type=CareerCaddyDeps,
        toolsets=[CareerCaddyToolset(scope="all")],
        system_prompt=SYSTEM_PROMPT.format(user_profile=user_profile),
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

    if not message:
        return JSONResponse({"error": "message is required"}, status_code=400)
    if not token:
        return JSONResponse({"error": "token is required"}, status_code=400)

    async def event_stream():
        user_profile = await _fetch_user_profile(token)
        agent = _build_agent(user_profile)
        deps = CareerCaddyDeps(api_token=token, base_url=API_BASE_URL)

        # Build message history for context
        messages = []
        for msg in history[-20:]:  # limit context window
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                messages.append({"role": "user", "content": content})
            elif role == "assistant":
                messages.append({"role": "assistant", "content": content})

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

                done_event = json.dumps({
                    "type": "done",
                    "content": full_text,
                    "conversation_id": conversation_id,
                })
                yield f"data: {done_event}\n\n"

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
