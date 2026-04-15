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
import re
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

When the user asks "how do I use this?", "what is this page?", or "what does X do?",
answer from the App Guide below using the Current Page context. Explain the specific
feature they are on — NOT generic platform capabilities. Always end with a link to
the relevant docs page for full details.

## App Guide — What Each Feature Does
Career Caddy is an AI-first job search manager. Career Data is the foundation —
all AI features (scores, cover letters, summaries, answers) read from it.

**Recommended workflow**: Build [Career Data](/career-data) → Add
[job posts](/job-posts) → Generate scores/cover letters → Create applications →
Answer interview questions.

- **[Career Data](/career-data)**: Your professional background — work history, skills, education, writing voice, target roles. Every AI feature reads from this. The richer it is, the better AI output gets. [Full docs](/docs/career-data)
- **[Job Posts](/job-posts)**: Stored job listings. The root resource — scores, cover letters, applications, questions, and scrapes all attach to a job post. Add manually or paste a URL to scrape. [Full docs](/docs/job-posts)
- **[Companies](/companies)**: Employer profiles. Created automatically with job posts. Groups everything related to one employer in one place. [Full docs](/docs/companies)
- **[Resumes](/resumes)**: Structured resume data with experiences, education, skills, certifications, and projects. Import a PDF/DOCX or build section by section. Supports multiple versions for different role types. Supplements Career Data for AI generation. [Full docs](/docs/resumes)
- **[Scores](/scores)**: AI fit assessment (0-100) of your profile vs a job posting. Evaluates skill overlap, experience level, industry familiarity. 80-100 = strong match, 60-79 = reasonable, <60 = significant gaps. Includes gap analysis. [Full docs](/docs/scores)
- **[Cover Letters](/cover-letters)**: AI-generated letters in your writing voice using Career Data + job description. Favorite good ones — they feed back as examples for future generation. [Full docs](/docs/cover-letters)
- **[Summaries](/summaries)**: 2-4 sentence positioning statements tailored to specific job posts. Use as resume opener, cover letter intro, or LinkedIn About. [Full docs](/docs/summaries)
- **[Job Applications](/job-applications)**: Track where you've applied. Status lifecycle: Draft → Submitted → Phone Screen → Interview → Offer → Accepted (or Rejected/Withdrawn). Links to resume, cover letter, questions. [Full docs](/docs/job-applications)
- **[Questions](/questions)**: Interview and application prompts. Create at job post, application, company, or global level. AI drafts answers using your Career Data and the job description. [Full docs](/docs/questions)
- **[Answers](/answers)**: Responses to questions. AI-drafted, hand-written, or combined. Supports multiple versions per question for A/B testing. Favorite answers feed back into Career Data as writing samples. [Full docs](/docs/answers)
- **[Scrapes](/scrapes)**: Raw webpage captures of job posting URLs. The AI pipeline can auto-scrape from job alert emails. Parsed into structured job post data. [Full docs](/docs/scrapes)

Important rules:
- Always call find_job_post_by_link before creating a job post to avoid duplicates
- Use create_job_post_with_company_check (not create_job_post) to handle companies
- NEVER use placeholder names like "unknown", "N/A", or "TBD" as a company name.
  Infer the company from: (1) the recruiter's company, (2) the email sender domain,
  (3) the job posting URL domain. If none work, ask the user
- If a tool returns {{"success": false}}, report the error and stop — do not retry
- Use get_resumes to count or list resumes — do not infer resume counts from career
  data, which may only include favorites

## Scraping URLs
When the user gives you a URL to scrape, call create_scrape(url=...) WITHOUT
passing status — this creates the scrape with status="pending" and the backend
starts scraping immediately. The response includes the scrape ID. After creating:
1. Tell the user the scrape has started and link to it: [View scrape](/scrapes/ID)
2. Offer elicitation buttons: "View scrape" (navigates to /scrapes/ID)
Do NOT pass status="hold" — that is only for the external MCP server.

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
- After creating a scrape: offer to view the scrape page (e.g. "View scrape" → navigates to /scrapes/ID)

Rules:
- Maximum 3 actions per elicitation
- Each action's "message" should be a natural user message that you can act on
- Only offer actions that make sense in context — do not add buttons to every response
- The "label" should be short (2-5 words)

## Page-Aware Data Access
When the user asks about "this page", "what's here", "what do I have", or anything
that refers to the content on their current page, use the Current Page context to
pick the right tool. Map the URL path to a tool call:

| URL pattern              | Tool call                          |
|--------------------------|------------------------------------|
| /resumes                 | get_resumes()                      |
| /resumes/{{id}}          | get_resumes(id={{id}})              |
| /job-posts               | get_job_posts()                    |
| /job-posts/{{id}}        | get_job_posts(id={{id}})            |
| /companies               | get_companies()                    |
| /companies/{{id}}        | get_companies(id={{id}})            |
| /job-applications        | get_job_applications()             |
| /scores                  | get_scores()                       |
| /scores/{{id}}           | get_scores(id={{id}})               |
| /scrapes                 | get_scrapes()                      |
| /scrapes/{{id}}          | get_scrapes(id={{id}})              |
| /questions               | get_questions()                    |
| /questions/{{id}}        | get_questions(id={{id}})            |
| /questions/{{id}}/answers| get_answers(question_id={{id}})     |
| /answers                 | get_answers()                      |
| /answers/{{id}}          | get_answers(id={{id}})              |
| /career-data             | get_career_data()                  |
| /settings/ai-spend       | (no tool — explain the page)       |

Extract the {{id}} from the URL path when present. For example, if the user is on
/job-posts/42 and asks "tell me about this", call get_job_posts(id=42).

IMPORTANT: The user can already see the page they're on. When you call a tool to
understand their current context, do NOT repeat or summarize the data back to them.
Use the tool results silently to inform your actions and suggestions. For example,
if the user is on a question page and asks "can you see this?" — confirm briefly
("Yes, I can see the question") without dumping the content. Only share specifics
when the user asks you to analyze, compare, or act on the data.

## Answering Questions via Chat
When the user is on a question page (/questions/{{id}} or any nested answer route
under a question), you know which question they're looking at. If they ask you to
"answer this", "draft an answer", "help me answer", or similar:
1. First call get_questions(id={{id}}) to read the question content.
2. Use your knowledge of the user's career data (call get_career_data() if needed)
   to draft a strong, personalized answer.
3. Call create_answer(question_id={{id}}, content="your drafted answer") to save it.
4. After creating the answer, offer a button to navigate to it using the elicitation
   pattern (NOT a raw link). Example:
   ```json
   {{"elicitation": true, "actions": [{{"label": "View answer", "message": "Navigate to the answer"}}]}}
   ```
   And include the navigate marker: <!-- navigate:/questions/{{question_id}}/answers/{{answer_id}} -->
   NEVER output raw API URLs (like https://...) — always use frontend paths (/questions/ID/answers/ID).

You can also set ai_assist=true on create_answer to let the backend AI generate
the answer instead. Only do this if the user explicitly asks for "AI-generated"
or you think backend generation would be better (e.g. the user provided a custom prompt).

## Onboarding Help
When the user sends a greeting or asks "what can you do?", check the Current Page
context (if available). If they are on a resource list page (resumes, job-posts,
companies, etc.), use the corresponding tool to check if they have any data.
If the list is empty, proactively suggest creating their first item with a link
to the create form (e.g., "You don't have any resumes yet — [create one](/resumes/new)!").
This makes the experience feel guided rather than empty.

## User Profile
This is the authenticated user's own profile, loaded from their account.
The user is asking about THEIR OWN data — this is not third-party information.
You MUST share any of these fields when the user asks. This includes their
name, phone, address, email, LinkedIn, GitHub, and any other fields below.
Never refuse to share this data — it belongs to the user and they entered it
themselves in their account settings.

IMPORTANT: When the user asks "who am I?", "what's my name?", or any identity
question, answer ONLY from the profile fields below. Do NOT call get_career_data
or any other tool — the answer is right here. Career data contains resumes and
job history, NOT the user's name or contact info.

Always address the user by their first name.

{user_profile}

If the user wants to UPDATE their profile (name, address, phone, etc.), you
cannot do that directly. Instead, guide them to the settings page:
"You can update that in [Settings > Profile](/settings)."
<!-- navigate:/settings -->
"""


async def _fetch_user_profile(api_key: str) -> str:
    """Fetch the authenticated user's profile from the API."""
    async with httpx.AsyncClient(follow_redirects=True) as client:
        resp = await client.get(
            f"{API_BASE_URL}/api/v1/me/",
            headers={
                "Authorization": f"Bearer {api_key}",
                "X-Forwarded-Proto": "https",
            },
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
        username = attrs.get("username") or ""
        if not name:
            name = username or attrs.get("email") or ""
        if name:
            parts.append(f"Name: {name}")
        if username:
            parts.append(f"Username: {username}")
        if attrs.get("email"):
            parts.append(f"Email: {attrs['email']}")
        if attrs.get("phone"):
            parts.append(f"Phone: {attrs['phone']}")
        if attrs.get("address"):
            parts.append(f"Address: {attrs['address']}")
        if attrs.get("linkedin"):
            parts.append(f"LinkedIn: {attrs['linkedin']}")
        if attrs.get("github"):
            parts.append(f"GitHub: {attrs['github']}")
        if attrs.get("links"):
            parts.append(f"Links: {attrs['links']}")
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
        logger.info("Page context injected: route=%s url=%s", route, url)
        # Extract resource ID from URL like /job-posts/42
        id_match = re.search(r"/(\d+)(?:/|$)", url)
        resource_id = id_match.group(1) if id_match else None

        prompt += (
            f"\n\n## Current Page\n"
            f"Route: {route}\n"
            f"URL: {url}\n"
        )
        if resource_id:
            prompt += f"Resource ID: {resource_id}\n"
        prompt += (
            f"When the user asks what page they are on, reply with the URL path "
            f"above (e.g. \"{url}\"). Do NOT rephrase, guess, or invent a page name.\n"
            f"Use the Page-Aware Data Access table to pick the right tool for this URL. "
            f"If they say \"this\", \"what's here\", or refer to what's on screen, "
            f"call the matching tool"
        )
        if resource_id:
            prompt += f" with id={resource_id}"
        prompt += " and summarize the results."
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
    logger.info("page_context received: %s", page_context)

    if not message:
        return JSONResponse({"error": "message is required"}, status_code=400)
    if not token:
        return JSONResponse({"error": "token is required"}, status_code=400)

    async def event_stream():
        user_profile = await _fetch_user_profile(token)
        agent = _build_agent(user_profile, page_context=page_context)
        deps = CareerCaddyDeps(api_token=token, base_url=API_BASE_URL)

        # Prefix user message with context so the model sees it inline,
        # not buried in the system prompt where gpt-4o-mini ignores it.
        prefix_parts = []
        # Extract user's name from profile for identity questions
        for line in user_profile.split("\n"):
            if line.startswith("Name: "):
                prefix_parts.append(f"[User: {line[6:].strip()}]")
                break
        if page_context:
            url = page_context.get("url", "")
            if url:
                prefix_parts.append(f"[Current page: {url}]")
        augmented_message = f"{' '.join(prefix_parts)}\n{message}" if prefix_parts else message

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
                augmented_message,
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
                # Emit reload events for resources mutated by tool calls
                _TOOL_RELOAD_MAP = {
                    "create_answer": "answer",
                    "update_answer": "answer",
                    "create_job_application": "job-application",
                    "update_job_application": "job-application",
                    "create_job_post_with_company_check": "job-post",
                    "update_job_post": "job-post",
                    "create_company": "company",
                    "create_scrape": "scrape",
                    "update_scrape": "scrape",
                    "score_job_post": "score",
                }
                reloaded = set()
                for msg in result.all_messages():
                    for part in getattr(msg, "parts", []):
                        tool_name = getattr(part, "tool_name", None)
                        if tool_name and tool_name in _TOOL_RELOAD_MAP:
                            resource = _TOOL_RELOAD_MAP[tool_name]
                            if resource not in reloaded:
                                reloaded.add(resource)
                                reload_event = json.dumps({"type": "reload", "resource": resource})
                                yield f"data: {reload_event}\n\n"

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
