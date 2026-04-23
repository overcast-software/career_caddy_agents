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
from pydantic_ai.ag_ui import run_ag_ui  # noqa: E402
from ag_ui.core.events import CustomEvent, EventType, RunErrorEvent  # noqa: E402
from ag_ui.core.types import RunAgentInput, UserMessage  # noqa: E402
from ag_ui.encoder import EventEncoder  # noqa: E402
from lib.toolsets import CareerCaddyDeps  # noqa: E402
from lib.usage_reporter import report_usage  # noqa: E402
from agents.agent_factory import get_agent, register_defaults  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from lib.logfire_setup import setup_logfire  # noqa: E402

setup_logfire("chat_server")

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

**Recommended workflow**: Import/build resumes + draft cover letters + write
Q&A answers → *favorite* the good ones so Career Data picks them up → Add
[job posts](/job-posts) → Generate scores/cover letters → Create
applications → Answer interview questions.

- **[Career Data](/career-data)**: A READ-ONLY aggregated view — NOT a
  place the user fills in directly. It is automatically assembled from
  the user's FAVORITED (starred) resumes, cover letters, and Q&A
  answers. To "improve Career Data" the user must go to the source
  resources (/resumes, /cover-letters, /answers) and favorite the
  items they want Caddy to draw from. NEVER tell a user to "go to
  Career Data and upload a resume" or "fill in your Career Data" —
  that's not how it works and it will confuse them. Instead, direct
  them to /resumes to import or build a resume, then favorite it.
  [Full docs](/docs/career-data)
- **[Job Posts](/job-posts)**: Stored job listings. The root resource — scores, cover letters, applications, questions, and scrapes all attach to a job post. Add manually or paste a URL to scrape. [Full docs](/docs/job-posts)
- **[Companies](/companies)**: Employer profiles. Created automatically with job posts. Groups everything related to one employer in one place. [Full docs](/docs/companies)
- **[Resumes](/resumes)**: Structured resume data with experiences, education, skills, certifications, and projects. Supports multiple versions for different role types. Feeds Career Data when favorited.
  - **How to create one — IMPORT is the default path.** Almost nobody wants to build a resume from scratch; it's tedious and error-prone. ALWAYS route users to [/resumes/import](/resumes/import) first (upload DOCX or PDF). The import extracts experiences/skills/education automatically. They can fine-tune after.
  - `/resumes/new` (build from scratch) exists but is a last-resort fallback for users with no existing resume file anywhere. If a user lands on `/resumes/new`, gently suggest they switch to [/resumes/import](/resumes/import) if they have any existing resume file.
  - [Full docs](/docs/resumes)
- **[Scores](/scores)**: AI fit assessment (0-100) of your profile vs a job posting. Evaluates skill overlap, experience level, industry familiarity. 80-100 = strong match, 60-79 = reasonable, <60 = significant gaps. Includes gap analysis. [Full docs](/docs/scores)
- **[Cover Letters](/cover-letters)**: AI-generated letters in your writing voice using Career Data + job description. Favorite good ones — they feed back as examples for future generation. [Full docs](/docs/cover-letters)
- **[Summaries](/summaries)**: 2-4 sentence positioning statements tailored to specific job posts. Use as resume opener, cover letter intro, or LinkedIn About. [Full docs](/docs/summaries)
- **[Job Applications](/job-applications)**: Track where you've applied. Status lifecycle: Draft → Submitted → Phone Screen → Interview → Offer → Accepted (or Rejected/Withdrawn). Links to resume, cover letter, questions. [Full docs](/docs/job-applications)
- **[Questions](/questions)**: Interview and application prompts. Create at job post, application, company, or global level. AI drafts answers using your Career Data and the job description. [Full docs](/docs/questions)
- **[Answers](/answers)**: Responses to questions. AI-drafted, hand-written, or combined. Supports multiple versions per question for A/B testing. Favorite answers feed back into Career Data as writing samples. [Full docs](/docs/answers)
- **[Scrapes](/scrapes)**: Raw webpage captures of job posting URLs. The AI pipeline can auto-scrape from job alert emails. Parsed into structured job post data. [Full docs](/docs/scrapes)

Linked resume resources are NOT editable via chat:
- Experiences, educations, certifications, projects, descriptions, and
  skills live UNDER a resume and are structured records, not free-form
  fields. Editing them from chat would mean reconciling dates, company
  links, and ordering — the resume form UI handles that correctly.
- When the user asks to change something INSIDE a resume ("change my
  title at Robert Half International", "add a bullet to my AWS
  experience", "fix the end date on my last job"), DO NOT attempt to
  use `edit_resume` — that tool only edits the resume's top-level
  label. Instead, navigate the user to the resume page
  (`/resumes/{{id}}`) with a `navigate` action button and tell them to
  edit the experience there.
- If the user says "change my title" on a resume page, "title" could
  mean the resume's own label (top-level `title` field) OR a job
  title inside one of the experiences. ASK WHICH before writing
  anything. Getting this wrong and silently overwriting the wrong
  field is much worse than asking.

Important rules:
- Always call find_job_post_by_link before creating a job post to avoid duplicates
- Use create_job_post_with_company_check (not create_job_post) to handle companies
- NEVER use placeholder names like "unknown", "N/A", or "TBD" as a company name.
  Infer the company from: (1) the recruiter's company, (2) the email sender domain,
  (3) the job posting URL domain. If none work, ask the user
- If a tool returns {{"success": false}}, report the error and stop — do not retry
- Use get_resumes to count or list resumes — do not infer resume counts from career
  data, which may only include favorites
- After any create or update tool call, do NOT echo the full resource content back
  in the chat. Your tool calls trigger a reload signal to the frontend — the user
  will see the updated data on the page. Instead, briefly confirm what you did and
  explain your reasoning (2-3 sentences). This applies to answers, job posts,
  scores, cover letters, and all other resources.

## Scraping URLs
When the user gives you a URL to scrape, call create_scrape(url=..., status="hold").
This creates the scrape with status="hold" so the hold-poller picks it up.
The response includes the scrape ID. After creating:
1. Tell the user the scrape has been queued and link to it: [View scrape](/scrapes/ID)
2. Offer elicitation buttons: "View scrape" (navigates to /scrapes/ID)

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

## Navigation & Action Buttons — call the `propose_actions` tool
When the user wants to GO somewhere, or you have a natural follow-up after
completing an action, DO NOT emit fenced JSON and DO NOT emit HTML comments.
Instead, call the `propose_actions` tool with 1–3 action objects. The
frontend renders each as a clickable button that acts without another LLM
turn (for navigate / model) or triggers a follow-up turn (for message).

Each action sets EXACTLY ONE of these three keys alongside `label`:

- `navigate: "/path"` — route transition, zero LLM cost. USE THIS for
  "View resumes", "Open settings", "See this score", anything whose natural
  outcome is "the user is now on page X".

- `model: {{"type": "...", "id": N, "patch": {{...}}}}` — direct Ember Data
  save, zero LLM cost. USE THIS for "Favorite this resume", "Dismiss
  guidance" (user.onboarding toggle), "Mark reviewed", "Star this answer".
  Allowed `type` + `patch` keys:
    - `resume`: favorite, title, name, notes
    - `cover-letter`: favorite, status
    - `answer`: favorite
    - `job-post`: favorite
    - `user`: onboarding (nest onboarding sub-keys under "onboarding")
  Any other key is silently dropped client-side.

- `message: "follow-up turn"` — sends a new user message, COSTS AN LLM TURN.
  Use only when a button genuinely needs the agent to think again (new
  scope, ambiguous follow-up). Do NOT use this for navigation or state
  changes — the other two shapes are cheaper and instant.

When the user says "take me to", "go to", "navigate to", "open", or "show me"
a page: call `propose_actions` with a single `navigate` action. Do NOT fetch
the resource's data — they want to be taken there, not read about it.

Examples:
- User says "take me to my resumes" →
  call `propose_actions(actions=[{{"label": "Open resumes", "navigate": "/resumes"}}])`
- After creating a job post with id 42 →
  call `propose_actions(actions=[{{"label": "View job post", "navigate": "/job-posts/42"}}, {{"label": "Score it", "message": "score this job post"}}])`
- After surfacing a resume →
  call `propose_actions(actions=[{{"label": "Open resume", "navigate": "/resumes/N"}}, {{"label": "Favorite", "model": {{"type": "resume", "id": N, "patch": {{"favorite": true}}}}}}])`
- User says "stop giving me advice" →
  call `propose_actions(actions=[{{"label": "Turn off wizard", "model": {{"type": "user", "id": ME, "patch": {{"onboarding": {{"wizard_enabled": false}}}}}}}}, {{"label": "Take me to settings", "navigate": "/settings/profile/edit"}}])`

Rules:
- Maximum 3 actions per call.
- Only call `propose_actions` when it makes sense — do NOT bolt buttons
  onto every response.
- Label: short (2–5 words), imperative.
- Prefer `navigate` and `model` over `message`.
- Only fetch data when the user asks a QUESTION about the data (e.g. "how
  many resumes do I have?"). If they just want to GO somewhere, use
  `propose_actions` with a navigate action — do not dump data.

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

## ID types are NOT interchangeable across resources
When the user says something like "take me to that job post" after you reported a
score, a cover letter, or an answer, the id in context belongs to the OTHER
resource — NOT the job post. A Score #38 has its own id (38); the job post it
describes lives at `score.job_post_id`, which is a different number.

Before navigating to a job post on the user's behalf, resolve the id:
- If you have a score id: call `get_scores(id=<score_id>)`, read `job_post_id`
  from the result, then navigate to `/job-posts/<job_post_id>`.
- If you have a cover letter, answer, question, or scrape id: resolve their
  `job_post_id` the same way.
- Never call `get_job_posts(id=<foreign_id>)` and assume a 404 means "deleted"
  — it means you guessed the wrong kind of id.

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

## Bulk-adding Question + Answer Pairs
When the user pastes a list of interview questions with answers (e.g. "add these
Q&A to company X"):
1. Resolve the target scope first — call find_company_by_name or use the
   user's current page context to get a company_id (and/or job_post_id).
2. For EACH Q/A pair, in order:
   a. Call create_question(content="the question text", company_id=<id>) first.
      NEVER call create_answer with a guessed question_id — the IDs from
      get_questions are for existing unrelated questions; using them attaches
      your answer to the wrong question silently (the create succeeds, but the
      answer is orphaned to a stranger's question).
   b. Read the `id` from the create_question response.
   c. Call create_answer(question_id=<new_id>, content="the answer text").
3. If create_question fails, STOP and report the failure — do not proceed to
   create_answer for that pair. Orphan answers (or answers attached to the
   wrong question) are worse than missing data.
4. After the batch, call propose_actions with a single navigate action to the
   company's questions tab (e.g. `/companies/<id>/questions`) so the user can
   review.
4. After creating or updating an answer, do NOT echo the full answer content back
   in the chat. The frontend will reload the data automatically and the user will
   see the answer on the page. Instead, briefly explain your reasoning or approach
   (e.g. "I emphasized your distributed systems experience because the role requires
   it"). Keep it to 2-3 sentences.
5. Call `propose_actions` with a single navigate action pointing at the
   answer: `{{"label": "View answer", "navigate": "/questions/{{question_id}}/answers/{{answer_id}}"}}`.
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
cannot do that directly. Instead, call `propose_actions` with a single
navigate action to /settings/profile:
`{{"label": "Open settings", "navigate": "/settings/profile"}}`
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
        # Always emit every field. Empty fields become "(blank)" so the agent
        # can see exactly what's missing rather than inferring from omitted
        # lines. This is load-bearing for the AW rule about naming the
        # specific missing profile fields.
        def _field(label, value):
            v = (value or "").strip() if isinstance(value, str) else (value or "")
            return f"{label}: {v if v else '(blank)'}"

        parts = [
            _field("First name", attrs.get("first_name")),
            _field("Last name", attrs.get("last_name")),
            _field("Username", attrs.get("username")),
            _field("Email", attrs.get("email")),
            _field("Phone", attrs.get("phone")),
            _field("Address", attrs.get("address")),
            _field("LinkedIn", attrs.get("linkedin")),
            _field("GitHub", attrs.get("github")),
        ]
        if attrs.get("links"):
            parts.append(f"Links: {attrs['links']}")
        logger.info("Resolved user profile: %s", parts[0])
        return "\n".join(parts)


register_defaults()


AGENT_WIZARD_PROMPT_TEMPLATE = """

## Agent Wizard — delegation rule (IMPORTANT)

You have a tool named `ask_onboarding_wizard` that takes one string
argument `user_message`. This tool runs a dedicated onboarding sub-agent
that already has the user's setup state, profile, and page context
loaded — it is ALWAYS the right place to answer onboarding questions.

Triggers — if the user's message matches ANY of these, IMMEDIATELY call
`ask_onboarding_wizard(user_message=<user's exact message>)`:
- asks about onboarding, setup, progress, or "what should I do?"
- greets while any setup step is still pending
- wants to stop / turn off / disable the wizard
- mentions importing, reviewing, or editing a resume as part of setup
- gives any evidence that contradicts a claimed onboarding step

Rules for this tool:
- DO call it in the same turn as the user's message. Do NOT write text
  like "I'll check your onboarding" before calling — just call it. The
  response to the user IS the tool's output.
- DO return the tool's output VERBATIM. Do NOT paraphrase, summarize,
  prepend your own intro, or strip anything. The reply may contain
  navigate markers (`<!-- navigate:/... -->`) the frontend uses for
  routing; stripping them breaks the UX.
- Do NOT answer onboarding questions yourself. Do NOT volunteer setup
  guidance outside the tool. That is the sub-agent's job.
"""



_PROMISE_RE = re.compile(
    r"\b("
    r"i(['’]?\s?)ll\s+(check|look|create|find|fetch|go|see|grab|pull|get|gather|collect|retrieve|prepare)"
    r"|let me\s+(check|look|see|find|pull|grab|fetch|get|retrieve)"
    r"|going to\s+(check|create|look|fetch|find|retrieve)"
    r"|i will\s+(check|create|look|find|fetch|get|retrieve)"
    r")\b",
    re.IGNORECASE,
)

# Tool-call retry policy: if the agent's text reply says "I'll X" but no
# tool call landed this turn, we re-prime it once. Cap at ONE retry so we
# never loop.
_MAX_FOLLOWUP_RETRIES = 1

_FOLLOWUP_PRIMING = (
    "You just told the user you'd perform an action (e.g. \"I'll check...\", "
    "\"I'll create...\", \"Let me look that up\") but did not call any tool. "
    "Either call the correct tool now to fulfill that promise, or tell the "
    "user clearly that you can't and why. Do NOT re-announce the intention — "
    "either ACT or EXPLAIN. Keep the follow-up short."
)


def _sanitize_for_anthropic(messages):
    """Strip tool_use / tool_result / retry parts from pydantic-ai
    ModelMessages before re-feeding as message_history.

    Anthropic's API 400s with 'unexpected tool_use_id in tool_result
    blocks' when a tool_result appears without its paired tool_use in
    an earlier message — which happens when all_messages() from pass 1
    is fed back as history and the adapter serializes a tool_result
    whose tool_use was elided or lived in the same ModelResponse we
    already truncated. Safer: keep only user-text and assistant-text
    parts across retry passes. The retry priming prompt re-establishes
    context in prose.
    """
    if not messages:
        return messages
    safe = []
    for m in messages:
        parts = getattr(m, "parts", None)
        if parts is None:
            safe.append(m)
            continue
        kept = [
            p
            for p in parts
            if isinstance(p, (UserPromptPart, TextPart))
        ]
        if not kept:
            continue
        safe.append(m.__class__(parts=kept))
    return safe


def _response_has_tool_call(messages) -> bool:
    """True if any ToolCallPart appears in the run's messages."""
    for msg in messages:
        for part in getattr(msg, "parts", []):
            if getattr(part, "tool_name", None) is not None:
                return True
    return False


def _is_unfulfilled_promise(full_text: str, messages) -> bool:
    """Pattern match: text sounds like a tool-call promise, but no tool
    was called this turn. Used to decide whether to auto-retry once.

    False-positive risk is acceptable because the retry prompt is safe
    (agent may reasonably decide to explain instead of call); false
    negatives just reproduce today's status quo.
    """
    if _response_has_tool_call(messages):
        return False
    if not full_text:
        return False
    return bool(_PROMISE_RE.search(full_text))


_ONBOARDING_LABELS = {
    "profile_basics": "Fill in name + email",
    "resume_imported": "Import a resume",
    "resume_reviewed": "Review extracted resume fields",
    "first_job_post": "Add a job post to target",
    "first_score": "Score your resume against a job",
    "first_cover_letter": "Generate a cover letter",
}


def _render_onboarding(onboarding: dict) -> tuple[str, str]:
    """Return (onboarding_lines, next_step_label) for prompt injection."""
    wizard_enabled = bool(onboarding.get("wizard_enabled", True))
    lines = [f"- wizard_enabled: {'yes' if wizard_enabled else 'no'}"]
    next_step = "none — all setup complete"
    for key, label in _ONBOARDING_LABELS.items():
        done = bool(onboarding.get(key, False))
        lines.append(f"- {key}: {'yes' if done else 'no'} ({label})")
        if not done and next_step == "none — all setup complete":
            next_step = label
    if not wizard_enabled:
        next_step = "wizard is disabled — do not volunteer onboarding advice"
    return "\n".join(lines), next_step


def _should_inject_aw(onboarding: dict | None) -> bool:
    """Only attach the Agent Wizard prompt block when it can actually fire.

    Skip when wizard is disabled (the block would just tell the agent to
    stay quiet — no need to burn context) and when every step is done
    (nothing to guide). This keeps the main chat prompt focused for the
    common case and lets the AW rules dominate when they matter.
    """
    if not onboarding:
        return False
    if not bool(onboarding.get("wizard_enabled", True)):
        return False
    for key in _ONBOARDING_LABELS:
        if not bool(onboarding.get(key, False)):
            return True
    return False


def _build_system_prompt(
    user_profile: str,
    page_context: dict | None = None,
    onboarding: dict | None = None,
) -> str:
    """Assemble the full system prompt for this turn.

    Split from _build_agent so the prompt logic is testable without
    requiring OpenAI credentials.
    """
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
    if _should_inject_aw(onboarding):
        # The delegation rule has no template placeholders — the sub-agent
        # receives the onboarding snapshot via CareerCaddyDeps, not via the
        # parent's system prompt.
        prompt += AGENT_WIZARD_PROMPT_TEMPLATE
    return prompt


def _build_agent(
    user_profile: str,
    page_context: dict | None = None,
    onboarding: dict | None = None,
    smart: bool = False,
):
    """Build a fresh agent with all career caddy tools.

    When `smart` is True the chat frontend is asking for a stronger model
    on this turn — see the toggle in <Chat::Panel>. The target is read
    from CHAT_SMART_MODEL (default: anthropic:claude-sonnet-4-6) so ops
    can swap without a redeploy.
    """
    prompt = _build_system_prompt(
        user_profile,
        page_context=page_context,
        onboarding=onboarding,
    )
    overrides = {"system_prompt": prompt}
    if smart:
        overrides["model"] = os.environ.get(
            "CHAT_SMART_MODEL", "anthropic:claude-sonnet-4-6"
        )
    return get_agent("chat", **overrides)


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


def _parse_sse_chunk(chunk: str) -> dict | None:
    """Best-effort extraction of the JSON event object from an SSE `data:` line.
    Returns None for keep-alives, comments, or unparseable chunks.
    """
    try:
        payload = chunk.split("data: ", 1)[1].split("\n\n", 1)[0]
        return json.loads(payload)
    except (IndexError, json.JSONDecodeError):
        return None


async def chat(request: Request):
    """POST /chat — AG-UI-protocol streaming chat endpoint.

    Request body (unchanged from legacy protocol):
        {
            "message": "What job posts do I have?",
            "token": "<jwt-or-api-key>",
            "history": [{"role": "user", "content": "..."}, ...],
            "conversation_id": "optional-uuid",
            "page_context": {...},
            "onboarding": {...}
        }

    The token field accepts either a JWT (forwarded by the Django proxy)
    or a jh_* API key (for direct callers). The Django API's auth stack
    handles both transparently.

    Response: text/event-stream of AG-UI protocol events. Standard vocabulary
    (RunStarted, TextMessageStart/Content/End, ToolCallStart/Args/End/Result,
    RunFinished, RunError) plus two CustomEvent extensions:
      - name: "reload"       value: {"resource": "<ember-type>"}
      - name: "session_meta" value: {"conversation_id": "...", "usage": {...}}
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    message = body.get("message", "").strip()
    token = body.get("token", "").strip()
    history = body.get("history", [])
    conversation_id = body.get("conversation_id") or str(uuid.uuid4())
    page_context = body.get("page_context")
    onboarding = body.get("onboarding")
    smart = bool(body.get("smart"))
    if onboarding is not None and not isinstance(onboarding, dict):
        logger.warning("Ignoring non-dict onboarding payload: %r", type(onboarding))
        onboarding = None
    logger.info(
        "page_context received: %s, onboarding keys: %s",
        page_context,
        list(onboarding.keys()) if onboarding else None,
    )

    if not message:
        return JSONResponse({"error": "message is required"}, status_code=400)
    if not token:
        return JSONResponse({"error": "token is required"}, status_code=400)

    async def event_stream():
        encoder = EventEncoder()
        user_profile = await _fetch_user_profile(token)
        agent = _build_agent(
            user_profile,
            page_context=page_context,
            onboarding=onboarding,
            smart=smart,
        )
        if smart:
            logger.info("chat: smart model requested for this turn")
        deps = CareerCaddyDeps(
            api_token=token,
            base_url=API_BASE_URL,
            user_profile=user_profile,
            onboarding=onboarding or {},
            page_context=page_context,
        )

        # Prefix user message with context so the model sees it inline,
        # not buried in the system prompt where gpt-4o-mini ignores it.
        prefix_parts = []
        for line in user_profile.split("\n"):
            if line.startswith("Name: "):
                prefix_parts.append(f"[User: {line[6:].strip()}]")
                break
        if page_context:
            url = page_context.get("url", "")
            if url:
                prefix_parts.append(f"[Current page: {url}]")
        augmented_message = f"{' '.join(prefix_parts)}\n{message}" if prefix_parts else message

        # Convert request history (JSON dicts) to pydantic-ai ModelMessage
        # list so run_ag_ui can pass it through to the agent as
        # `message_history` (AG-UI's own run_input.messages carries only
        # the current-turn prompt).
        messages = []
        for msg in history[-20:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                messages.append(ModelRequest(parts=[UserPromptPart(content=content)]))
            elif role == "assistant":
                messages.append(ModelResponse(parts=[TextPart(content=content)]))

        # Accumulators spanning the initial pass + any follow-up retries.
        reloaded: set[str] = set()
        full_text = ""
        usage_total = {
            "request_tokens": 0,
            "response_tokens": 0,
            "total_tokens": 0,
            "requests": 0,
        }
        # Pydantic-AI ModelMessage transcript (incl. history) — retried
        # passes feed this back in as `message_history` so the model sees
        # its own previous promise.
        last_all_messages: list | None = None
        # This-turn-only messages — used to detect whether a tool fired.
        last_new_messages: list = []

        try:
            async def _run_pass(prompt: str, history_messages: list | None):
                """One agent turn via AG-UI adapter. Tees the protocol stream
                to accumulate text (for the unfulfilled-promise heuristic),
                track tool-call → tool_name, and inject a CustomEvent=reload
                after any ToolCallResult for a reload-mapped tool.
                """
                nonlocal full_text, last_all_messages, last_new_messages
                tool_name_by_id: dict[str, str] = {}

                run_input = RunAgentInput(
                    thread_id=conversation_id,
                    run_id=str(uuid.uuid4()),
                    state={},
                    messages=[UserMessage(id=str(uuid.uuid4()), content=prompt)],
                    tools=[],
                    context=[],
                    forwarded_props={},
                )

                def _on_complete(result):
                    nonlocal last_all_messages, last_new_messages
                    last_all_messages = result.all_messages()
                    if hasattr(result, "new_messages"):
                        last_new_messages = result.new_messages()
                    else:
                        last_new_messages = last_all_messages
                    u = result.usage()
                    usage_total["request_tokens"] += u.request_tokens or 0
                    usage_total["response_tokens"] += u.response_tokens or 0
                    usage_total["total_tokens"] += u.total_tokens or 0
                    usage_total["requests"] += u.requests or 0

                stream = run_ag_ui(
                    agent,
                    run_input,
                    deps=deps,
                    message_history=history_messages,
                    on_complete=_on_complete,
                )

                async for chunk in stream:
                    ev = _parse_sse_chunk(chunk)
                    if ev is not None:
                        et = ev.get("type", "")
                        if et == EventType.TEXT_MESSAGE_CONTENT.value:
                            full_text += ev.get("delta", "")
                        elif et == EventType.TOOL_CALL_START.value:
                            tcid = ev.get("toolCallId", "")
                            tname = ev.get("toolCallName", "")
                            if tcid and tname:
                                tool_name_by_id[tcid] = tname

                    yield chunk

                    if ev is not None and ev.get("type") == EventType.TOOL_CALL_RESULT.value:
                        tcid = ev.get("toolCallId", "")
                        tname = tool_name_by_id.get(tcid, "")
                        if tname in _TOOL_RELOAD_MAP:
                            resource = _TOOL_RELOAD_MAP[tname]
                            if resource not in reloaded:
                                reloaded.add(resource)
                                yield encoder.encode(
                                    CustomEvent(
                                        name="reload",
                                        value={"resource": resource},
                                    )
                                )

            async for event in _run_pass(
                augmented_message, messages if messages else None
            ):
                yield event

            # Follow-up retry when the model emitted an unfulfilled promise
            # ("I'll check...") without actually calling a tool. Capped at
            # _MAX_FOLLOWUP_RETRIES so we never loop.
            retries = 0
            while (
                retries < _MAX_FOLLOWUP_RETRIES
                and _is_unfulfilled_promise(full_text, last_new_messages)
            ):
                retries += 1
                logger.info(
                    "Detected unfulfilled promise (retry %s/%s); re-priming agent",
                    retries, _MAX_FOLLOWUP_RETRIES,
                )
                async for event in _run_pass(
                    _FOLLOWUP_PRIMING,
                    _sanitize_for_anthropic(last_all_messages),
                ):
                    yield event

            logger.info(
                "Chat usage (incl. retries): %s",
                usage_total,
            )

            # Session metadata — carries conversation_id + usage that AG-UI's
            # RunFinishedEvent doesn't have slots for. Frontend reads this
            # after the terminal RunFinished to finalize the message.
            yield encoder.encode(
                CustomEvent(
                    name="session_meta",
                    value={
                        "conversation_id": conversation_id,
                        "usage": dict(usage_total),
                    },
                )
            )

            # pydantic-ai RequestUsage has a `.requests` attr the usage
            # reporter expects; build a lightweight shim.
            from types import SimpleNamespace

            # Track which model actually ran so the spend breakdown
            # attributes tokens correctly when the smart toggle routed
            # the turn to CHAT_SMART_MODEL. trigger='chat_smart' for
            # smart runs so admins can filter by trigger instead of
            # regex'ing model_name.
            effective_model = (
                os.environ.get("CHAT_SMART_MODEL", "anthropic:claude-sonnet-4-6")
                if smart
                else DEFAULT_MODEL
            )
            asyncio.create_task(report_usage(
                api_token=token,
                agent_name="career_caddy_chat",
                model_name=effective_model,
                usage=SimpleNamespace(**usage_total),
                trigger="chat_smart" if smart else "chat",
                base_url=API_BASE_URL,
            ))

        except Exception as e:
            logger.exception("Chat agent error")
            yield encoder.encode(RunErrorEvent(message=str(e)))

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
