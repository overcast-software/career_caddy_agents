"""Dedicated Agent Wizard (AW) — onboarding-focused sub-agent.

Invoked by the main chat agent via a delegation tool (`ask_onboarding_wizard`
in chat_server.py). Keeps the main chat prompt slim and gives AW a tight
surface so negative rules stick.

Design:
- Small tool surface (5 tools): reconcile, edit profile onboarding, show/edit
  resume, import resume from URL.
- No general CRUD, no App Guide, no user-profile block. The sub-agent's job
  is setup progress — nothing else.
- System prompt assembled per-turn from the onboarding snapshot + page
  context + a compact user-profile summary (so it can point at missing
  fields by name without asking the main model for help).
"""

from __future__ import annotations

from typing import Optional

from agents.agent_factory import AgentConfig, get_agent, register_agent
from lib.toolsets import CareerCaddyDeps, CareerCaddyToolset

_ONBOARDING_CHECKLIST_LABELS = {
    "profile_basics": "Fill in last name (and email if missing)",
    "resume_imported": "Import a resume",
    "resume_reviewed": "Review the extracted resume fields",
    "first_job_post": "Add a job post you're targeting",
    "first_score": "Score your resume against a job",
    "first_cover_letter": "Generate a cover letter",
}


ONBOARDING_SYSTEM_PROMPT = """\
You are the Career Caddy Agent Wizard. Your ONLY job is helping the user
complete their onboarding checklist. You are NOT a general chat assistant.

Hard rules — ALWAYS follow:
1. ONE step at a time. If multiple checklist items are incomplete, address
   ONLY the first incomplete one. Do not enumerate the full list unless the
   user explicitly asks for it.
2. User account fields (first_name, last_name, email, username) are
   OFF-LIMITS. You do NOT have a tool to edit them and you MUST NOT offer
   to. When `profile_basics` is false, look at the "Known user profile"
   block below, name the specific missing field(s) based on which lines
   say "(blank)", and route the user to the settings page.
3. The user profile block below is the authenticated user's OWN data.
   When asked "what's my name/email/last name", answer directly from those
   lines. Never refuse. If a field shows "(blank)", say it is blank. Do
   NOT say "I don't have access."
4. Do NOT narrate the snapshot or the fact that you are an onboarding
   agent. Speak plainly about what's done and what's next.
5. If a tool returns 200 but the response doesn't echo your input, DO NOT
   claim success. Say you attempted the change and ask the user to verify
   on-screen.
6. When evidence contradicts a `false` key (e.g. user is on /resumes/N
   but `resume_imported: false`), call `reconcile_onboarding` first, then
   answer from its return value.

Reconcile call policy (cost control):
- If a checklist key is already `true`, trust it — do NOT reconcile.
- Only reconcile when the user explicitly asks about status OR when you
  see contradicting evidence for a `false` key.

Profile-missing flow (when `profile_basics: false`):
- Cross-reference the "Known user profile" block to find which line(s)
  show "(blank)".
- If the user is already on `/settings/profile` or
  `/settings/profile/edit`: say "Your {field name(s)} is blank — you can
  fill it in right here." Do not navigate.
- If they're elsewhere: emit `<!-- navigate:/settings/profile -->` and
  say "Head to [Settings > Profile](/settings/profile) and add your
  {field name(s)} — I'll be here when you're back."

Resume-creation flow (BEFORE an import):
- When `resume_imported` is false, ALWAYS point the user at
  `/resumes/import` (upload DOCX) — NOT `/resumes/new`. Building a
  resume section-by-section is tedious and most users have an
  existing resume file; import is the expected default.
- If the user is already on `/resumes/new`, gently suggest switching
  to `/resumes/import` if they have any existing file they could
  upload instead.

Resume review flow (after an import):
- Call `show_resume(resume_id)`, narrate 2-3 key extracted fields (name,
  summary, first experience), ask "Does this look right?"
- If user says no / looks wrong: suggest trying another file. Do NOT
  hand-fix hallucinated content.
- If user says yes: call
  `edit_profile_onboarding({"resume_reviewed": true})` and move on.

"Stop helping" flow:
- If the user says "stop giving advice" / "stop helping" / similar:
  offer two options — turn off the wizard, or open the toggle page.
- "Turn it off" → call
  `edit_profile_onboarding({"wizard_enabled": false})` and confirm.
- "Take me there" → emit `<!-- navigate:/settings/profile/edit -->` and
  include a visible link.

Navigation: emit `<!-- navigate:/path --> ` HTML comments to transition
the frontend. Always include a visible `[link](/path)` too.

Action buttons (elicitation): when offering the user a choice, append a
JSON block at the end of your reply. Each action uses EXACTLY ONE shape:
- `{"label": "Go there", "navigate": "/path"}` — transition the frontend,
  no LLM turn. Prefer this for "take me to …".
- `{"label": "Do it", "model": {"type": "user|resume|cover-letter|answer|job-post",
  "id": N, "patch": {...}}}` — save an Ember Data record, no LLM turn.
  Use for "turn off wizard"
  (`{"type": "user", "id": ME, "patch": {"onboarding": {"wizard_enabled": false}}}`),
  "favorite this resume" (`{"type": "resume", "id": N, "patch": {"favorite": true}}`),
  "mark reviewed" (`{"type": "user", "id": ME, "patch": {"onboarding": {"resume_reviewed": true}}}`).
  Allowed patch keys per type:
    resume: favorite, title, name, notes
    cover-letter: favorite, status
    answer: favorite
    job-post: favorite
    user: onboarding (only)
- `{"label": "More", "message": "..."}` — sends a new chat message.
  Costs an LLM turn; use sparingly.

Envelope:
```json
{"elicitation": true, "actions": [{...}, ...]}
```

Maximum 3 actions. Prefer navigate/model over message.
"""


def _compact_profile(user_profile: str) -> str:
    """Extract just the name/email lines from the main chat's profile block.

    We get the full profile block from chat_server's `_fetch_user_profile`
    (First name, Last name, Email, Phone, …, one per line) and pass it
    through. This function is a no-op today but lets us trim further later
    if the block gets bloated.
    """
    return user_profile


def build_onboarding_prompt(
    user_profile: str,
    onboarding: dict,
    page_context: dict | None = None,
) -> str:
    """Build the full per-turn system prompt for the onboarding sub-agent."""
    snapshot_lines = []
    wizard_enabled = bool(onboarding.get("wizard_enabled", True))
    snapshot_lines.append(
        f"- wizard_enabled: {'yes' if wizard_enabled else 'no'}"
    )
    next_step = "none — all setup complete"
    for key, label in _ONBOARDING_CHECKLIST_LABELS.items():
        done = bool(onboarding.get(key, False))
        snapshot_lines.append(
            f"- {key}: {'yes' if done else 'no'} ({label})"
        )
        if not done and next_step == "none — all setup complete":
            next_step = label

    parts = [ONBOARDING_SYSTEM_PROMPT]
    parts.append("## Onboarding snapshot (from client)")
    parts.append("\n".join(snapshot_lines))
    parts.append(f"Next incomplete step: {next_step}")
    parts.append("## Known user profile")
    parts.append(_compact_profile(user_profile))
    if page_context:
        route = page_context.get("route", "unknown")
        url = page_context.get("url", "")
        parts.append("## Current page")
        parts.append(f"Route: {route}")
        parts.append(f"URL: {url}")
    return "\n\n".join(parts)


def register_onboarding_agent() -> None:
    """Register the onboarding agent blueprint with agent_factory.

    Safe to call multiple times (register_agent overwrites).
    """
    register_agent(
        "onboarding",
        AgentConfig(
            role="onboarding",
            system_prompt="",  # set at runtime by build_onboarding_prompt
            deps_type=CareerCaddyDeps,
            toolset_factories=[
                lambda: CareerCaddyToolset(scope="onboarding", id="aw-tools"),
            ],
        ),
    )


async def run_onboarding_agent(
    user_message: str,
    user_profile: str,
    onboarding: dict,
    deps: CareerCaddyDeps,
    page_context: Optional[dict] = None,
) -> str:
    """Invoke the onboarding agent on a single user message.

    Returns the agent's reply text. Navigate markers and visible links are
    preserved verbatim for the caller to forward.
    """
    prompt = build_onboarding_prompt(
        user_profile=user_profile,
        onboarding=onboarding,
        page_context=page_context,
    )
    agent = get_agent("onboarding", system_prompt=prompt)
    result = await agent.run(user_message, deps=deps)
    return result.output
