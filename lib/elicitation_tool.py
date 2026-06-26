"""
propose_actions tool — structured elicitation for chat buttons.

Phase 1 of the AG-UI migration. This tool carries the button payload via
its call args; under AG-UI the frontend reads ToolCallArgs events to render
buttons instead of regex-parsing fenced JSON from the assistant's text.

The tool itself has no side effects — it validates the shape and returns
{"ok": True}. The "work" is the pydantic schema the agent sees and the
args the frontend consumes.

Not yet wired into chat_server.py or the frontend — that happens in
phases 2 and 3 of the AG-UI migration.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator
from pydantic_ai import RunContext
from pydantic_ai.toolsets.function import FunctionToolset

from lib.toolsets import CareerCaddyDeps


MAX_ACTIONS = 3

# Per-type fields the agent is allowed to patch via a {model} action. Mirrors
# the frontend's ALLOWED_ACTION_PATCHES in chat/message.js — kept in sync
# manually. The frontend is the authority at save time; this list is here so
# the agent can see it in the tool schema description and pick valid keys.
ALLOWED_MODEL_PATCH_KEYS: dict[str, list[str]] = {
    "resume": ["favorite", "title", "name", "notes"],
    "cover-letter": ["favorite", "status"],
    "answer": ["favorite"],
    "job-post": ["favorite"],
    # user.onboarding is nested — the patch value is itself a dict under "onboarding"
    "user": ["onboarding"],
}


class ModelActionTarget(BaseModel):
    """Target of a {model} action — an Ember Data record + patch.

    Allowed type/key pairs (frontend enforces; agent should stay within):
        resume:       favorite, title, name, notes
        cover-letter: favorite, status
        answer:       favorite
        job-post:     favorite
        user:         onboarding (nested dict of onboarding sub-keys)
    Any other key is silently dropped client-side.
    """

    type: str = Field(description="Ember model type, e.g. 'resume', 'cover-letter', 'answer', 'job-post', 'user'")
    # Ember Data record ids are strings on the wire. Since CC-77 the
    # NanoID-keyed models (resume/cover-letter/answer/job-post) carry opaque
    # string ids; the user record stays numeric but serializes fine as a
    # string. A plain `int` here let the MCP client strip a NanoID before the
    # action reached the frontend.
    id: str = Field(description="Record id")
    patch: dict[str, Any] = Field(
        description="Fields to set on the record. See ALLOWED_MODEL_PATCH_KEYS in the tool description."
    )


class ElicitationAction(BaseModel):
    """One button in a propose_actions payload.

    Set EXACTLY ONE of navigate, model, or message:
      - navigate: "/path" — route transition, zero LLM cost. Prefer for "go
        to X" buttons.
      - model: {type, id, patch} — direct Ember Data save, zero LLM cost.
        Prefer for "favorite this", "mark reviewed", "toggle setting".
      - message: "follow-up user turn" — costs an LLM turn. Use only when a
        button genuinely needs another agent response.
    """

    label: str = Field(description="Button text — 2-5 words, imperative.")
    navigate: Optional[str] = Field(
        default=None, description="Absolute path like '/resumes/42' — no LLM turn."
    )
    model: Optional[ModelActionTarget] = Field(
        default=None, description="Ember Data save target — no LLM turn."
    )
    message: Optional[str] = Field(
        default=None, description="User message to send as a new chat turn — COSTS AN LLM TURN."
    )

    @model_validator(mode="after")
    def _exactly_one_shape(self) -> "ElicitationAction":
        count = sum(
            x is not None
            for x in (self.navigate, self.model, self.message)
        )
        if count != 1:
            raise ValueError(
                "action must set exactly one of: navigate, model, message"
            )
        return self


async def propose_actions(
    ctx: RunContext[CareerCaddyDeps],
    actions: list[ElicitationAction],
) -> dict:
    """Offer the user 1-3 quick-action buttons beneath your response.

    Call this INSTEAD OF emitting a fenced JSON elicitation block. The
    frontend reads the call args and renders buttons that act directly
    (navigate / model save / send follow-up) without regex-parsing text.

    Rules:
    - Maximum 3 actions.
    - Only offer buttons that make sense in context — do not bolt them
      onto every response.
    - Prefer 'navigate' and 'model' shapes over 'message' (the first two
      cost zero LLM turns).
    """
    if not actions:
        return {"ok": False, "error": "at least one action required"}
    if len(actions) > MAX_ACTIONS:
        return {
            "ok": False,
            "error": f"at most {MAX_ACTIONS} actions (got {len(actions)})",
        }
    return {"ok": True, "count": len(actions)}


def elicitation_toolset(
    *, id: str = "elicitation",
) -> FunctionToolset[CareerCaddyDeps]:
    """FunctionToolset exposing just propose_actions. Add alongside
    CareerCaddyToolset on the chat Agent once the AG-UI swap lands."""
    toolset: FunctionToolset[CareerCaddyDeps] = FunctionToolset(id=id)
    toolset.add_function(
        propose_actions,
        takes_ctx=True,
        name="propose_actions",
    )
    return toolset
