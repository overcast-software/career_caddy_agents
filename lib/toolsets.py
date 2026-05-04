"""
Pydantic-AI toolsets for Career Caddy agents.

Provides CareerCaddyToolset — an in-process toolset that wraps lib/api_tools.py
functions. Replaces MCPServerStdio subprocess spawning for pydantic-ai agents.

Security: This module MUST NOT import from lib/browser/, mcp_servers/browser_server,
mcp_servers/email_server, or anything that touches secrets.yml / notmuch.
"""

import inspect
import functools
from dataclasses import dataclass
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.toolsets.function import FunctionToolset

from lib.api_tools import ApiClient
from lib import api_tools


# ---------------------------------------------------------------------------
# Deps
# ---------------------------------------------------------------------------


@dataclass
class CareerCaddyDeps:
    """Dependencies for Career Caddy toolsets. Passed to Agent.run(deps=...)."""

    api_token: str
    base_url: str = "http://localhost:8000"
    # Per-turn context threaded through to delegation tools (e.g.
    # ask_onboarding_wizard) so the sub-agent can build its own system
    # prompt without re-fetching /me/ or re-reading the request body.
    user_profile: str = ""
    onboarding: dict | None = None
    page_context: dict | None = None


# ---------------------------------------------------------------------------
# Tool registry — maps tool names to api_tools functions
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, Any] = {
    # Companies
    "create_company": api_tools.create_company,
    "find_company_by_name": api_tools.find_company_by_name,
    "search_companies": api_tools.search_companies,
    "get_companies": api_tools.get_companies,
    # Job posts
    "create_job_post_with_company_check": api_tools.create_job_post_with_company_check,
    "find_job_post_by_link": api_tools.find_job_post_by_link,
    "search_job_posts": api_tools.search_job_posts,
    "get_job_posts": api_tools.get_job_posts,
    "update_job_post": api_tools.update_job_post,
    # Job applications
    "create_job_application": api_tools.create_job_application,
    "get_job_applications": api_tools.get_job_applications,
    "get_applications_for_job_post": api_tools.get_applications_for_job_post,
    "update_job_application": api_tools.update_job_application,
    # Career data
    "get_career_data": api_tools.get_career_data,
    "get_current_user": api_tools.get_current_user,
    # Resumes
    "get_resumes": api_tools.get_resumes,
    # Questions & Answers
    "get_questions": api_tools.get_questions,
    "create_question": api_tools.create_question,
    "get_answers": api_tools.get_answers,
    "create_answer": api_tools.create_answer,
    "update_answer": api_tools.update_answer,
    # Scrapes
    "create_scrape": api_tools.create_scrape,
    "get_scrapes": api_tools.get_scrapes,
    "update_scrape": api_tools.update_scrape,
    # Scores
    "score_job_post": api_tools.score_job_post,
    "get_scores": api_tools.get_scores,
    # Composite
    "scrape_and_score": api_tools.scrape_and_score,
    # Agent Wizard
    "show_resume": api_tools.show_resume,
    "edit_resume": api_tools.edit_resume,
    "show_cover_letter": api_tools.show_cover_letter,
    "edit_cover_letter": api_tools.edit_cover_letter,
    "import_resume_from_url": api_tools.import_resume_from_url,
    "edit_profile_onboarding": api_tools.edit_profile_onboarding,
    "reconcile_onboarding": api_tools.reconcile_onboarding,
}


# ---------------------------------------------------------------------------
# Named scopes — subsets of tools for different agent roles
# ---------------------------------------------------------------------------

SCOPES: dict[str, set[str]] = {
    "all": set(TOOL_REGISTRY.keys()),
    "career_caddy": {
        "create_company", "find_company_by_name", "search_companies", "get_companies",
        "create_job_post_with_company_check", "find_job_post_by_link",
        "search_job_posts", "get_job_posts", "update_job_post",
        "create_job_application", "get_job_applications",
        "get_applications_for_job_post", "update_job_application",
        "get_career_data",
        "get_current_user",
        "get_resumes",
        "get_questions",
        "create_question",
        "get_answers",
        "create_answer",
        "update_answer",
        # Agent Wizard
        "show_resume", "edit_resume",
        "show_cover_letter", "edit_cover_letter",
        "import_resume_from_url",
        "edit_profile_onboarding",
        "reconcile_onboarding",
    },
    "job_discovery": {
        "find_company_by_name", "search_companies", "get_companies",
        "create_company", "create_job_post_with_company_check",
        "find_job_post_by_link", "search_job_posts",
    },
    "scoring": {
        "get_job_posts", "get_career_data", "score_job_post", "get_scores",
        "scrape_and_score",
    },
    "application_tracking": {
        "create_job_application", "get_job_applications",
        "get_applications_for_job_post", "update_job_application",
        "update_job_post", "find_job_post_by_link",
    },
    "scrape_management": {
        "create_scrape", "get_scrapes", "update_scrape",
    },
    # Dedicated Agent Wizard scope — kept deliberately small so the sub-agent
    # has a focused surface and negative rules aren't drowned out by general
    # CRUD. Excludes cover-letter tools (not part of onboarding flow).
    "onboarding": {
        "reconcile_onboarding",
        "edit_profile_onboarding",
        "show_resume",
        "edit_resume",
        "import_resume_from_url",
    },
    # Main chat's toolset — EVERYTHING except the onboarding-only tools.
    # Those tools reach the chat only via agent delegation
    # (ask_onboarding_wizard → sub-agent). Giving the main agent direct
    # access would let it answer onboarding questions without delegating,
    # which defeats the sub-agent's focused-prompt advantage.
    "main_chat": set(),  # populated below from TOOL_REGISTRY
}

# "main_chat" = all tools minus the onboarding-only surface.
SCOPES["main_chat"] = set(TOOL_REGISTRY.keys()) - {
    "reconcile_onboarding",
    "edit_profile_onboarding",
    "import_resume_from_url",
}


# ---------------------------------------------------------------------------
# Wrapper builder
# ---------------------------------------------------------------------------


def _make_tool_wrapper(fn):
    """Build a RunContext-aware wrapper for an api_tools function.

    Takes a function like `async def foo(api: ApiClient, x: int) -> str`
    and returns `async def foo(ctx: RunContext[CareerCaddyDeps], x: int) -> str`
    that builds the ApiClient from ctx.deps.
    """
    sig = inspect.signature(fn)
    original_params = list(sig.parameters.values())
    # Drop the first param (api: ApiClient)
    tool_params = original_params[1:]

    @functools.wraps(fn)
    async def wrapper(ctx: RunContext[CareerCaddyDeps], **kwargs):
        api = ApiClient(ctx.deps.base_url, ctx.deps.api_token)
        return await fn(api, **kwargs)

    # Rebuild signature: ctx first, then original params (minus api)
    ctx_param = inspect.Parameter(
        "ctx",
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        annotation=RunContext[CareerCaddyDeps],
    )
    wrapper.__signature__ = sig.replace(parameters=[ctx_param] + tool_params)

    # Rebuild annotations for pydantic-ai schema generation
    annotations = {"ctx": RunContext[CareerCaddyDeps]}
    for p in tool_params:
        if p.annotation != inspect.Parameter.empty:
            annotations[p.name] = p.annotation
    wrapper.__annotations__ = annotations

    return wrapper


# ---------------------------------------------------------------------------
# CareerCaddyToolset
# ---------------------------------------------------------------------------


def CareerCaddyToolset(
    scope: str | list[str] = "all",
    *,
    id: str | None = "career-caddy",
) -> FunctionToolset[CareerCaddyDeps]:
    """Build a scoped FunctionToolset from lib/api_tools functions.

    Args:
        scope: A named scope (e.g. "scoring", "job_discovery") or a list
               of specific tool names.
        id: Toolset ID for pydantic-ai (must be unique per agent).

    Returns:
        A FunctionToolset ready to pass to Agent(toolsets=[...]).
    """
    if isinstance(scope, str):
        tool_names = SCOPES[scope]
    else:
        tool_names = set(scope)

    toolset: FunctionToolset[CareerCaddyDeps] = FunctionToolset(id=id)

    for name in sorted(tool_names):
        fn = TOOL_REGISTRY[name]
        toolset.add_function(
            _make_tool_wrapper(fn),
            takes_ctx=True,
            name=name,
        )

    return toolset


# ---------------------------------------------------------------------------
# Convenience factories
# ---------------------------------------------------------------------------


def job_discovery_toolset(**kwargs) -> FunctionToolset[CareerCaddyDeps]:
    return CareerCaddyToolset(scope="job_discovery", **kwargs)


def scoring_toolset(**kwargs) -> FunctionToolset[CareerCaddyDeps]:
    return CareerCaddyToolset(scope="scoring", **kwargs)


def application_tracking_toolset(**kwargs) -> FunctionToolset[CareerCaddyDeps]:
    return CareerCaddyToolset(scope="application_tracking", **kwargs)


def scrape_management_toolset(**kwargs) -> FunctionToolset[CareerCaddyDeps]:
    return CareerCaddyToolset(scope="scrape_management", **kwargs)


def onboarding_delegation_toolset(
    *, id: str = "aw-delegation",
) -> FunctionToolset[CareerCaddyDeps]:
    """A one-tool toolset the main chat agent uses to delegate onboarding
    turns to the dedicated Agent Wizard sub-agent (see
    agents/onboarding_agent.py).

    Kept in its own toolset (rather than crammed into TOOL_REGISTRY) because
    `ask_onboarding_wizard` doesn't hit the API — it invokes a nested
    pydantic-ai Agent. Mixing it into the HTTP-tool registry would muddy
    the mental model.
    """
    # Deferred import to avoid circular imports at module load time.
    from agents.onboarding_agent import run_onboarding_agent

    async def ask_onboarding_wizard(
        ctx: RunContext[CareerCaddyDeps], user_message: str
    ) -> str:
        """Delegate an onboarding-related user message to the Agent Wizard
        sub-agent. Pass the user's message unchanged; the sub-agent's reply
        is returned VERBATIM (do not paraphrase — it may contain navigate
        markers like <!-- navigate:/settings/profile -->).

        Use this when the user asks about their onboarding / setup /
        progress / what to do, OR when you notice evidence that a checklist
        step hasn't been done. Do NOT answer onboarding questions yourself.
        """
        deps = ctx.deps
        onboarding = deps.onboarding or {}
        return await run_onboarding_agent(
            user_message=user_message,
            user_profile=deps.user_profile,
            onboarding=onboarding,
            deps=deps,
            page_context=deps.page_context,
        )

    toolset: FunctionToolset[CareerCaddyDeps] = FunctionToolset(id=id)
    toolset.add_function(
        ask_onboarding_wizard,
        takes_ctx=True,
        name="ask_onboarding_wizard",
    )
    return toolset
