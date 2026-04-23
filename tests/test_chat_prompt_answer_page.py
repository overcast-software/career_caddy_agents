"""Tests for answer-route page-context injection in the main chat prompt.

When the user is on /job-posts/:jp/questions/:q/answers/:a, the prompt
must expose all three IDs so the agent can target the correct question
when creating a variant answer. The generic first-number fallback would
misdirect the agent at job_post_id.
"""

from mcp_servers.chat_server import _build_system_prompt


PROFILE = "First name: Jane\nLast name: Doe\nEmail: jane@example.com"


class TestAnswerRouteContext:
    def test_answer_show_exposes_all_three_ids(self):
        prompt = _build_system_prompt(
            PROFILE,
            page_context={
                "route": "job-posts.show.questions.show.answers.show",
                "url": "/job-posts/42/questions/7/answers/19",
            },
        )
        assert "Job Post ID: 42" in prompt
        assert "Question ID: 7" in prompt
        assert "Answer ID: 19" in prompt
        # Legacy generic "Resource ID" should NOT duplicate when we've
        # already emitted specific IDs.
        assert "Resource ID:" not in prompt
        # The "call the matching tool with id=…" suffix should point at
        # the most specific ID — the answer.
        assert "with id=19" in prompt

    def test_question_show_without_answer_exposes_question_id(self):
        prompt = _build_system_prompt(
            PROFILE,
            page_context={
                "route": "job-posts.show.questions.show",
                "url": "/job-posts/42/questions/7",
            },
        )
        assert "Job Post ID: 42" in prompt
        assert "Question ID: 7" in prompt
        assert "Answer ID:" not in prompt
        assert "with id=7" in prompt

    def test_non_nested_route_falls_back_to_generic_resource_id(self):
        prompt = _build_system_prompt(
            PROFILE,
            page_context={
                "route": "resumes.show",
                "url": "/resumes/5",
            },
        )
        # Resumes aren't job-posts/questions/answers — use the generic
        # Resource ID slot so the rest of the prompt still works.
        assert "Resource ID: 5" in prompt
        assert "with id=5" in prompt


class TestAnswerModificationGuidance:
    def test_prompt_includes_modifying_existing_answer_section(self):
        """The answer-tweak behavior lives in the static SYSTEM_PROMPT,
        so every build should carry it regardless of page context."""
        prompt = _build_system_prompt(PROFILE)
        assert "Modifying an Existing Answer" in prompt
        # Defaults: prefer create, offer replace via propose_actions.
        assert "default to CREATE" in prompt or "default to CREATE a new answer" in prompt
        assert "Replace original instead" in prompt


class TestHostnameLinkBan:
    """Regression test for the example.com/... link bug: the agent was
    emitting links like `example.com/job-posts/1` (hostname-prefixed bare
    path) which the SPA router can't follow. The prompt must explicitly
    ban any scheme/host prefix on navigation targets."""

    def test_prompt_forbids_hostname_prefixed_paths(self):
        prompt = _build_system_prompt(PROFILE)
        assert "example.com/job-posts/1" in prompt  # negative example
        assert "scheme, hostname, or domain" in prompt
        assert "bare paths" in prompt
