"""Tests for answer-route page-context injection in the main chat prompt.

When the user is on /job-posts/:jp/questions/:q/answers/:a, the prompt
must expose all three IDs so the agent can target the correct question
when creating a variant answer. The generic first-number fallback would
misdirect the agent at job_post_id.
"""

from mcp_servers.chat_server import _build_system_prompt


PROFILE = "First name: Jane\nLast name: Doe\nEmail: jane@example.com"

# Real NanoID shapes (CC-77). Ids are 10-char strings, not integers — the
# page-context regexes anchor to [A-Za-z0-9_-]{10}, so numeric fixtures
# (the old "42"/"7"/"19"/"5") no longer match. Using a real shape keeps
# these tests honest instead of false-greening on a broken \d+ pattern.
JP_ID = "V1StGXR8_Z"
Q_ID = "a1fFQQe1xV"
A_ID = "Ksd7bRkHpL"
RESUME_ID = "Op9NmLkJh2"


class TestAnswerRouteContext:
    def test_answer_show_exposes_all_three_ids(self):
        prompt = _build_system_prompt(
            PROFILE,
            page_context={
                "route": "job-posts.show.questions.show.answers.show",
                "url": f"/job-posts/{JP_ID}/questions/{Q_ID}/answers/{A_ID}",
            },
        )
        assert f"Job Post ID: {JP_ID}" in prompt
        assert f"Question ID: {Q_ID}" in prompt
        assert f"Answer ID: {A_ID}" in prompt
        # Legacy generic "Resource ID" should NOT duplicate when we've
        # already emitted specific IDs.
        assert "Resource ID:" not in prompt
        # The "call the matching tool with id=…" suffix should point at
        # the most specific ID — the answer.
        assert f"with id={A_ID}" in prompt

    def test_question_show_without_answer_exposes_question_id(self):
        prompt = _build_system_prompt(
            PROFILE,
            page_context={
                "route": "job-posts.show.questions.show",
                "url": f"/job-posts/{JP_ID}/questions/{Q_ID}",
            },
        )
        assert f"Job Post ID: {JP_ID}" in prompt
        assert f"Question ID: {Q_ID}" in prompt
        assert "Answer ID:" not in prompt
        assert f"with id={Q_ID}" in prompt

    def test_non_nested_route_falls_back_to_generic_resource_id(self):
        prompt = _build_system_prompt(
            PROFILE,
            page_context={
                "route": "resumes.show",
                "url": f"/resumes/{RESUME_ID}",
            },
        )
        # Resumes aren't job-posts/questions/answers — use the generic
        # Resource ID slot so the rest of the prompt still works.
        assert f"Resource ID: {RESUME_ID}" in prompt
        assert f"with id={RESUME_ID}" in prompt


class TestAnswerModificationGuidance:
    def test_prompt_includes_modifying_existing_answer_section(self):
        """The answer-tweak behavior lives in the static SYSTEM_PROMPT,
        so every build should carry it regardless of page context."""
        prompt = _build_system_prompt(PROFILE)
        assert "Modifying an Existing Answer" in prompt
        # Defaults: prefer create, offer replace via propose_actions.
        # Whitespace-tolerant — the rule is about CREATE-as-default, not
        # exact line wrap.
        normalized = " ".join(prompt.split())
        assert "DEFAULT to `create_answer`" in normalized
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
