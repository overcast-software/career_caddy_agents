"""Tests for the chat-server follow-up retry helpers.

When a model emits an 'I'll check X' promise but doesn't actually call any
tool, chat_server retries the turn once with a priming message. This file
covers the detector (regex + tool-call inspection) in isolation — the full
SSE streaming path is exercised via the live walkthrough.
"""

from mcp_servers.chat_server import (
    _is_unfulfilled_promise,
    _response_has_tool_call,
)


class _FakeToolCallPart:
    def __init__(self, tool_name):
        self.tool_name = tool_name


class _FakeTextPart:
    def __init__(self, content):
        self.content = content

    # no tool_name attribute — exercises the fallback
    def __getattr__(self, name):
        if name == "tool_name":
            return None
        raise AttributeError(name)


class _FakeMessage:
    def __init__(self, parts):
        self.parts = parts


class TestResponseHasToolCall:
    def test_false_when_no_messages(self):
        assert _response_has_tool_call([]) is False

    def test_false_when_only_text_parts(self):
        msgs = [_FakeMessage([_FakeTextPart("hello")])]
        assert _response_has_tool_call(msgs) is False

    def test_true_when_any_part_has_tool_name(self):
        msgs = [
            _FakeMessage([_FakeTextPart("about to")]),
            _FakeMessage([_FakeToolCallPart("get_job_posts")]),
        ]
        assert _response_has_tool_call(msgs) is True


class TestIsUnfulfilledPromise:
    def test_false_when_empty_text(self):
        assert _is_unfulfilled_promise("", []) is False

    def test_false_when_tool_was_called(self):
        """Even with promise phrasing, if a tool actually fired, nothing
        to retry."""
        msgs = [_FakeMessage([_FakeToolCallPart("get_resumes")])]
        assert _is_unfulfilled_promise("I'll check your resumes.", msgs) is False

    def test_true_for_ill_check(self):
        assert _is_unfulfilled_promise("I'll check that for you.", []) is True

    def test_true_for_let_me_check(self):
        assert _is_unfulfilled_promise("Let me check your progress.", []) is True

    def test_true_for_going_to_create(self):
        assert _is_unfulfilled_promise(
            "I'm going to create a job post now.", []
        ) is True

    def test_true_for_ill_look_up(self):
        assert _is_unfulfilled_promise("I'll look up what you have.", []) is True

    def test_true_for_i_will_fetch(self):
        assert _is_unfulfilled_promise("I will fetch the data.", []) is True

    def test_true_for_curly_apostrophe(self):
        """Real LLM output often uses curly apostrophes (U+2019). Don't
        miss a retry just because of Unicode niceties."""
        assert _is_unfulfilled_promise("I\u2019ll check that.", []) is True

    def test_false_for_generic_phrases(self):
        """Regular answers that aren't tool-promise-shaped."""
        assert _is_unfulfilled_promise(
            "Your last name is blank.", []
        ) is False
        assert _is_unfulfilled_promise(
            "You're all set up — nothing to do.", []
        ) is False

    def test_false_when_tool_was_called_even_with_curly_quote(self):
        msgs = [_FakeMessage([_FakeToolCallPart("get_career_data")])]
        assert _is_unfulfilled_promise(
            "I\u2019ll check your career data.", msgs
        ) is False
