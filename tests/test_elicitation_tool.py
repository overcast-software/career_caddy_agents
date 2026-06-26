"""Tests for lib.elicitation_tool — schema validation and the stub tool body.

The tool itself is a no-op; its value is in the pydantic schema the agent
sees and the validator semantics enforced on the args. These tests lock
both so the AG-UI wiring in phase 2 has a stable contract.
"""

import pytest
from pydantic import ValidationError

from lib.elicitation_tool import (
    ALLOWED_MODEL_PATCH_KEYS,
    MAX_ACTIONS,
    ElicitationAction,
    ModelActionTarget,
    elicitation_toolset,
    propose_actions,
)


# ---------------------------------------------------------------------------
# ElicitationAction schema — exactly-one-of navigate/model/message
# ---------------------------------------------------------------------------


class TestElicitationActionSchema:
    def test_navigate_shape_valid(self):
        a = ElicitationAction(label="View resume", navigate="/resumes/42")
        assert a.navigate == "/resumes/42"
        assert a.model is None
        assert a.message is None

    def test_model_shape_valid(self):
        # Record ids are NanoID strings since CC-77 (was int) — see CC-101.
        a = ElicitationAction(
            label="Favorite",
            model={"type": "resume", "id": "Rs7mAaaaaa", "patch": {"favorite": True}},
        )
        assert a.model is not None
        assert a.model.type == "resume"
        assert a.model.id == "Rs7mAaaaaa"
        assert a.model.patch == {"favorite": True}

    def test_message_shape_valid(self):
        a = ElicitationAction(label="Tell me more", message="what should I do next?")
        assert a.message == "what should I do next?"

    def test_rejects_no_shape(self):
        with pytest.raises(ValidationError) as exc:
            ElicitationAction(label="Click me")
        assert "exactly one" in str(exc.value)

    def test_rejects_two_shapes(self):
        with pytest.raises(ValidationError) as exc:
            ElicitationAction(
                label="Bad", navigate="/x", message="also no"
            )
        assert "exactly one" in str(exc.value)

    def test_rejects_all_three_shapes(self):
        with pytest.raises(ValidationError):
            ElicitationAction(
                label="Very bad",
                navigate="/x",
                message="y",
                model={"type": "resume", "id": "1", "patch": {"favorite": True}},
            )

    def test_label_required(self):
        with pytest.raises(ValidationError):
            ElicitationAction(navigate="/x")


class TestModelActionTarget:
    def test_all_fields_required(self):
        with pytest.raises(ValidationError):
            ModelActionTarget(type="resume", id="1")  # missing patch
        with pytest.raises(ValidationError):
            ModelActionTarget(type="resume", patch={})  # missing id
        with pytest.raises(ValidationError):
            ModelActionTarget(id="1", patch={})  # missing type

    def test_patch_accepts_arbitrary_keys(self):
        """The tool side does NOT enforce the allow-list — the frontend does.
        The agent picks valid keys because the tool's docstring lists them."""
        target = ModelActionTarget(
            type="user",
            id="1",
            patch={"onboarding": {"wizard_enabled": False}},
        )
        assert target.patch == {"onboarding": {"wizard_enabled": False}}


# ---------------------------------------------------------------------------
# Allow-list reference — keeps frontend + backend in sync
# ---------------------------------------------------------------------------


class TestAllowListReference:
    def test_covers_all_five_types(self):
        assert set(ALLOWED_MODEL_PATCH_KEYS.keys()) == {
            "resume", "cover-letter", "answer", "job-post", "user",
        }

    def test_resume_has_expected_keys(self):
        assert set(ALLOWED_MODEL_PATCH_KEYS["resume"]) == {
            "favorite", "title", "name", "notes",
        }

    def test_user_only_onboarding(self):
        # User action is intentionally narrow — staff flags, email, etc. must
        # never be agent-patchable. Kept to onboarding JSONB only.
        assert ALLOWED_MODEL_PATCH_KEYS["user"] == ["onboarding"]


# ---------------------------------------------------------------------------
# propose_actions tool body
# ---------------------------------------------------------------------------


class TestProposeActions:
    @pytest.mark.asyncio
    async def test_ok_for_single_action(self):
        result = await propose_actions(
            ctx=None,  # ignored — tool has no side effects
            actions=[ElicitationAction(label="Go", navigate="/x")],
        )
        assert result == {"ok": True, "count": 1}

    @pytest.mark.asyncio
    async def test_ok_for_max_actions(self):
        result = await propose_actions(
            ctx=None,
            actions=[
                ElicitationAction(label=f"a{i}", navigate=f"/x{i}")
                for i in range(MAX_ACTIONS)
            ],
        )
        assert result == {"ok": True, "count": MAX_ACTIONS}

    @pytest.mark.asyncio
    async def test_rejects_empty_list(self):
        result = await propose_actions(ctx=None, actions=[])
        assert result["ok"] is False
        assert "at least one" in result["error"]

    @pytest.mark.asyncio
    async def test_rejects_over_limit(self):
        actions = [
            ElicitationAction(label=f"a{i}", navigate=f"/x{i}")
            for i in range(MAX_ACTIONS + 1)
        ]
        result = await propose_actions(ctx=None, actions=actions)
        assert result["ok"] is False
        assert f"at most {MAX_ACTIONS}" in result["error"]


# ---------------------------------------------------------------------------
# Toolset factory — ensures the tool is registered under the expected name
# ---------------------------------------------------------------------------


class TestElicitationToolset:
    def test_toolset_registers_propose_actions(self):
        ts = elicitation_toolset()
        # FunctionToolset stores tools in a dict keyed by name; expose via the
        # public API (prefer inspect over relying on internals).
        tool_names = list(ts.tools.keys()) if hasattr(ts, "tools") else []
        if not tool_names:
            # Fall back to private attr if pydantic-ai's public API changes.
            tool_names = list(getattr(ts, "_tools", {}).keys())
        assert "propose_actions" in tool_names

    def test_toolset_id_customizable(self):
        ts = elicitation_toolset(id="custom-id")
        assert ts.id == "custom-id"
