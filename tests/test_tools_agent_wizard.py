"""Tests for the Agent Wizard tools registered in lib.api_tools."""

from unittest.mock import AsyncMock, patch

import pytest
import yaml

from lib.api_tools import (
    ApiClient,
    edit_cover_letter,
    edit_profile_onboarding,
    edit_resume,
    reconcile_onboarding,
    show_cover_letter,
    show_resume,
)


def _ok(data, status=200):
    """Build the new YAML response shape — agent-facing tools return the
    payload directly with no outer envelope."""
    return yaml.safe_dump(data, sort_keys=False, default_flow_style=False)


@pytest.fixture
def api():
    return ApiClient("http://api:8000", "jh_test")


class TestShowResume:
    @pytest.mark.asyncio
    async def test_returns_raw_markdown(self, api):
        expected = "## Jane Doe\n## SRE Resume\n"
        with patch.object(ApiClient, "get_text", new=AsyncMock(return_value=expected)) as mock:
            result = await show_resume(api, resume_id=42)
        assert result == expected
        mock.assert_awaited_once_with("/api/v1/resumes/42/markdown/")


class TestEditResume:
    @pytest.mark.asyncio
    async def test_patches_resume_with_supplied_fields(self, api):
        with patch.object(
            ApiClient,
            "patch",
            new=AsyncMock(return_value=_ok({"data": {"type": "resume", "id": "42"}})),
        ) as mock:
            await edit_resume(api, resume_id=42, title="Senior SRE", favorite=True)

        mock.assert_awaited_once()
        path, payload = mock.await_args.args
        assert path == "/api/v1/resumes/42/"
        attrs = payload["data"]["attributes"]
        assert attrs == {"title": "Senior SRE", "favorite": True}

    @pytest.mark.asyncio
    async def test_rejects_empty_update(self, api):
        result = yaml.safe_load(await edit_resume(api, resume_id=1))
        assert "at least one field" in result["error"]


class TestShowCoverLetter:
    @pytest.mark.asyncio
    async def test_returns_raw_markdown(self, api):
        expected = "# Cover Letter\nCreated: 2026-04-18\n\nDear hiring manager,"
        with patch.object(ApiClient, "get_text", new=AsyncMock(return_value=expected)) as mock:
            result = await show_cover_letter(api, cover_letter_id=7)
        assert result == expected
        mock.assert_awaited_once_with("/api/v1/cover-letters/7/markdown/")


class TestEditCoverLetter:
    @pytest.mark.asyncio
    async def test_patches_cover_letter(self, api):
        with patch.object(
            ApiClient,
            "patch",
            new=AsyncMock(return_value=_ok({"data": {"type": "cover-letter", "id": "3"}})),
        ) as mock:
            await edit_cover_letter(
                api, cover_letter_id=3, content="Revised body.", favorite=True
            )

        path, payload = mock.await_args.args
        assert path == "/api/v1/cover-letters/3/"
        assert payload["data"]["id"] == "3"
        assert payload["data"]["attributes"] == {
            "content": "Revised body.",
            "favorite": True,
        }

    @pytest.mark.asyncio
    async def test_rejects_empty_update(self, api):
        result = yaml.safe_load(await edit_cover_letter(api, cover_letter_id=3))
        assert "error" in result


class TestReconcileOnboarding:
    @pytest.mark.asyncio
    async def test_posts_empty_body_to_reconcile_endpoint(self, api):
        with patch.object(
            ApiClient,
            "post",
            new=AsyncMock(return_value=_ok({"wizard_enabled": True, "resume_imported": True})),
        ) as mock:
            result = yaml.safe_load(await reconcile_onboarding(api))
        mock.assert_awaited_once()
        path, payload = mock.await_args.args
        assert path == "/api/v1/onboarding/reconcile/"
        assert payload == {}
        assert result["resume_imported"] is True


class TestEditProfileOnboarding:
    @pytest.mark.asyncio
    async def test_resolves_me_then_patches_user(self, api):
        # edit_profile_onboarding now uses get_data (parsed) for the /me/
        # lookup so it can read the user id without round-tripping through
        # the agent-facing YAML serializer.
        me_payload = (
            {"data": {"type": "user", "id": "11", "attributes": {"onboarding": {}}}},
            None,
            200,
        )
        patch_response = _ok(
            {
                "data": {
                    "type": "user",
                    "id": "11",
                    "attributes": {"onboarding": {"resume_reviewed": True}},
                }
            }
        )
        with patch.object(ApiClient, "get_data", new=AsyncMock(return_value=me_payload)) as mock_get, \
             patch.object(ApiClient, "patch", new=AsyncMock(return_value=patch_response)) as mock_patch:
            result = yaml.safe_load(
                await edit_profile_onboarding(api, {"resume_reviewed": True})
            )

        mock_get.assert_awaited_once_with("/api/v1/me/")
        path, payload = mock_patch.await_args.args
        assert path == "/api/v1/users/11/"
        assert payload["data"]["attributes"]["onboarding"] == {"resume_reviewed": True}
        assert "error" not in result
        assert result["data"]["id"] == "11"

    @pytest.mark.asyncio
    async def test_rejects_empty_patch(self, api):
        result = yaml.safe_load(await edit_profile_onboarding(api, {}))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_rejects_non_dict_patch(self, api):
        result = yaml.safe_load(await edit_profile_onboarding(api, None))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_propagates_me_lookup_failure(self, api):
        failure = (None, "401 - unauthorized", 401)
        with patch.object(ApiClient, "get_data", new=AsyncMock(return_value=failure)), \
             patch.object(ApiClient, "patch", new=AsyncMock()) as mock_patch:
            result = yaml.safe_load(await edit_profile_onboarding(api, {"resume_reviewed": True}))
        assert "error" in result
        assert "401" in result["error"]
        mock_patch.assert_not_called()
