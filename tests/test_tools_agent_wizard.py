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
        assert path == "/api/v1/users/me/onboarding/reconcile/"
        assert payload == {}
        assert result["resume_imported"] is True


class TestEditProfileOnboarding:
    @pytest.mark.asyncio
    async def test_patches_onboarding_endpoint_directly(self, api):
        # edit_profile_onboarding uses the nested singleton-per-user
        # /api/v1/users/me/onboarding/ endpoint. The `me` alias resolves
        # server-side to request.user. Response is JSON:API:
        # {"data": {"type": "onboarding", "id": "...", "attributes": {...}}}.
        patch_response = _ok(
            {
                "data": {
                    "type": "onboarding",
                    "id": "11",
                    "attributes": {
                        "derived": {},
                        "subjective": {
                            "resume_reviewed": True,
                            "wizard_enabled": True,
                        },
                    },
                },
                "meta": {"source": "stored"},
            }
        )
        with patch.object(ApiClient, "patch", new=AsyncMock(return_value=patch_response)) as mock_patch:
            result = yaml.safe_load(
                await edit_profile_onboarding(api, {"resume_reviewed": True})
            )

        path, payload = mock_patch.await_args.args
        assert path == "/api/v1/users/me/onboarding/"
        assert payload == {"resume_reviewed": True}
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_rejects_empty_patch(self, api):
        result = yaml.safe_load(await edit_profile_onboarding(api, {}))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_rejects_non_dict_patch(self, api):
        result = yaml.safe_load(await edit_profile_onboarding(api, None))
        assert "error" in result
