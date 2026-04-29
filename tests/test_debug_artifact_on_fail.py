"""Tests for the capture_debug_artifact invariant.

Every Fail terminal should snapshot the page BEFORE writing its status
note, so the admin UI has a screenshot + DOM to render post-mortem. The
helper is best-effort: detached pages, closed browsers, or API failures
don't raise.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _quarantine_browser_imports():
    before = set(sys.modules)
    yield
    for name in list(sys.modules):
        if name in before:
            continue
        if name.startswith("lib.browser") or name.startswith("mcp_servers.browser_server"):
            sys.modules.pop(name, None)


class _FakePage:
    def __init__(
        self,
        screenshot_bytes: bytes | None = b"\x89PNG\r\n\x1a\nfake",
        content_html: str = "<html><body>blocked</body></html>",
        fail_screenshot: bool = False,
        fail_content: bool = False,
    ):
        self._png = screenshot_bytes
        self._html = content_html
        self._fail_screenshot = fail_screenshot
        self._fail_content = fail_content

    async def screenshot(self, full_page: bool = False) -> bytes:
        if self._fail_screenshot:
            raise RuntimeError("page detached")
        return self._png  # type: ignore[return-value]

    async def content(self) -> str:
        if self._fail_content:
            raise RuntimeError("frame detached")
        return self._html


class _FakeState:
    def __init__(self, scrape_id: int = 1234):
        self.scrape_id = scrape_id
        self.submitted_url = "https://www.linkedin.com/jobs/view/1"
        self.canonical_url = "https://linkedin.com/jobs/view/1"
        self.screenshot_name: str | None = None


class TestCaptureDebugArtifact:
    @pytest.mark.asyncio
    async def test_screenshot_uploaded_when_page_cooperates(self):
        from scrape_graph import _artifacts as mod

        page = _FakePage()
        state = _FakeState()

        post_resp = MagicMock(status_code=200)
        get_resp = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"data": {"attributes": {"html": ""}}}),
        )
        patch_resp = MagicMock(status_code=200)

        with patch.object(mod, "httpx") as httpx_mod:
            httpx_mod.post = MagicMock(return_value=post_resp)
            httpx_mod.get = MagicMock(return_value=get_resp)
            httpx_mod.patch = MagicMock(return_value=patch_resp)
            result = await mod.capture_debug_artifact(
                page, state, reason="obstacle_fail",
            )

        assert result["screenshot_uploaded"] is True
        assert result["dom_saved"] is True
        assert result["reason"] == "obstacle_fail"
        assert state.screenshot_name is not None
        assert "obstacle_fail" in state.screenshot_name
        # Filename should have host + reason + timestamp shape.
        assert state.screenshot_name.startswith("linkedin.com_obstacle_fail_")

    @pytest.mark.asyncio
    async def test_skips_dom_patch_when_html_already_present(self):
        """Don't clobber a successful captured DOM with the post-mortem one."""
        from scrape_graph import _artifacts as mod

        page = _FakePage()
        state = _FakeState()

        post_resp = MagicMock(status_code=200)
        get_resp = MagicMock(
            status_code=200,
            json=MagicMock(return_value={
                "data": {"attributes": {"html": "<existing-captured-dom/>"}},
            }),
        )

        with patch.object(mod, "httpx") as httpx_mod:
            httpx_mod.post = MagicMock(return_value=post_resp)
            httpx_mod.get = MagicMock(return_value=get_resp)
            httpx_mod.patch = MagicMock()
            result = await mod.capture_debug_artifact(
                page, state, reason="extract_fail",
            )

        assert result["screenshot_uploaded"] is True
        assert result["dom_saved"] is False
        # PATCH must NOT be called when html is already populated.
        assert not httpx_mod.patch.called

    @pytest.mark.asyncio
    async def test_tolerates_detached_page(self):
        """A detached page raises on screenshot() and content() — helper
        must still return a structured result and not crash."""
        from scrape_graph import _artifacts as mod

        page = _FakePage(fail_screenshot=True, fail_content=True)
        state = _FakeState()

        with patch.object(mod, "httpx") as httpx_mod:
            httpx_mod.post = MagicMock()
            httpx_mod.get = MagicMock()
            httpx_mod.patch = MagicMock()
            result = await mod.capture_debug_artifact(
                page, state, reason="obstacle_fail",
            )

        assert result == {
            "screenshot_uploaded": False,
            "dom_saved": False,
            "reason": "obstacle_fail",
        }
        # Nothing called — both reads failed, no payload to upload.
        assert not httpx_mod.post.called
        assert not httpx_mod.patch.called

    @pytest.mark.asyncio
    async def test_none_page_returns_empty_result(self):
        from scrape_graph import _artifacts as mod

        result = await mod.capture_debug_artifact(
            None, _FakeState(), reason="obstacle_fail",
        )
        assert result == {
            "screenshot_uploaded": False,
            "dom_saved": False,
            "reason": "obstacle_fail",
        }

    @pytest.mark.asyncio
    async def test_screenshot_upload_failure_still_returns_structured(self):
        from scrape_graph import _artifacts as mod

        page = _FakePage()
        state = _FakeState()
        post_resp = MagicMock(
            status_code=500, text="upstream db lock"
        )
        get_resp = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"data": {"attributes": {"html": ""}}}),
        )
        patch_resp = MagicMock(status_code=200)

        with patch.object(mod, "httpx") as httpx_mod:
            httpx_mod.post = MagicMock(return_value=post_resp)
            httpx_mod.get = MagicMock(return_value=get_resp)
            httpx_mod.patch = MagicMock(return_value=patch_resp)
            result = await mod.capture_debug_artifact(
                page, state, reason="obstacle_fail",
            )

        assert result["screenshot_uploaded"] is False
        # DOM path still ran and succeeded.
        assert result["dom_saved"] is True
