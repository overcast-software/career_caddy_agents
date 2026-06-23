"""Attended scrape routing — the runner-side half of the FROZEN wire contract.

The api partitions the hold queue by an ``attended`` flag: a claim with body
``attended=False`` (or absent) matches only non-attended holds, and
``attended=True`` matches only attended holds. This keeps a default headless
runner from grabbing a scrape that a human queued to solve a captcha/login in
the ``--attended`` resident window.

These tests mock the HTTP layer (no live api) and assert the exact wire key
``attended`` and that the runner threads its ``--attended`` flag through to the
claim call.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

import runners.scrape_runner as runner
from lib.api_tools import claim_next_scrape


def _api_returning(payload, error=None, status=200):
    """Fake ApiClient whose post_data returns a parsed (payload, error, status)."""
    api = MagicMock()
    api.post_data = AsyncMock(return_value=(payload, error, status))
    return api


_CLAIMED = {
    "data": {
        "type": "scrape",
        "id": "5",
        "attributes": {"url": "https://example.com/job/5", "status": "running"},
    }
}


# ---------------------------------------------------------------------------
# lib.api_tools.claim_next_scrape — request body
# ---------------------------------------------------------------------------


class TestClaimNextScrapeBody:
    @pytest.mark.asyncio
    async def test_attended_true_sends_attended_true(self):
        api = _api_returning(_CLAIMED)
        await claim_next_scrape(api, runner_name="omarchy", attended=True)
        api.post_data.assert_awaited_once()
        path, body = api.post_data.await_args.args
        assert path == "/api/v1/scrapes/claim-next/"
        # Exact wire key, exact boolean value.
        assert body["attended"] is True

    @pytest.mark.asyncio
    async def test_default_sends_attended_false(self):
        api = _api_returning(_CLAIMED)
        await claim_next_scrape(api, runner_name="omarchy")
        _path, body = api.post_data.await_args.args
        # The default runner explicitly claims the non-attended partition.
        assert "attended" in body
        assert body["attended"] is False

    @pytest.mark.asyncio
    async def test_attended_false_explicit_sends_false(self):
        api = _api_returning(_CLAIMED)
        await claim_next_scrape(api, runner_name="omarchy", attended=False)
        _path, body = api.post_data.await_args.args
        assert body["attended"] is False

    @pytest.mark.asyncio
    async def test_runner_name_still_sent_alongside_attended(self):
        api = _api_returning(_CLAIMED)
        await claim_next_scrape(api, runner_name="pibu", attended=True)
        _path, body = api.post_data.await_args.args
        assert body["runner_name"] == "pibu"
        assert body["attended"] is True


# ---------------------------------------------------------------------------
# runners.scrape_runner.poll_once — threads --attended through to the claim
# ---------------------------------------------------------------------------


class TestPollOnceThreadsAttended:
    @pytest.mark.asyncio
    async def test_attended_mode_passes_attended_true(self, monkeypatch):
        seen = {}

        async def fake_claim(api, *, runner_name=None, attended=False):
            seen["attended"] = attended
            return "data: null"  # 204 sentinel → empty queue, no processing

        monkeypatch.setattr(runner, "claim_next_scrape", fake_claim)
        processed = await runner.poll_once(MagicMock(), attended=True)
        assert processed == 0
        assert seen["attended"] is True

    @pytest.mark.asyncio
    async def test_default_mode_passes_attended_false(self, monkeypatch):
        seen = {}

        async def fake_claim(api, *, runner_name=None, attended=False):
            seen["attended"] = attended
            return "data: null"

        monkeypatch.setattr(runner, "claim_next_scrape", fake_claim)
        processed = await runner.poll_once(MagicMock())
        assert processed == 0
        assert seen["attended"] is False
