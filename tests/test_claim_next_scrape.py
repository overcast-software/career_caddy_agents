"""claim_next_scrape — single-FIFO claim body (post-attended removal).

The hold queue is one FIFO; a scrape is a scrape. The claim body carries
only the optional ``runner_name`` — there is no ``attended`` partition key
on the wire any more (CC-114). These tests mock the HTTP layer (no live api)
and pin the body shape so the partition can't sneak back in.
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
        "id": "abc1234567",
        "attributes": {"url": "https://example.com/job/abc1234567", "status": "running"},
    }
}


# ---------------------------------------------------------------------------
# lib.api_tools.claim_next_scrape — request body has NO attended key
# ---------------------------------------------------------------------------


class TestClaimNextScrapeBody:
    @pytest.mark.asyncio
    async def test_body_carries_only_runner_name(self):
        api = _api_returning(_CLAIMED)
        await claim_next_scrape(api, runner_name="omarchy")
        api.post_data.assert_awaited_once()
        path, body = api.post_data.await_args.args
        assert path == "/api/v1/scrapes/claim-next/"
        assert body == {"runner_name": "omarchy"}
        # The partition key must not be on the wire.
        assert "attended" not in body

    @pytest.mark.asyncio
    async def test_no_runner_name_posts_empty_body(self):
        api = _api_returning(_CLAIMED)
        await claim_next_scrape(api)
        _path, body = api.post_data.await_args.args
        assert body == {}
        assert "attended" not in body


# ---------------------------------------------------------------------------
# runners.scrape_runner.poll_once — claims with no attended signal
# ---------------------------------------------------------------------------


class TestPollOnceClaim:
    @pytest.mark.asyncio
    async def test_poll_once_claims_without_attended(self, monkeypatch):
        seen = {}

        async def fake_claim(api, **kwargs):
            # **kwargs so a re-introduced attended= would be captured, not
            # rejected — the assertion below is what fails if it comes back.
            seen["kwargs"] = kwargs
            return "data: null"  # 204 sentinel → empty queue, no processing

        monkeypatch.setattr(runner, "claim_next_scrape", fake_claim)
        processed = await runner.poll_once(MagicMock())
        assert processed == 0
        # poll_once threads only runner_name — never an attended signal.
        assert "runner_name" in seen["kwargs"]
        assert "attended" not in seen["kwargs"]
