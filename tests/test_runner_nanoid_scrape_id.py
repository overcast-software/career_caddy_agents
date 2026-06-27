"""CC-100 regression — the scrape runner must carry NanoID ids as strings.

Resource ids (Scrape, ScrapeProfile) are 10-char NanoID strings under the
CC-77 cutover; ``int("V1StGXR8_Z")`` raises ValueError. The runner used to
``int(scrape["id"])`` in ``process_scrape`` and ``int(p["id"])`` in
``_fetch_profile`` — both crash on EVERY real claimed scrape, the row never
flips past ``hold``, and the failure bubbles to the poll loop's broad except
as an endless "Poll cycle failed, will retry".

These tests feed a real NanoID shape and assert no crash + the string id
flowing through verbatim. They FAIL (ValueError) against the pre-fix int()
casts and pass after. The prior green suite false-negatived this because its
fixtures used numeric-string ids ("5") that survive a broken int().
"""

from unittest.mock import MagicMock

import pytest
import yaml

import runners.scrape_runner as runner

# Real NanoID shape (alphabet [A-Za-z0-9_-], 10 chars) — non-numeric on
# purpose so int() would raise.
NANOID_SCRAPE = "V1StGXR8_Z"
NANOID_PROFILE = "a1fFQQe1xV"


class TestProcessScrapeNanoId:
    @pytest.mark.asyncio
    async def test_process_scrape_nanoid_id_does_not_crash(self, monkeypatch):
        seen: dict = {"update_ids": []}

        async def fake_update(api, scrape_id, **kw):
            seen["update_ids"].append(scrape_id)
            return "data:\n  ok: true"

        async def fake_fetch_profile(api, hostname):
            return None

        async def fake_run_graph(api, scrape_id, *a, **kw):
            seen["graph_id"] = scrape_id
            return True

        monkeypatch.setattr(runner, "update_scrape", fake_update)
        monkeypatch.setattr(runner, "_fetch_profile", fake_fetch_profile)
        monkeypatch.setattr(runner, "_run_graph", fake_run_graph)

        scrape = {
            "id": NANOID_SCRAPE,
            "attributes": {"url": "https://example.com/job/1"},
        }
        # Pre-fix this raised ValueError on int(scrape["id"]).
        result = await runner.process_scrape(MagicMock(), scrape)

        assert result is True
        # The NanoID string flows through verbatim — no int() coercion.
        assert seen["graph_id"] == NANOID_SCRAPE
        assert seen["update_ids"] == [NANOID_SCRAPE]


class TestFetchProfileNanoId:
    @pytest.mark.asyncio
    async def test_fetch_profile_returns_nanoid_string(self, monkeypatch):
        profile_body = {
            "data": [
                {
                    "id": NANOID_PROFILE,
                    "attributes": {
                        "css-selectors": {"job_data": {"title": "h1"}},
                    },
                }
            ]
        }

        async def fake_get_profile(api, hostname):
            return yaml.safe_dump(profile_body)

        monkeypatch.setattr(runner, "get_scrape_profile", fake_get_profile)

        # Pre-fix this raised ValueError on int(p["id"]).
        result = await runner._fetch_profile(MagicMock(), "example.com")

        assert result is not None
        assert result["id"] == NANOID_PROFILE
        assert result["css_selectors"] == {"job_data": {"title": "h1"}}
