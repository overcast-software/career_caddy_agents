"""CC-101 regression — the score poller must read JobPost ids as NanoID
strings, not int().

``_job_post_id_from_scrape`` used to ``int(raw)`` and swallow the ValueError
to None. Under CC-77 every JobPost id is a 10-char NanoID string, so int()
raised on EVERY row → the function returned None for every post →
``_collect_candidates`` was permanently empty → the (opt-in) score poller
silently scored nothing, with no error in the logs.

These assert the string id flows through. They return None (the swallowed
ValueError) against the pre-fix int() cast and the real id after. The prior
suite had no NanoID coverage here at all.
"""

from pollers.score_poller import _job_post_id_from_scrape

NANOID_JP = "V1StGXR8_Z"


def _row(rel_key: str, data):
    return {"relationships": {rel_key: {"data": data}}}


def test_returns_nanoid_string_dasherized_key():
    row = _row("job-post", {"type": "job-post", "id": NANOID_JP})
    result = _job_post_id_from_scrape(row)
    assert result == NANOID_JP
    assert isinstance(result, str)


def test_returns_nanoid_string_underscored_key():
    row = _row("job_post", {"type": "job-post", "id": NANOID_JP})
    assert _job_post_id_from_scrape(row) == NANOID_JP


def test_none_when_no_relationship():
    assert _job_post_id_from_scrape({"relationships": {}}) is None
    assert _job_post_id_from_scrape({}) is None


def test_none_when_relationship_data_null():
    assert _job_post_id_from_scrape(_row("job-post", None)) is None
