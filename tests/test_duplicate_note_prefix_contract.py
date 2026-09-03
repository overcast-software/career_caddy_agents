"""CC-257 Stage 4c — pin the `duplicate:` scrape-note prefix as a contract.

CROSS-REPO PROSE CONTRACT. `DuplicateShortCircuit` patches the scrape to
`completed` with a note that begins with the literal string ``duplicate:``.
The frontend reads that prefix back off `latestStatusNote` and branches the
paste-form flash on it:

    frontend/app/controllers/job-posts/new/paste.js:203
        const isDuplicate = note.startsWith('duplicate:');

Nothing in either repo's type system connects the two — the coupling is the
literal itself. Renaming the prefix here (to "dupe:", "duplicate -", a
localized string, or anything structured) silently degrades the paste form to
the generic "Job post created." success message on a duplicate, which is the
opposite of what happened. This test fails loudly instead.

If the prefix genuinely must change, change `paste.js` in the same PR-pair
and update this docstring; do not just re-pin the assertion.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from scrape_graph.nodes_scrape import DuplicateShortCircuit
from scrape_graph.state import ScrapeGraphState

# The literal the frontend parses. Keep in sync with paste.js:203.
DUPLICATE_NOTE_PREFIX = "duplicate:"


def test_duplicate_short_circuit_note_starts_with_the_parsed_prefix():
    state = ScrapeGraphState(scrape_id="a1fFQQe1xV", submitted_url="https://x.test/job/1")
    state.job_post_id = "V1StGXR8_Z"
    state.was_duplicate = True

    with patch("scrape_graph.nodes_scrape._patch_scrape_status") as patch_status, \
         patch("scrape_graph.nodes_scrape.trace_node"):
        asyncio.run(DuplicateShortCircuit().run(SimpleNamespace(state=state)))

    assert patch_status.call_count == 1
    _, kwargs = patch_status.call_args
    note = kwargs["note"]
    assert note.startswith(DUPLICATE_NOTE_PREFIX), (
        f"scrape note {note!r} no longer starts with {DUPLICATE_NOTE_PREFIX!r} — "
        "frontend paste.js:203 will stop recognizing duplicates"
    )
    # The id rides along after the prefix so the flash can be made specific later.
    assert state.job_post_id in note
    assert state.outcome == "duplicate"
