"""Tests for the ReviewCompleteness node.

The node exists to answer one question the graph already knows the
answer to, for free: *we* emitted this description as a stub. That is a
fact recorded in `state.stub_reason` by EvaluateExtraction, not a
judgement — so it must not depend on a model agreeing with us.

Before the node did the write, that fact died in the graph.
EvaluateExtraction set `partial_render=True`, PersistJobPost POSTed only
`state.parsed`, and the JobPost landed on `JobPost.complete`'s
`default=True`. The one component that knew the post was a stub told
nobody, so the record presented as a normal complete post — and
`complete=True` suppresses the browser extension's re-send affordance,
which means fabrication does not merely produce a bad record, it hides
itself and disables the repair path.

Worked example: jp `rHeRo6qWCG` (Siemens "Software Engineer", scraped
from linkedin.com/jobs/view/4453904340 as scrape `X04b4IjnTi`). Its
graph trace shows all 19 description ready-selectors timing out over 4
passes, an 808-character capture, Tier1Mini, and a clean run to End —
with `complete=True` on the resulting post.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from scrape_graph.nodes_extract import ReviewCompleteness, UpdateProfile
from scrape_graph.state import ScrapeGraphState


def _run(node, state: ScrapeGraphState):
    ctx = SimpleNamespace(state=state)
    return asyncio.run(node.run(ctx))


def _state(stub_reason: str | None, job_post_id: str | None = "rHeRo6qWCG"):
    state = ScrapeGraphState(
        scrape_id="X04b4IjnTi",
        submitted_url="https://www.linkedin.com/jobs/view/4453904340",
    )
    state.stub_reason = stub_reason
    state.job_post_id = job_post_id
    return state


def test_stub_marks_job_post_incomplete():
    state = _state("partial_render")

    with patch("scrape_graph.nodes_extract.httpx.patch") as mock_patch:
        mock_patch.return_value.status_code = 200
        next_node = _run(ReviewCompleteness(), state)

    assert isinstance(next_node, UpdateProfile)
    assert mock_patch.called, (
        "a stub extraction must be marked complete=False — otherwise it "
        "reads as a normal post and suppresses the re-send affordance"
    )
    url = mock_patch.call_args.args[0]
    assert url.endswith("/api/v1/job-posts/rHeRo6qWCG/")
    body = mock_patch.call_args.kwargs["json"]
    assert body["data"]["attributes"]["complete"] is False


def test_ungrounded_stub_marks_job_post_incomplete():
    """Every stub_reason marks the post, not just partial_render."""
    state = _state("ungrounded")

    with patch("scrape_graph.nodes_extract.httpx.patch") as mock_patch:
        mock_patch.return_value.status_code = 200
        _run(ReviewCompleteness(), state)

    body = mock_patch.call_args.kwargs["json"]
    assert body["data"]["attributes"]["complete"] is False


def test_real_extraction_is_left_alone():
    """No stub_reason means the description is real. The node must not
    touch `complete` — the api's own reviewer owns that judgement."""
    state = _state(None)

    with patch("scrape_graph.nodes_extract.httpx.patch") as mock_patch:
        next_node = _run(ReviewCompleteness(), state)

    assert isinstance(next_node, UpdateProfile)
    assert not mock_patch.called


def test_no_job_post_id_is_a_noop():
    """Nothing was persisted, so there is nothing to mark. Must not
    PATCH a null id."""
    state = _state("no_description", job_post_id=None)

    with patch("scrape_graph.nodes_extract.httpx.patch") as mock_patch:
        next_node = _run(ReviewCompleteness(), state)

    assert isinstance(next_node, UpdateProfile)
    assert not mock_patch.called


def test_patch_failure_does_not_break_the_graph():
    """Losing the write is bad — it is the bug this node prevents — but
    it must not cost us the JobPost we just persisted. Log and continue.
    """
    state = _state("partial_render")

    with patch("scrape_graph.nodes_extract.httpx.patch") as mock_patch:
        mock_patch.side_effect = RuntimeError("connection reset")
        next_node = _run(ReviewCompleteness(), state)

    assert isinstance(next_node, UpdateProfile)


def test_patch_rejected_by_api_does_not_break_the_graph():
    state = _state("partial_render")

    with patch("scrape_graph.nodes_extract.httpx.patch") as mock_patch:
        mock_patch.return_value.status_code = 403
        mock_patch.return_value.text = "Forbidden"
        next_node = _run(ReviewCompleteness(), state)

    assert isinstance(next_node, UpdateProfile)
