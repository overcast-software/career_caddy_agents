"""Tests for `truncate_dom` keeping `<body>` on head-heavy hosts.

PACA CC-284 — the cap was a plain prefix slice, so a host whose pre-body
markup exceeds `_MAX_DOM_BYTES` persisted a `<head>` and nothing else.
jobright.ai's scrape `kW0zmoVWjF` was 200,021 bytes with zero `div` / `a` /
`button` / `body` elements while its `job_content` held the complete job
description: the browser saw the whole page, only persistence lost it. The
row looked healthy, so every selector probe silently returned
`match_count: 0` and no Tier-0 selector could be derived for the host.

Two guarantees this file pins:

1. **The body survives when it can.** Stripping `<script>` / `<style>`
   before the cap is applied is what buys the room — that is the read-time
   prune from `lib.scrape_inspector` moved earlier in the pipeline.
2. **The failure is loud when it cannot.** If the cut still has to land
   before `<body>`, the persisted html carries a distinct marker and the
   logger gets a WARNING. Never again a healthy-looking head-only row.
"""
from __future__ import annotations

import logging

from scrape_graph._artifacts import (
    _DOM_BODY_LOST_TRAILER,
    _DOM_NOISE_STRIPPED_TRAILER,
    _DOM_TRUNCATION_TRAILER,
    _MAX_DOM_BYTES,
    _MAX_PERSISTED_BYTES,
    truncate_dom,
)


def _head_heavy_doc(*, body: str = '<div id="x">hi</div>') -> str:
    """A Next.js-shaped page: >200KB of inline critical CSS, then a body.

    Mirrors jobright.ai — 27 stylesheet links plus a large block of inline
    Ant Design / emotion CSS, all of it before `<body>`.
    """
    css = ".ant-flex-justify-normal{justify-content:normal;}" * 5_000
    links = "".join(
        f'<link rel="stylesheet" href="/_next/static/css/{i}.css">'
        for i in range(27)
    )
    return (
        f"<html><head>{links}<style>{css}</style></head>"
        f"<body>{body}</body></html>"
    )


def _link_only_head_doc() -> str:
    """>200KB of pre-body markup that stripping cannot shrink.

    No `<script>` and no `<style>`, so the noise strip has nothing to take
    and the cap genuinely has to land inside the head.
    """
    links = "".join(
        f'<link rel="preload" as="style" href="/_next/static/css/chunk-{i}.css">'
        for i in range(4_000)
    )
    return f"<html><head>{links}</head><body><div id=\"x\">hi</div></body></html>"


# ── The ticket's smallest test ────────────────────────────────────────────

def test_head_heavy_doc_still_persists_its_body():
    """The regression from CC-284: `id="x"` used to be sliced away."""
    dom = _head_heavy_doc()
    assert len(dom) > _MAX_DOM_BYTES  # guard: the input really is oversized

    out = truncate_dom(dom)

    assert 'id="x"' in out
    assert "<body" in out.lower()


def test_head_heavy_doc_reports_the_strip_and_stays_under_cap():
    out = truncate_dom(_head_heavy_doc())

    assert out.endswith(_DOM_NOISE_STRIPPED_TRAILER)
    assert _DOM_TRUNCATION_TRAILER not in out
    assert _DOM_BODY_LOST_TRAILER not in out
    # The whole document fit once the CSS went, so nothing was cut.
    assert len(out) <= _MAX_PERSISTED_BYTES
    assert "ant-flex-justify-normal" not in out


# ── The loud-marker case ─────────────────────────────────────────────────

def test_cut_before_body_is_marked_loudly(caplog):
    dom = _link_only_head_doc()
    assert len(dom) > _MAX_DOM_BYTES

    with caplog.at_level(logging.WARNING, logger="scrape_graph._artifacts"):
        out = truncate_dom(dom, scrape_id="kW0zmoVWjF")

    # The body is genuinely unreachable here — say so on the artifact...
    assert out.endswith(_DOM_BODY_LOST_TRAILER)
    assert "<body" not in out.lower()
    assert _DOM_TRUNCATION_TRAILER not in out
    # ...and in the log, naming the scrape so it is greppable.
    warnings = [
        r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
    ]
    assert any(
        "BEFORE the opening BODY tag" in m and "kW0zmoVWjF" in m for m in warnings
    ), warnings


def test_body_lost_marker_respects_the_persistence_ceiling():
    """CC-134's 200KB column expectation holds for the longer trailer too."""
    out = truncate_dom(_link_only_head_doc())

    assert out.endswith(_DOM_BODY_LOST_TRAILER)
    assert len(out) <= _MAX_PERSISTED_BYTES


# ── Unchanged behaviour ──────────────────────────────────────────────────

def test_under_cap_dom_is_returned_byte_for_byte():
    dom = '<html><head><style>a{color:red}</style></head><body>ok</body></html>'
    assert len(dom) <= _MAX_DOM_BYTES

    out = truncate_dom(dom)

    # The strip only ever runs on documents that would otherwise be cut, so
    # an under-cap capture keeps its <style> and its exact bytes.
    assert out == dom


def test_oversized_body_still_takes_the_plain_truncation_trailer():
    """A DOM that is huge *after* the body opens keeps the old contract."""
    dom = "<html><body>" + ("x" * (_MAX_DOM_BYTES * 2)) + "</body></html>"

    out = truncate_dom(dom)

    assert out.endswith(_DOM_TRUNCATION_TRAILER)
    assert _DOM_BODY_LOST_TRAILER not in out
    assert out[:_MAX_DOM_BYTES] == dom[:_MAX_DOM_BYTES]
    assert len(out) == _MAX_PERSISTED_BYTES


def test_jsonld_survives_the_strip():
    """Tier-0's JSON-LD extractor reads the persisted DOM — keep its input."""
    block = '{"@type": "JobPosting", "title": "Senior Backend Engineer"}'
    dom = (
        "<html><head>"
        f'<style>{"a{color:red}" * 30_000}</style>'
        '<script>window.__NEXT_DATA__ = 1;</script>'
        f'<script type="application/ld+json">{block}</script>'
        "</head><body><div id=\"x\">hi</div></body></html>"
    )
    assert len(dom) > _MAX_DOM_BYTES

    out = truncate_dom(dom)

    assert "JobPosting" in out
    assert "Senior Backend Engineer" in out
    assert "__NEXT_DATA__" not in out
    assert 'id="x"' in out
