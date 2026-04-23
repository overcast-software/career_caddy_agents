"""Drift guard for the api-side graph_static.json snapshot.

The ai/ codebase is the single source of truth for the scrape-graph.
api/ ships a committed snapshot for its d3 / mermaid introspection
endpoints; if someone changes node topology here without re-running
`uv run caddy-export-graph`, this test fails with a reproduction hint.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lib.scrape_graph.graph import NODE_META, export_graph_structure


_SNAPSHOT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "api" / "job_hunting" / "api" / "views" / "graph_static.json"
)


def test_exported_structure_shape():
    data = export_graph_structure()
    assert {"nodes", "edges"}.issubset(data)
    ids = {n["id"] for n in data["nodes"]}
    assert ids == NODE_META.keys()
    for edge in data["edges"]:
        assert edge["from"] in ids
        assert edge["to"] in ids


def test_every_node_has_description():
    """The d3 force graph renders description text as hover tooltips;
    a missing description on any node is a user-visible gap. Assert
    all nodes carry non-trivial prose so the drift test catches new
    nodes shipped without docs."""
    for node in export_graph_structure()["nodes"]:
        desc = node.get("description", "")
        assert desc and len(desc) >= 40, (
            f"node {node['id']} needs a description "
            f"(got {desc!r}) — update NODE_META in lib/scrape_graph/graph.py"
        )


def test_snapshot_has_no_self_edges():
    for edge in export_graph_structure()["edges"]:
        assert edge["from"] != edge["to"], edge


@pytest.mark.skipif(
    not _SNAPSHOT_PATH.exists(),
    reason="api/ checkout not present (ai-only test run)",
)
def test_committed_snapshot_matches_live_graph():
    live = export_graph_structure()
    committed = json.loads(_SNAPSHOT_PATH.read_text())
    assert committed == live, (
        f"graph_static.json drift — regenerate with `uv run caddy-export-graph`.\n"
        f"snapshot: {_SNAPSHOT_PATH}"
    )
