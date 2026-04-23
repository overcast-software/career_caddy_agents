#!/usr/bin/env python3
"""Write the scrape-graph's live node/edge snapshot to api/.

Usage:
    uv run caddy-export-graph

Run whenever nodes or return-annotation edges change. CI has a drift
test that fails if this file is stale.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure the ai/ root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.scrape_graph.graph import export_graph_structure  # noqa: E402

# ai/ and api/ are siblings under the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_TARGET = _REPO_ROOT / "api" / "job_hunting" / "api" / "views" / "graph_static.json"


def main() -> int:
    structure = export_graph_structure()
    _TARGET.parent.mkdir(parents=True, exist_ok=True)
    _TARGET.write_text(json.dumps(structure, indent=2) + "\n")
    print(f"wrote {_TARGET} ({len(structure['nodes'])} nodes, {len(structure['edges'])} edges)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
