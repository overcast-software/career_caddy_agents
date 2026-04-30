# career_caddy_agents

The agent runtime for [Career Caddy](https://github.com/overcast-software/career_caddy). Pydantic-AI agents drive every other piece in this tree — they call MCP-served tools, drive the browser engine, and run the scrape-graph extraction pipeline.

## Layout

```
agents/                         (this repo)
├── agents/                     # Pydantic-AI agent definitions (job_extractor, obstacle, onboarding, career_caddy CRUD)
├── mcp_servers/                # 4 MCP servers — see mcp_servers/README.md for the deploy table
│   ├── public_server.py        #   prod :8030 — mcp.careercaddy.online (per-client jh_* keys, read-only)
│   ├── chat_server.py          #   prod :8031 — frontend chat (proxied via api)
│   ├── browser_server.py       #   local-only :3004 — Camoufox/Playwright
│   └── career_caddy_server.py  #   local-only stdio — CRUD against the Career Caddy REST API
├── browser/                    # Browser engine, sessions, credentials (local-only)
├── scrape_graph/               # pydantic-graph state machine for scrape + extract
├── pollers/                    # Long-running daemons
│   ├── hold_poller.py          #   caddy-poller — production worker
│   └── score_poller.py         #   caddy-score
├── tools/                      # One-shot operator scripts
│   ├── manual_login.py
│   ├── discover_sites.py
│   ├── export_graph_structure.py
│   └── fetch_chromium.py
├── lib/                        # Shared utilities (api_tools, toolsets, history, models, …)
├── tests/                      # 25 test modules (pytest)
├── sites.yml                   # Versioned login selectors per domain
├── secrets.yml.example         # Credentials template (gitignored: secrets.yml)
├── pyproject.toml
└── Dockerfile
```

## Entry points (from `pyproject.toml`)

| Command | Module | What it does |
|---|---|---|
| `caddy-pipeline` | `agents.job_email_to_caddy:run` | Scrape one URL → extract → post to Career Caddy |
| `caddy-poller` | `pollers.hold_poller:main_sync` | Production hold-poller daemon |
| `caddy-score` | `pollers.score_poller:run` | Score/rank scrapes (heuristic, no LLM) |
| `caddy-public` | `mcp_servers.public_server:main` | Public MCP gateway (prod entrypoint) |
| `caddy-chat` | `mcp_servers.chat_server:main` | SSE chat service (prod entrypoint) |
| `caddy-export-graph` | `tools.export_graph_structure:main` | Dump scrape-graph nodes/edges as JSON for viz |
| `caddy-fetch-chromium` | `tools.fetch_chromium:main` | Download Playwright Chromium (ARM/Pi) |
| `caddy-fetch-browser` | `camoufox.__main__:main` | Download Camoufox/Firefox |

## Deploy posture

The Docker image runs as **two prod services** (`caddy-public` and `caddy-chat`) under different entrypoints — no browser, no Camoufox. The browser-mcp server, hold-poller, and score-poller are **local-only**, intended to run on a desktop or Raspberry Pi.

| Surface | Where it runs |
|---|---|
| `mcp_servers/public_server.py` | Prod VPS (`:8030`, `mcp.careercaddy.online`) |
| `mcp_servers/chat_server.py` | Prod VPS (`:8031`, internal-only behind api proxy) |
| `mcp_servers/browser_server.py` | Local dev / Pi (`:3004`) |
| `pollers/hold_poller.py` | Local dev / Pi (drives the production scrape path) |

## Setup

```bash
# Install
pip install uv && uv sync

# Browser binary (one-time)
python -m camoufox fetch          # Camoufox (~200 MB, default engine)
# OR for ARM/Pi:
uv run caddy-fetch-chromium       # Playwright Chromium

# Configure
cp secrets.yml.example secrets.yml   # Login credentials for browser automation
# Set CC_API_TOKEN, OPENAI_API_KEY (or ANTHROPIC_API_KEY) in your .envrc / .env
```

## Tests

```bash
uv run pytest tests/
```

See `CLAUDE.md` for agent responsibilities, model selection, scrape-graph status, and other detail.
