# MCP Servers

Four FastMCP servers live here. Two ship to production from this image; two are local-only.

| Server | File | Transport | Port | Auth | Deploy posture | Consumers |
|---|---|---|---|---|---|---|
| Public MCP | `public_server.py` | SSE | `:8030` (prod), `:8000` (local) | Per-client `jh_*` API key (read-only) | **Prod** at `mcp.careercaddy.online` (entrypoint `caddy-public`) | External MCP clients (Claude Desktop, cc_auto, Cursor, etc.) |
| Chat | `chat_server.py` | SSE | `:8031` (prod), `:8000` (local) | Session cookie (proxied through api) | **Prod** internal-only (entrypoint `caddy-chat`) | The Ember frontend, via the api SSE proxy |
| Browser | `browser_server.py` | stdio + SSE | `:3004` | None (local trust boundary) | **Local-only** (Camoufox + Playwright; never on the VPS) | Local agents in pipelines; the hold-poller as a Python import (not over MCP) |
| Career Caddy CRUD | `career_caddy_server.py` | stdio | — | `CC_API_TOKEN` | **Local-only** (subprocess, not a daemon) | Pydantic-AI agents that need read/write API access |

The same Docker image runs as **two** prod services (`caddy-public` and `caddy-chat`) under different entrypoints — see `docker-compose.prod.yml`. Browser and the Career Caddy CRUD server never run on prod.

The browser server's `scrape_page` tool stays exposed for ad-hoc / paste-form debugging, but the production scrape path is `pollers/hold_poller.py`, which calls `scrape_page` as a Python function rather than over MCP.
