# agents/CLAUDE.md

Guidance for Claude Code when working in `agents/`. This file is a
pointer; the canonical state lives in `agents/notes.org`.

## Source of truth — read FIRST

- **`agents/notes.org`** (drill via `claude/cag-*`) — scrape-graph
  state machine, MCP server boundary (prod vs local), agent-memory-
  does-not-flow-to-production rule, ScrapeProfile as source of truth,
  selector-engine boundary, runner safety invariants.
- **Parent `todo.org`** (drill via `claude/cc-*`) — agents
  work-items are filed under the parent `Inbox`; there is no
  `agents/todo.org`.

Boot sequence (every cc-agents / scrape-profile-enhancer session):

```
emacsclient --eval '(claude/cag-help)'
emacsclient --eval '(claude/cag-notes-toc)'
emacsclient --eval '(claude/cag-notes-read "Architecture/Scrape-graph state machine")'
emacsclient --eval '(claude/cag-notes-read "Architecture/MCP server boundary")'
emacsclient --eval '(claude/cag-notes-read "Architecture/Agent memory does not flow to production")'
```

For ScrapeProfile / selector / hint work, also read
`Architecture/ScrapeProfile as source of truth` and
`Architecture/Selector-engine boundary`.

## What This Is

Career Caddy AI provides browser automation, job extraction, and chat agents for the Career Caddy backend API. Email-based workflows (notmuch classification, email pipeline) have been moved to the parent's `automation/` submodule (formerly the `career_caddy_automation` sibling repo, promoted to first-class on 2026-05-30).

## Where `agents/` fits in the four-submodule layout

The parent repo (`career_caddy`) has four submodules:

- `api/` — Django REST + MCP backend.
- `frontend/` — Ember.js 6.x SPA.
- **`agents/`** (this submodule) — **server-side, service-driven**. Runs as Docker containers in the prod stack for *everyone*: Camoufox + Playwright browser, scrape-graph state machine, prod MCP servers (`chat_server.py` + `public_server.py` at `:8031` + `:8030`), the scrape runner (`runners/scrape_runner.py` — formerly `pollers/hold_poller.py`, renamed 2026-05-30; claims hold scrapes via `POST /api/v1/scrapes/claim-next/` with `SELECT FOR UPDATE SKIP LOCKED` so N concurrent runners coexist safely), the score poller (`pollers/score_poller.py`, retiring via the django-q2 phased rollout in parent).
- `automation/` — **user-side, operator-driven**. Email triage pipeline, caddy-web copilot, A2A orchestrator, link traverser, sharpen_profiles. Runs on *one user's* machines (laptop, pibu, home server). HTTP-only contract with the api + public MCP — no Python imports cross.

**The boundary is service vs operator.** When deciding whether a piece of code belongs in `agents/` or `automation/`:

- *Is it a service for everyone?* → `agents/`. Browser/Camoufox, scrape pipeline, prod MCP servers, pollers/workers, anything Docker-shipped to all users.
- *Is it an operator for one user?* → `automation/`. Email-driven flows, user-side copilot, anything that runs on one human's machine.

Cross-link: parent's `CLAUDE.md` has the same role-split text and is the canonical home for the framing. The full reconstruction plan lives at `career_caddy/notes.org/Plans/Promoting cc_auto → automation/ — first-class submodule`.

## Environment Setup

Dependencies are managed via `pyproject.toml` with `uv`. Environment variables come from `.envrc` (use `direnv` or `source .envrc`):

| Variable | Required | Purpose |
|----------|----------|---------|
| `CC_API_TOKEN` | Yes | Career Caddy backend auth token |
| `CC_API_BASE_URL` | No | API base URL (default: `https://api.careercaddy.online`; set to `http://localhost:8000` for local dev) |
| `CC_RUNNER_NAME` | No | Runner identifier recorded in `Scrape.claimed_by`. Default: `socket.gethostname()`. Set per host when running multi-runner (`omarchy`, `pibu`, …) so logfire / db queries can attribute work to the right box. |
| `OPENAI_API_KEY` | Yes* | LLM provider (* or use ANTHROPIC_API_KEY) |
| `ANTHROPIC_API_KEY` | No | Alternative LLM provider |
| `FASTMCP_HOST` | No | MCP server bind host (default: `0.0.0.0`) |
| `FASTMCP_PORT` | No | MCP server port (default: `3002`) |
| `CAMOUFOX_DATA_DIR` | No | Where camoufox stores its browser binary |
| `BROWSER_ENGINE` | No | `camoufox` (default) or `chrome` (Playwright Chromium + stealth) |
| `BROWSER_HEADLESS` | No | `true` (default) or `false` — also settable via `--headless`/`--headed` CLI flags |
| `BROWSER_PROXY_SERVER` | No | e.g. `socks5://localhost:1080` or `http://host:3128`. Applied to both engines. |
| `BROWSER_PROXY_USERNAME` | No | Proxy auth. **Chromium ignores auth on SOCKS proxies** — use camoufox for authed SOCKS5. |
| `BROWSER_PROXY_PASSWORD` | No | Proxy auth. See caveat above. |
| `BROWSER_PROXY_BYPASS` | No | Comma-separated host list to exclude from the proxy. |
| `OBSTACLE_AGENT_MODEL` | No | LLM for the obstacle agent that resolves login walls / account choosers. Falls back to `BROWSER_SCRAPER_MODEL`. |
| `LOGFIRE_TOKEN` | No | Observability / tracing |
| `OLLAMA_API_BASE` | No | Local Ollama endpoint (default: `http://127.0.0.1:11434`) |

```bash
# 1. Install dependencies (requires Python 3.13+)
pip install uv
uv sync

# 2. Download browser binary (one-time)
python -m camoufox fetch          # Camoufox/Firefox (~200MB, default engine)
# OR for ARM/Raspberry Pi:
uv run caddy-fetch-chromium       # Playwright Chromium (ARM-compatible)

# 3. Configure environment
cp .envrc.example .envrc    # Required: CC_API_TOKEN, OPENAI_API_KEY
source .envrc               # or: use direnv

# 4. Set up browser credentials (needed for browser automation)
cp secrets.yml.example secrets.yml   # fill in your job site credentials
```

**Creating `CC_API_TOKEN`**: After initializing the Career Caddy API (see `api/CLAUDE.md`), create a long-lived key:

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/token/ \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"secret"}' | python3 -c "import sys,json; print(json.load(sys.stdin)['access'])")

curl -X POST http://localhost:8000/api/v1/api-keys/ \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name":"ai-agent"}'
# Copy the "key" field from the response → CC_API_TOKEN
```

## Browser Engine

Two engines are available — select via `--engine` CLI flag or `BROWSER_ENGINE` env var:

| Engine | Binary | Anti-fingerprint | ARM/Pi |
|--------|--------|------------------|--------|
| `camoufox` (default) | Firefox fork | C++-level patches | No |
| `chrome` | Playwright Chromium | `playwright-stealth` CDP patches | Yes |

```bash
# Browser MCP server
python mcp_servers/browser_server.py --engine chrome --headless

# Scrape runner (Raspberry Pi) — see Multi-Runner Deployment below
uv run caddy-runner --engine chrome --headless

# Manual login (always headed)
python tools/manual_login.py --engine chrome linkedin.com
```

Sessions (`~/.career_caddy/sessions/`) are stored in Playwright's universal cookie format and are portable across engines. A session saved with Camoufox works with Chromium and vice versa.

## Running the Pipeline

```bash
# Scrape a single job URL and add it to Career Caddy
uv run caddy-pipeline https://example.com/job/posting
```

## Architecture

### Core Pattern: Pydantic-AI Agents + MCP Servers

Agents (in `agents/`) use the `pydantic-ai` framework. They access tools through **MCP servers** (in `mcp_servers/`) using two transport types:
- `MCPServerStdio` — launches the server as a subprocess (used by the pipeline; no external service needed)
- `MCPServerSSE` — connects to a running HTTP/SSE server (used when the `browser-mcp` docker service is running)

**Default LLM model**: `openai:gpt-4o-mini`. Change via the `model=` argument to `Agent(...)`.

### MCP Servers

See `mcp_servers/README.md` for the canonical table (prod vs local-only, transport, port, auth, consumers). Quick summary:

| Server | Transport | Port | Deploy |
|--------|-----------|------|--------|
| `public_server.py` | SSE | `:8030` prod / `:8000` local | **Prod** (`mcp.careercaddy.online`, per-client `jh_*` keys) |
| `chat_server.py` | SSE | `:8031` prod / `:8000` local | **Prod** (frontend chat, internal-only) |
| `browser_server.py` | stdio + SSE | `:3004` | Local-only (Camoufox/Playwright) |
| `career_caddy_server.py` | stdio | — | Local-only (CRUD against api) |

**Hold-poller is the only supported scrape driver.** The historical
synchronous flow — api `Scraper.dispatch()` POSTing to a browser-MCP
HTTP endpoint (`/scrape_job` on `localhost:3012`) — is gone. Every
scrape created through `POST /api/v1/scrapes/` defaults to
`status="hold"`; the scrape runner (`runners/scrape_runner.py`)
claims them via `POST /api/v1/scrapes/claim-next/`, drives
extraction through the scrape-graph, and patches the row when done.
`browser_server.py`'s `scrape_page` MCP tool stays for ad-hoc
exploration (paste-form fallback, manual debugging) but no api code
calls it anymore.

### Agent Responsibilities

- **`career_caddy_agent.py`** — Validates and posts jobs to Career Caddy API; checks for duplicates before creating
- **`job_extractor_agent.py`** — Extracts structured `JobPostData` from raw job posting text
- **`job_email_to_caddy.py`** — URL scraping pipeline: browser scrape → extract → Career Caddy post

**Critical rules in `career_caddy_agent.py`** (enforced via system prompt):
1. Always call `find_job_post_by_link(url)` before creating — never create duplicates
2. Use `create_job_post_with_company_check(company_name)`, not `create_job_post()` (avoids FK errors)
3. Stop immediately if any tool returns `{"success": false, ...}`
4. Never retry failed tool calls or scan by incrementing IDs

### Data Models

- `lib/models/job_models.py` — `JobPostData`, `CompanyData` (primary DTOs between agents)
- `lib/models/career_caddy.py` — API-specific models (`JobPostCreate`, `APIResponse`, `APICredentials`)

### Credentials & Browser Auth

`browser/credentials.py` loads two YAML files:

**`secrets.yml`** (gitignored — create from `secrets.yml.example`):
```yaml
linkedin.com:
  username: your_email@example.com
  password: your_password
```

**`sites.yml`** (versioned — add login automation config for new sites):
```yaml
linkedin.com:
  login_url: https://www.linkedin.com/login
  username_selector: "#username"
  password_selector: "#password"
  submit_selector: ".login__form_action_container button"
  post_login_check: ".global-nav__me"
```

Domain lookup normalizes subdomains automatically (`www.linkedin.com` → `linkedin.com`).

## Multi-Runner Deployment

N scrape runners can point at the same Career Caddy api and split
the `status='hold'` queue without coordinating with each other. The
canonical deployment is omarchy (workstation) + pibu (Raspberry Pi),
but the same pattern extends to any host with outbound HTTPS to the
api and a Camoufox or Chromium install.

### Why it's safe

Three pieces have to line up — all shipped:

1. **Atomic claim** (api Phase 1, `POST /api/v1/scrapes/claim-next/`).
   Wrapped in `SELECT FOR UPDATE SKIP LOCKED` against the oldest
   `status='hold'` row. Concurrent runners see different rows; the
   loser in a race picks up the next one down the queue, never the
   same row twice.
2. **Heartbeat on every non-terminal write** (api Phase 1, in
   `_log_scrape_status`). Each status update (`running`,
   `extracting`, `updating_profile`, …) bumps `claimed_at = NOW()`,
   so a long-running scrape doesn't look stale to the sweep.
   Terminal writes (`completed`, `failed`) clear `claimed_at` +
   `claimed_by` so post-mortem queries can tell finished work from
   in-flight work.
3. **Lease sweep** (api Phase 2, `sweep_stale_scrape_claims`).
   django-q2 schedule, every 5 min, resets non-terminal rows whose
   `claimed_at` is older than 15 min back to `status='hold'` with
   `claimed_at=NULL`. Catches OOM kills, host reboots, network
   splits, anything that severs a runner mid-scrape. The 15 min
   default lives in `api/job_hunting/lib/tasks.py::_DEFAULT_LEASE_MINUTES`
   — bump it via the schedule arg if you have a profile that
   legitimately takes longer.

The combined invariant: **a Scrape row is either claimable
(`hold` + `claimed_at IS NULL`), actively being heartbeat-ed by a
live runner (non-terminal + `claimed_at` recent), or done
(`completed`/`failed` + `claimed_at NULL`)**. No fourth state, no
manual cleanup needed.

### Runner naming

Set `CC_RUNNER_NAME` per host. The runner uses
`os.environ.get("CC_RUNNER_NAME") or socket.gethostname()` and
passes it as `runner_name` on the claim POST; api stores it on
`Scrape.claimed_by`. Naming matters because:

- Logfire spans tag the runner via the same value.
- Blame attribution when the sweep resets a stale claim ("runner
  `pibu` lost claim on scrape 4271 after 18 min").
- Local `claimed_by` queries (`SELECT claimed_by, COUNT(*) FROM
  scrapes WHERE status='running' GROUP BY 1`) tell you which runner
  is hot.

Stick to short, host-shaped names (`omarchy`, `pibu`, `pibu-2`).
Avoid PIDs or timestamps — the value persists past the runner's
lifetime in the audit row.

### Omarchy (existing workstation runner)

Already deployed. The only change required for multi-runner is
setting the name:

```bash
# .envrc on omarchy
export CC_RUNNER_NAME=omarchy
```

Then re-run via the existing entry:

```bash
make runner                              # camoufox, headless, against prod
make runner ARGS="--engine chrome"       # chromium + stealth
make runner ARGS="--headed"              # headed; one resident window, ephemeral tabs (watch + screenshot to verify)
make runner-local                        # against http://localhost:8000
# `--attended` is retained as a hidden deprecated alias of `--headed` (logs a
# warning) so the live pibu unit + tmuxinator invocations keep working through
# the rollout. It no longer partitions the scrape queue — a scrape is a scrape.
```

### Pibu (Raspberry Pi runner)

Pibu has ~400Mi RAM headroom under steady state; Camoufox eats
200–400Mi while a scrape is open, so the box can host exactly one
runner. Use Chromium (`--engine chrome`) — Camoufox is x86_64-only.

```bash
# One-time on pibu
ssh pibu 'curl -LsSf https://astral.sh/uv/install.sh | sh'
ssh pibu 'git clone --depth=1 https://github.com/overcast-software/career_caddy_agents.git ~/Projects/career_caddy_agents'
ssh pibu 'cd ~/Projects/career_caddy_agents && uv sync && uv run caddy-fetch-chromium'

# Per-host env (~/.config/environment.d/career-caddy.conf or .envrc)
CC_API_BASE_URL=https://api.careercaddy.online
CC_API_TOKEN=<long-lived jh_* key>
CC_RUNNER_NAME=pibu
BROWSER_ENGINE=chrome
BROWSER_HEADLESS=true
```

Systemd-user unit (`~/.config/systemd/user/caddy-runner.service`):

```ini
[Unit]
Description=Career Caddy scrape runner
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=%h/Projects/career_caddy_agents
EnvironmentFile=%h/.config/environment.d/career-caddy.conf
ExecStart=%h/.local/bin/uv run caddy-runner --engine chrome --headless
Restart=on-failure
RestartSec=30s

[Install]
WantedBy=default.target
```

```bash
ssh pibu systemctl --user daemon-reload
ssh pibu systemctl --user enable --now caddy-runner.service
ssh pibu journalctl --user -fu caddy-runner.service
```

### Verifying the split

After both runners are live, queue up two-plus scrapes via the UI
or `POST /api/v1/scrapes/` and check that they fan out:

```sql
-- Inside the api container: docker compose exec db psql -U postgres -d job_hunting
SELECT id, status, claimed_by, claimed_at
FROM job_hunting_scrape
WHERE claimed_at > NOW() - interval '15 minutes'
ORDER BY claimed_at DESC;
```

Both `omarchy` and `pibu` should appear in `claimed_by`. If only
one host ever shows, the other runner isn't claiming — check its
journal for auth errors (`CC_API_TOKEN` revoked) or DNS/HTTPS
reachability to the api.

For real-time visibility, logfire's runner spans tag
`runner_name` — query by attribute in the dashboard to see per-host
throughput.

### What's deferred

- **Admin dashboard surface** (per-runner claim counts, last-seen,
  current scrape) — deferred to *Plans/Scrape runner Phase 5 —
  operational surface*. The SQL above is the interim view.
- **Capability routing** (route X-host scrapes to a specific
  runner) — intentionally out of scope. The plan rejected
  PeerTube's job-runner protocol for being heavier than we need;
  any scrape can run on any runner. If a domain breaks under
  Chromium, fix it in `ScrapeProfile` rather than pinning to
  Camoufox runners.
- **Runner-scoped API keys** (a `runner` scope distinct from staff
  keys) — Phase 4 nice-to-have noted in the claim-next view
  docstring; defer until there's an actual privilege boundary to
  enforce.

## LLM Configuration

Per-agent model overrides are controlled via environment variables in `agents/agent_factory.py:get_model()`.

Resolution order: role-specific env var → `CADDY_DEFAULT_MODEL` → hardcoded `openai:gpt-4o-mini`.

| Env Var | Agent Role | Recommended |
|---------|------------|-------------|
| `CADDY_MODEL` | career_caddy_agent (CRUD) | gpt-4o-mini |
| `CHAT_MODEL` | chat_server (user-facing) | claude-haiku-4-5 |
| `JOB_EXTRACTOR_MODEL` | job_extractor | claude-haiku-4-5 |
| `BROWSER_SCRAPER_MODEL` | browser scraper | gpt-4o-mini |
| `CADDY_DEFAULT_MODEL` | fallback for all roles | gpt-4o-mini |

The scrape runner (`runners/scrape_runner.py`) skips the browser_scraper LLM entirely — it calls `scrape_page()` directly as a Python function, then hands content to the job extractor.

## Scrape Graph (Phase 1b skeleton)

The scrape+extract pipeline is being migrated to an explicit
pydantic-graph state machine. agents/ owns the runtime; api/ exposes thin
persistence endpoints the graph nodes POST to.

**Status**: Phase 1b skeleton merged. Feature flag defaults to off so
nothing in production touches the graph yet. Phase 1c lands the
frontend d3/mermaid visualization; Phase 1d wires browser_server to
actually dispatch the graph.

**Callers**: these entry points all feed the same extract sub-graph
once Phase 1d ships. The graph itself has no knowledge of who called:
- Hold-poller (via browser_server) — runs the full scrape + extract
  sub-graph with an active Playwright page.
- Browser-extension bookmarklet → paste form — enters at
  `StartExtract` (no Playwright needed, text already posted).
- Chat ingest — same as paste, enters at `StartExtract`.
- cc_auto email pipeline — same, enters at `StartExtract` with
  `source="email"`. cc_auto is a caller, not a participant; it runs
  as its own process and never imports scrape_graph directly.

**Per-tier model overrides**:
- `SCRAPE_GRAPH_TIER1_MODEL` (default `openai:gpt-4o-mini`)
- `SCRAPE_GRAPH_TIER2_MODEL` (default `anthropic:claude-haiku-4-5`)
- `SCRAPE_GRAPH_TIER3_MODEL` (default `anthropic:claude-sonnet-4-6`)
- `SCRAPE_GRAPH_ENABLE_TIER3=1` to allow escalation into Tier 3;
  otherwise the graph terminates at `ExtractFail` after Tier 2.

**Visualization**:
- `GET /api/v1/admin/graph-structure/` — static {nodes, edges} for
  a d3 force-layout.
- `GET /api/v1/admin/graph-mermaid/` — mermaid stateDiagram-v2
  source, renderable via mermaid.js or mermaid.live.
- `GET /api/v1/scrapes/:id/graph-trace/` — ordered transitions for a
  single scrape; walks `source_scrape` chain so a tracker URL + its
  canonical child render as one path.
- `GET /api/v1/admin/graph-aggregate/?since=7d` — per-edge counts +
  success rates for the eval loop.
- `python manage.py dump_graph_traces --since 7d --format jsonl`
  emits training data for offline analysis.

**Canonical node registry**: `agents/scrape_graph/graph.py`. The static
snapshot in `api/job_hunting/api/views/graph.py` must stay in sync;
Phase 1d will export from the agents side to make that automatic.

## Agent memory does NOT flow to production extraction tiers

`.claude/agent-memory/<agent-name>/` is loaded only when the
matching Claude Code subagent is next invoked. None of the
production scrape pipeline reads it:

- **Hold-poller + scrape-graph** (Python service) — reads
  `ScrapeProfile` rows + `nodes_*.py` code.
- **Tier1Mini / Tier2Haiku / Tier3** (Pydantic-AI agents in
  `agents/scrape_graph/nodes_extract.py`) — hardcoded system
  prompts + `ScrapeProfile.extraction_hints` blob, that's it.
- **browser-mcp** (Camoufox/Playwright) — `ScrapeProfile.css_selectors`,
  `sites.yml`, `secrets.yml`.

So when an investigator agent (e.g. `scrape-profile-enhancer`)
discovers a generalizable pattern, recording it as memory keeps the
insight available to future *investigator* sessions but does NOT
change runtime extraction behavior. To make a learning actually
shape scrapes you must land it in one of three places production
reads:

1. **`ScrapeProfile` fields** — declarative rules read at scrape
   time. Best home for host-specific selectors (`css_selectors`,
   `ready_selector`), URL canonicalisation (`url_rewrites`),
   apply-resolver hints (`apply_resolver_config`), and any
   "previous extractions found …" priors fed into the LLM
   (`extraction_hints`). The 2026-05-01 LinkedIn `/comm/` →
   `/jobs/view/` fix landed via `url_rewrites` for exactly this
   reason.
2. **Pydantic response models + validators** — bake structural
   guarantees into `ParsedJobData` (or sibling models) so a Tier1
   model that hallucinates can't get its output past the
   schema. The 2026-05-01 closed-banner guard is the example:
   `closed_evidence: Optional[str]` validated as a verbatim
   substring of `scrape.job_content`, plus a
   `_strip_closed_banner_prefix` sanitizer on the persisted
   `description`. Untrusted LLM-rendered prefixes never reach
   downstream consumers.
3. **Periodic enhancer pass** — schedule the
   `scrape-profile-enhancer` subagent (via CronCreate or the
   /loop autonomous mode) to scan recent failures, read its own
   memory, and apply learnings as profile mutations on a cadence.
   The memory remains the source of truth for *why*; the cron is
   the bridge that makes it production-effective.

When you find yourself writing memory text like "in the future, the
extractor should …" — stop and ask which of (1)/(2)/(3) is the right
home. Memory entries that don't have a corresponding production
landing site decay into folklore.

## Tests

```bash
uv run pytest tests/
```
