"""FastMCP middleware that hides staff-only tools from non-staff clients.

The api enforces authorization on every endpoint these tools call
through, so a non-staff user invoking `inspect_scrape_html(...)` for
someone else's scrape gets the same 404 they'd get from `get_scrapes`.
Hiding the tools from `tools/list` is a UX / prompt-context concern,
not a security boundary — regular users' LLMs shouldn't waste tokens
on tool definitions they can't usefully invoke.

Three enhancer tools are gated:
  - inspect_scrape_html       — BS4 inspector for scrape.html
  - test_url_rewrite           — dry-run ScrapeProfile url_rewrites
  - find_selectors_for_text    — ranked stable selectors for design

A regular user's `tools/list` returns 28 tools; a staff user's
returns 31. The tools/call surface is NOT gated here — the api gate
is the authoritative one, and double-gating would mean two error
paths for the same condition (api 403 vs middleware 403). If a
non-staff user knows the tool name and calls it anyway, the api
either lets it through (test_url_rewrite reads shared profile data)
or 404s (scrape they don't own). That's already correct behavior.
"""
from __future__ import annotations

from fastmcp.server.dependencies import get_access_token
from fastmcp.server.middleware import Middleware


# Tools hidden from non-staff `tools/list` responses. The set is
# intentionally enumerated here (not derived from a tag) so a future
# reader can grep `staff_tool_filter.py` for "which tools are gated"
# and get a complete answer.
STAFF_ONLY_TOOLS: frozenset[str] = frozenset({
    "inspect_scrape_html",
    "test_url_rewrite",
    "find_selectors_for_text",
})


class StaffOnlyToolFilter(Middleware):
    """Filters STAFF_ONLY_TOOLS out of `tools/list` for non-staff sessions.

    Reads `is_staff` from the current AccessToken's claims (set by
    `ApiKeyTokenVerifier` in `public_server.py` from `/me/`'s
    `is_staff` attribute). Sessions with no AccessToken — direct
    `python public_server.py` smoke tests, unauthenticated probes —
    are treated as non-staff, which is the safe default.
    """

    async def on_list_tools(self, context, call_next):
        tools = await call_next(context)
        if _caller_is_staff():
            return tools
        return [t for t in tools if t.name not in STAFF_ONLY_TOOLS]


def _caller_is_staff() -> bool:
    """Return True iff the current session's claims report is_staff.

    Defensive: any failure to read the access token defaults to False
    (non-staff). FastMCP raises when there's no active request scope,
    which is the case in unit tests that exercise `server.list_tools()`
    directly — those tests assert the unfiltered surface.
    """
    try:
        access = get_access_token()
    except Exception:
        return False
    if access is None:
        return False
    return bool(access.claims.get("is_staff"))
