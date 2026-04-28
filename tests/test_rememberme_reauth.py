"""_try_rememberme_reauth is now profile-driven: site-specific selectors
live in ScrapeProfile.css_selectors and are passed in as
`profile_candidates`. The probation-graduated single is passed as
`graduated_selector`. The helper trusts the operator's selectors —
no in-code text/aria gates — so a host with a well-curated profile
unblocks rememberme without LLM help, and a fresh host with no
profile data falls through fast (no false positives from generic
heuristics).
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from mcp_servers.browser_server import _try_rememberme_reauth


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_element(text: str = "", aria_label: str | None = None):
    el = MagicMock()
    el.inner_text = AsyncMock(return_value=text)
    el.get_attribute = AsyncMock(return_value=aria_label)
    el.click = AsyncMock()
    return el


def _make_page(matches: dict):
    """`matches` maps selector → element (or omit for no-match)."""
    page = MagicMock()
    page.query_selector = AsyncMock(side_effect=lambda sel: matches.get(sel))
    page.wait_for_load_state = AsyncMock()
    return page


def test_clicks_first_profile_candidate_that_matches():
    """LinkedIn /uas/login rememberme tile case. The seeded profile lists
    .member-profile__details first; that's the click."""
    el = _make_element(
        text="Doug Headley\nl*****@passiveobserver.com",
        aria_label="Login as Doug Headley",
    )
    page = _make_page({"button.member-profile__details": el})
    candidates = [
        "button.member-profile__details",
        "button[aria-label^='Login as']",
        "button:has-text('Continue as')",
    ]
    assert _run(_try_rememberme_reauth(page, profile_candidates=candidates)) is True
    el.click.assert_called_once()


def test_falls_through_to_second_candidate_when_first_misses():
    """First selector doesn't match the page; second one does. The
    helper iterates the list in order."""
    el = _make_element(text="Continue as Doug")
    page = _make_page({"button:has-text('Continue as')": el})
    candidates = [
        "button.member-profile__details",
        "button:has-text('Continue as')",
    ]
    assert _run(_try_rememberme_reauth(page, profile_candidates=candidates)) is True
    el.click.assert_called_once()


def test_graduated_selector_runs_before_profile_candidates():
    """The probation-graduated selector is the host's "best known"
    answer — try it before the broader candidate list."""
    graduated_el = _make_element(text="Login", aria_label="Login as X")
    candidate_el = _make_element(text="Login", aria_label="Login as X")
    page = _make_page({
        "div.graduated": graduated_el,
        "button.member-profile__details": candidate_el,
    })
    assert _run(
        _try_rememberme_reauth(
            page,
            profile_candidates=["button.member-profile__details"],
            graduated_selector="div.graduated",
        )
    ) is True
    graduated_el.click.assert_called_once()
    candidate_el.click.assert_not_called()


def test_returns_false_with_no_profile_data():
    """Architectural contract: a host with no profile candidates and no
    graduated selector gets a fast False. No in-code heuristics."""
    page = _make_page({})
    assert _run(_try_rememberme_reauth(page)) is False


def test_returns_false_when_candidates_dont_match_page():
    """Profile lists selectors but none of them are on the current
    page — fall through, let the caller route to ObstacleAgent."""
    page = _make_page({})
    candidates = [
        "button.member-profile__details",
        "button[aria-label^='Login as']",
    ]
    assert _run(_try_rememberme_reauth(page, profile_candidates=candidates)) is False


def test_no_text_or_aria_gate_trust_the_operator():
    """A button matching a profile-supplied selector with unrelated
    visible text and no aria-label is STILL clicked. The operator
    chose the selector; we trust it. Pre-refactor the legacy in-code
    list paired every selector with a 'continue' text gate to defend
    against generic catch-alls; profile-driven config doesn't need
    that defense."""
    el = _make_element(text="Get the app", aria_label=None)
    page = _make_page({"div.unusual-but-correct-tile": el})
    assert _run(
        _try_rememberme_reauth(
            page,
            profile_candidates=["div.unusual-but-correct-tile"],
        )
    ) is True
    el.click.assert_called_once()
