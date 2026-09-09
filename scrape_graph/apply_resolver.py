"""Apply-destination resolver — pure logic split out of the node so it
is testable without pydantic-graph plumbing.

Three strategies, tried in order against a profile config dict:

1. ``internal_apply_markers`` — selectors that, if present, mean the
   posting uses an internal apply flow (LinkedIn Easy Apply etc.).
   Result: status=``internal``, no navigation, ``apply_url=None``.

2. ``apply_link_selectors`` — selectors whose first match has an
   ``href`` we can read directly. No click, no navigation; cheapest
   path and immune to anti-bot. Result: status=``resolved``,
   ``apply_url=<href>``.

3. ``apply_button_selectors`` — JS-driven buttons. Click in a way that
   captures both same-page navigation and new-tab opens; record the
   landing URL. Slowest and most fragile. Result: status=``resolved``
   on success, ``failed`` on any exception/timeout.

Missing ``state._browser_page`` (chat/paste/email path) → status=
``unknown``. Missing/empty config → status=``unknown``.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_NAV_TIMEOUT_MS = 10_000


async def resolve_apply_url(page: Any, config: Optional[dict]) -> dict:
    """Return ``{"apply_url": str|None, "apply_url_status": str,
    "reason": str}``. Pure: caller persists the result.
    """
    if page is None:
        return {
            "apply_url": None,
            "apply_url_status": "unknown",
            "reason": "no_browser_page",
        }
    if not config or not isinstance(config, dict):
        return {
            "apply_url": None,
            "apply_url_status": "unknown",
            "reason": "no_config",
        }

    try:
        marker = await _first_visible_match(page, config.get("internal_apply_markers") or [])
        if marker:
            return {
                "apply_url": None,
                "apply_url_status": "internal",
                "reason": f"internal_marker: {marker}",
            }

        href, sel = await _first_link_href(page, config.get("apply_link_selectors") or [])
        if href:
            return {
                "apply_url": href,
                "apply_url_status": "resolved",
                "reason": f"link_selector: {sel}",
            }

        for selector in config.get("apply_button_selectors") or []:
            url = await _click_and_capture_url(page, selector)
            if url:
                return {
                    "apply_url": url,
                    "apply_url_status": "resolved",
                    "reason": f"button_selector: {selector}",
                }

        if (
            config.get("apply_button_selectors")
            or config.get("apply_link_selectors")
        ):
            # Profile told us to look but nothing matched / nothing landed.
            return {
                "apply_url": None,
                "apply_url_status": "failed",
                "reason": "no_selector_matched",
            }
        return {
            "apply_url": None,
            "apply_url_status": "unknown",
            "reason": "config_empty",
        }
    except Exception as exc:
        logger.warning("resolve_apply_url crashed: %s", exc, exc_info=True)
        return {
            "apply_url": None,
            "apply_url_status": "failed",
            "reason": f"exception: {type(exc).__name__}",
        }


async def _first_visible_match(page: Any, selectors: list[str]) -> Optional[str]:
    for sel in selectors:
        try:
            handle = await page.query_selector(sel)
            if handle is not None:
                return sel
        except Exception:
            continue
    return None


async def _first_link_href(
    page: Any, selectors: list[str]
) -> tuple[Optional[str], Optional[str]]:
    for sel in selectors:
        try:
            handle = await page.query_selector(sel)
            if handle is None:
                continue
            href = await handle.get_attribute("href")
            if href and href.startswith(("http://", "https://")):
                return href, sel
        except Exception:
            continue
    return None, None


_CANDIDATE_SCAN_JS = r"""
(() => {
  const isApplyish = (s) => /\bapply\b/i.test(s || "");
  const buildSelector = (el) => {
    const tag = el.tagName.toLowerCase();
    if (el.id && /^[A-Za-z][\w-]*$/.test(el.id)) {
      return `#${el.id}`;
    }
    const classes = (el.className && typeof el.className === "string"
      ? el.className.split(/\s+/) : [])
      .filter((c) => c && /^[A-Za-z][\w-]*$/.test(c))
      .slice(0, 3);
    const dataAttrs = Array.from(el.attributes || [])
      .filter((a) => a.name && a.name.startsWith("data-"))
      .slice(0, 2)
      .map((a) => `[${a.name}="${(a.value || "").slice(0, 32)}"]`)
      .join("");
    return tag + classes.map((c) => `.${c}`).join("") + dataAttrs;
  };
  // Pass 3 gate: a tagless element only counts as a control if it LOOKS
  // clickable. Attribute checks first — getComputedStyle forces style
  // resolution and is the expensive one, so it goes last.
  const clickAffordance = (el) => {
    if (el.hasAttribute("onclick") || typeof el.onclick === "function") {
      return "onclick handler";
    }
    const ti = el.getAttribute("tabindex");
    if (ti !== null && ti !== "-1") { return "tabindex"; }
    try {
      const cs = window.getComputedStyle(el);
      if (cs && cs.cursor === "pointer") { return "cursor: pointer"; }
    } catch (e) { /* detached / cross-origin — no affordance signal */ }
    return null;
  };
  const out = [];
  const seen = new Set();
  for (const a of document.querySelectorAll("a[href]")) {
    seen.add(a);
    const href = a.href || a.getAttribute("href") || "";
    const text = (a.innerText || "").trim().slice(0, 100);
    const aria = a.getAttribute("aria-label") || "";
    let score = 0;
    const reasons = [];
    if (isApplyish(href)) { score += 0.5; reasons.push("href contains 'apply'"); }
    if (isApplyish(text)) { score += 0.4; reasons.push("text 'apply'"); }
    if (isApplyish(aria)) { score += 0.3; reasons.push("aria-label 'apply'"); }
    if (score === 0) continue;
    out.push({
      selector: buildSelector(a),
      href: href.startsWith("http") ? href : null,
      text: text,
      tag: "a",
      score: Math.min(score, 1),
      reason: reasons.join(" AND "),
    });
  }
  for (const b of document.querySelectorAll('button, [role="button"]')) {
    seen.add(b);
    const text = (b.innerText || "").trim().slice(0, 100);
    const aria = b.getAttribute("aria-label") || "";
    let score = 0;
    const reasons = [];
    if (isApplyish(text)) { score += 0.4; reasons.push("text 'apply'"); }
    if (isApplyish(aria)) { score += 0.3; reasons.push("aria-label 'apply'"); }
    if (score === 0) continue;
    out.push({
      selector: buildSelector(b),
      href: null,
      text: text,
      tag: b.tagName.toLowerCase(),
      score: Math.min(score, 1),
      reason: reasons.join(" AND "),
    });
  }
  // Pass 3 — div/span-rendered controls (CC-285). jobright.ai's
  // "Apply on Employer Site", the only control that leads to the real
  // ATS, is a plain div and is invisible to both passes above; the two
  // buttons that ARE visible are decoys. Gated on applyish text AND a
  // click affordance so a text-heavy page cannot flood the list, and
  // capped independently of max_candidates.
  const EXTRA_CAP = 15;
  const MAX_TEXT = 120;
  let extra = 0;
  for (const el of document.querySelectorAll('div, span, a:not([href])')) {
    if (extra >= EXTRA_CAP) break;
    if (seen.has(el)) continue;
    const full = (el.innerText || "").trim();
    const text = full.slice(0, 100);
    const aria = el.getAttribute("aria-label") || "";
    if (!isApplyish(text) && !isApplyish(aria)) continue;
    // Cheap containment guard, applied before any subtree walk: a page
    // section is not a control. Everything past here has a small subtree.
    if (full.length > MAX_TEXT) continue;
    // A wrapper whose applyish text comes from a nested element is not
    // itself the control — the inner one is the candidate, not this.
    if (el.querySelector('a[href], button, [role="button"]')) continue;
    let wrapsInner = false;
    for (const d of el.querySelectorAll("*")) {
      if (isApplyish(d.innerText || "")) { wrapsInner = true; break; }
    }
    if (wrapsInner) continue;
    const affordance = clickAffordance(el);
    if (!affordance) continue;
    let score = 0;
    const reasons = [];
    if (isApplyish(text)) { score += 0.4; reasons.push("text 'apply'"); }
    if (isApplyish(aria)) { score += 0.3; reasons.push("aria-label 'apply'"); }
    if (score === 0) continue;
    reasons.push(affordance);
    out.push({
      selector: buildSelector(el),
      href: null,
      text: text,
      tag: el.tagName.toLowerCase(),
      score: Math.min(score, 1),
      reason: reasons.join(" AND "),
    });
    extra += 1;
  }
  return out;
})()
"""


async def scan_apply_candidates(page: Any, max_candidates: int = 50) -> list[dict]:
    """Heuristic scan for "Apply"-shaped elements, used when the configured
    resolver missed (no config, or config didn't match anything on the page).

    Phase 3 learning loop — captured candidates flow back to the api via
    the apply-url PATCH and accumulate on the Scrape row for later
    aggregation per host. Aggregation + promotion into
    ``ScrapeProfile.apply_resolver_config`` is a separate slice; this
    function is capture-only.

    Three passes: ``a[href]``, then ``button, [role="button"]``, then
    (CC-285) ``div, span, a:not([href])`` gated on applyish text AND a
    click affordance and capped at 15, so div-rendered controls are
    visible without a text-heavy page flooding the list.

    Returns a list of ``{"selector", "href", "text", "tag", "score",
    "reason"}`` dicts, sorted by score descending, capped at
    ``max_candidates`` (default 50). ``tag`` is the element's own tag
    name, so a consumer can tell a ``div`` (no href to read) from an
    href-bearing anchor without re-deriving it. Empty list if the page is
    None or the scan crashes — never raises.
    """
    if page is None:
        return []
    try:
        results = await page.evaluate(_CANDIDATE_SCAN_JS)
    except Exception:
        logger.warning("scan_apply_candidates: page.evaluate crashed", exc_info=True)
        return []
    if not isinstance(results, list):
        return []
    cleaned = []
    for r in results:
        if not isinstance(r, dict):
            continue
        if not r.get("selector"):
            continue
        cleaned.append(r)
    cleaned.sort(key=lambda r: float(r.get("score") or 0), reverse=True)
    return cleaned[:max_candidates]


async def _click_and_capture_url(page: Any, selector: str) -> Optional[str]:
    """Click ``selector`` and return the landing URL.

    Handles two outcomes — new tab open (most ATS apply buttons) and
    in-place navigation. Closes any new tab it opens.
    """
    try:
        handle = await page.query_selector(selector)
        if handle is None:
            return None
    except Exception:
        return None

    context = page.context
    new_page = None
    landing_url: Optional[str] = None

    try:
        async with context.expect_page(timeout=_NAV_TIMEOUT_MS) as new_page_info:
            await handle.click()
        new_page = await new_page_info.value
        try:
            await new_page.wait_for_load_state(
                "domcontentloaded", timeout=_NAV_TIMEOUT_MS
            )
        except Exception:
            pass
        landing_url = new_page.url
    except Exception:
        # No new page opened — try same-page navigation fallback.
        before = page.url
        try:
            await page.wait_for_url(
                lambda url: url != before, timeout=_NAV_TIMEOUT_MS
            )
            landing_url = page.url
        except Exception:
            landing_url = None
    finally:
        if new_page is not None:
            try:
                await new_page.close()
            except Exception:
                pass

    if landing_url and landing_url.startswith(("http://", "https://")):
        return landing_url
    return None
