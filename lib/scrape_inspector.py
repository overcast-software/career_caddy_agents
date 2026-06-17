"""HTML inspector primitives for the scrape-profile-enhancer MCP tools.

The enhancer's job is to fill out ScrapeProfile fields (ready_selector,
apply_link_selectors, url_rewrites, etc.) by reasoning about real
captured HTML from a scrape. Without these primitives the enhancer
either fetches raw HTML (often 200KB+ of noise) and burns tokens
trying to find anchors, or makes selector decisions blind and lets
production runs validate them.

Three primitives live here, each consumed by exactly one MCP tool in
public_server.py:
  - `trim_html` / `extract_skeleton` — server-side BS4 prune so the
    enhancer never sees `<script>`, `<style>`, comments, tracking
    pixels, inline event handlers.
  - `query_selector` — run a CSS selector and return matches with a
    walkable parent-chain outline.
  - `find_selectors_for_text` — given a text fragment, propose ranked
    stable selectors that anchor it.

All three are pure functions over a string of HTML — no network, no
db, no Playwright. The MCP tool layer is responsible for fetching the
scrape row from the api and passing in `scrape.html`. Keeps unit
tests simple and lets the same helpers be reused outside MCP (e.g.
from a maintenance script or test fixture).
"""
from __future__ import annotations

from typing import Iterable, Optional

from bs4 import BeautifulSoup, Comment, NavigableString, Tag


# Tags that carry no useful selector / text information for the
# enhancer's job. Stripped wholesale before any output.
_NOISE_TAGS = ("script", "style", "noscript", "svg", "iframe", "template")

# Attributes that exist purely for tracking / analytics and clutter
# the selector landscape without contributing stability. Element-level
# inline event handlers also dropped because they bloat output and
# never anchor a selector.
_NOISE_ATTR_PREFIXES = ("on",)  # onclick, onload, onerror, ...
_NOISE_ATTRS = {
    "data-tracking",
    "data-tracking-control-name",
    "data-tracking-info",
    "data-analytics",
    "data-analytics-event",
    "data-li-fields",
    "data-impression-id",
    "data-gtm-id",
    "data-ga-event",
    "data-sentry-component",
    "data-sentry-element",
    "data-sentry-source-file",
}

# Attribute name prefixes that should be preserved verbatim — these
# are the load-bearing selector anchors the enhancer cares about.
_KEEP_ATTR_PREFIXES = ("aria-", "data-testid", "data-test-", "role")

# Hard ceiling on returned HTML size. Callers can pass a smaller limit
# but never larger — protects the MCP transport + the consuming LLM's
# context budget. 40_000 chars ≈ 10K tokens upper bound.
_MAX_OUTPUT_CHARS = 40_000


def _clean_attrs(tag: Tag) -> None:
    """Strip noise attributes from a tag in-place.

    Whitelist trumps blacklist: `aria-*` / `data-testid*` / `role`
    are always preserved because the enhancer leans on them for
    stable selectors. Inline event handlers and the per-host
    tracking attrs go.
    """
    if not tag.attrs:
        return
    to_drop = []
    for name in list(tag.attrs.keys()):
        lname = name.lower()
        if any(lname.startswith(p) for p in _KEEP_ATTR_PREFIXES):
            continue
        if name in _NOISE_ATTRS:
            to_drop.append(name)
            continue
        if any(lname.startswith(p) for p in _NOISE_ATTR_PREFIXES):
            to_drop.append(name)
            continue
    for name in to_drop:
        del tag.attrs[name]


def _strip_noise(soup: BeautifulSoup) -> None:
    """Remove `<script>` / `<style>` / comments / tracking pixels."""
    for tag in soup.find_all(_NOISE_TAGS):
        tag.decompose()
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()
    # Tracking pixel pattern: <img width=1 height=1> or display:none images.
    for img in soup.find_all("img"):
        w = img.get("width", "").strip()
        h = img.get("height", "").strip()
        if w in ("1", "0") and h in ("1", "0"):
            img.decompose()


def trim_html(html: str, limit_chars: int = _MAX_OUTPUT_CHARS) -> str:
    """Return `html` with noise tags / attrs / comments stripped.

    Idempotent. Output is truncated at the requested character limit
    (default `_MAX_OUTPUT_CHARS`); truncation appends a sentinel so
    the consuming LLM can tell it's incomplete.
    """
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    _strip_noise(soup)
    for tag in soup.find_all(True):
        _clean_attrs(tag)
    out = str(soup)
    limit = min(max(limit_chars, 1_000), _MAX_OUTPUT_CHARS)
    if len(out) > limit:
        return out[:limit] + f"\n<!-- ... truncated at {limit} chars -->"
    return out


def extract_skeleton(html: str, limit_chars: int = _MAX_OUTPUT_CHARS) -> str:
    """Return the tag+class+id structure of `html` with no text content.

    For first-pass orientation when the captured HTML is huge — the
    enhancer can see the document shape without paying for every
    paragraph of job description. Text-only NavigableStrings drop;
    tags keep their attrs (post-`_clean_attrs`).
    """
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    _strip_noise(soup)
    for tag in soup.find_all(True):
        _clean_attrs(tag)
    for text in soup.find_all(string=True):
        if isinstance(text, NavigableString) and not isinstance(text, Comment):
            text.extract()
    out = str(soup)
    limit = min(max(limit_chars, 1_000), _MAX_OUTPUT_CHARS)
    if len(out) > limit:
        return out[:limit] + f"\n<!-- ... truncated at {limit} chars -->"
    return out


def _element_id(tag: Tag) -> str:
    """Build a short readable id for a tag: tag#id.class1.class2."""
    bits = [tag.name]
    if tag.get("id"):
        bits.append(f"#{tag.get('id')}")
    classes = tag.get("class")
    if classes:
        # First two classes only — keeps output scannable.
        bits.append("." + ".".join(classes[:2]))
    return "".join(bits)


def _outline_path(tag: Tag, max_depth: int = 6) -> str:
    """Render the parent chain for `tag` as a `>`-joined outline.

    Walks up at most `max_depth` ancestors. The enhancer uses this to
    judge selector stability — a match that lives inside an `<article
    class="jobs-description">` is more stable than one nested under
    six anonymous divs.
    """
    chain: list[str] = []
    cur = tag
    depth = 0
    while cur is not None and isinstance(cur, Tag) and depth < max_depth:
        chain.append(_element_id(cur))
        cur = cur.parent
        depth += 1
    return " > ".join(reversed(chain))


def _text_snippet(tag: Tag, max_chars: int = 160) -> str:
    """Compact single-line text summary of a tag's contents."""
    text = " ".join(tag.get_text(" ", strip=True).split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"


def query_selector(
    html: str,
    selector: str,
    *,
    max_matches: int = 25,
) -> dict:
    """Run a CSS selector against `html`; return matches + diagnostics.

    Output shape:
      {
        "selector": "...",
        "match_count": int,
        "matches": [
          {"outline": "html > body > main > h2.foo",
           "text_snippet": "About the job",
           "attrs": {"class": ["foo", "bar"], "data-testid": "..."}},
          ...
        ],
      }

    `max_matches` caps the returned list — the count tells the enhancer
    whether the selector is broad or narrow. BS4's `.select()` is used
    (standard CSS3 only), so Playwright-engine pseudo-selectors like
    `:has-text()` will raise; let the caller (MCP tool) translate that
    into a friendly error.
    """
    if not html:
        return {"selector": selector, "match_count": 0, "matches": []}
    soup = BeautifulSoup(html, "html.parser")
    _strip_noise(soup)
    hits = soup.select(selector)
    matches = []
    for tag in hits[: max(1, max_matches)]:
        _clean_attrs(tag)
        matches.append({
            "outline": _outline_path(tag),
            "text_snippet": _text_snippet(tag),
            "attrs": {k: v for k, v in tag.attrs.items()},
        })
    return {
        "selector": selector,
        "match_count": len(hits),
        "matches": matches,
    }


# ParsedJobData field → the css_selectors.job_data keys that can supply
# it. Selector discovery (browser_server._discover_job_selectors) keys
# the company field as "company_name"; the browser extension's
# captured_payload keys it as "company". Accept either so a profile
# authored by either side resolves to the ParsedJobData "company_name"
# the extract tiers read.
_JOB_DATA_FIELD_SOURCES: dict[str, tuple[str, ...]] = {
    "title": ("title",),
    "company_name": ("company_name", "company"),
    "description": ("description",),
    "location": ("location",),
}


def css_extract_job_data(html: str, selectors: dict) -> dict:
    """Deterministically extract job fields from `html` using a per-host
    ``css_selectors.job_data`` selector map (field name → CSS3 selector).

    Returns a dict in ParsedJobData vocabulary —
    ``{"title", "company_name", "description", "location"}`` — where each
    value is the collapsed visible text of the first matching element, or
    ``""`` when the selector key is absent or matches nothing.

    This is the Tier-0 ($0, no-LLM) extraction primitive: when a domain's
    profile already carries graduated ``job_data`` selectors, the
    scrape-graph parses the captured HTML here instead of paying for an
    LLM tier.

    Pure function over a string of HTML — no network, no Playwright. CSS3
    only (BS4 ``.select_one``), the same dialect the browser extension
    used to discover these selectors; Playwright pseudo-classes such as
    ``:has-text()`` are unsupported and raise, in which case that field
    yields ``""`` rather than crashing the caller.
    """
    out = {"title": "", "company_name": "", "description": "", "location": ""}
    if not html or not isinstance(selectors, dict):
        return out
    soup = BeautifulSoup(html, "html.parser")
    _strip_noise(soup)
    for field, source_keys in _JOB_DATA_FIELD_SOURCES.items():
        selector = next(
            (
                selectors[k]
                for k in source_keys
                if isinstance(selectors.get(k), str) and selectors[k].strip()
            ),
            None,
        )
        if not selector:
            continue
        try:
            match = soup.select_one(selector)
        except Exception:
            # Bad/Playwright-only selector — skip the field, don't crash.
            continue
        if match is not None:
            out[field] = " ".join(match.get_text(" ", strip=True).split())
    return out


def _selector_candidates_for(tag: Tag) -> Iterable[tuple[int, str]]:
    """Yield (stability_score, selector) pairs anchoring `tag`.

    Higher score = more stable. The enhancer chooses among them with
    full context; this function never picks for it.

    Scoring rationale (relative, not absolute):
      90 — `[data-testid=...]`               # explicitly stable by design
      85 — `[role=...]`                      # ARIA semantic anchor
      80 — `[aria-label=...]`                # ARIA text anchor
      70 — `tag#id`                          # only when id looks stable (no digits at the end suggesting hash)
      60 — `tag.semantic-class`              # short class with letters only
      40 — `tag.class1.class2`               # multi-class composite
      20 — fallback `tag` with parent chain
    """
    candidates: list[tuple[int, str]] = []

    if tag.get("data-testid"):
        candidates.append((90, f'[data-testid="{tag["data-testid"]}"]'))
    role = tag.get("role")
    if role:
        candidates.append((85, f'{tag.name}[role="{role}"]'))
    aria_label = tag.get("aria-label")
    if aria_label and len(aria_label) < 80:
        candidates.append((80, f'{tag.name}[aria-label="{aria_label}"]'))

    el_id = tag.get("id")
    if el_id and not _looks_hashed(el_id):
        candidates.append((70, f"{tag.name}#{el_id}"))

    classes = tag.get("class") or []
    stable_classes = [c for c in classes if _looks_semantic_class(c)]
    if len(stable_classes) == 1:
        candidates.append((60, f"{tag.name}.{stable_classes[0]}"))
    elif len(stable_classes) >= 2:
        joined = ".".join(stable_classes[:2])
        candidates.append((40, f"{tag.name}.{joined}"))

    if not candidates:
        candidates.append((20, tag.name))

    return candidates


def _looks_hashed(value: str) -> bool:
    """Heuristic: does this id/class look like a build-time hash?

    Hashed ids (Tailwind JIT, CSS-in-JS) typically end in 5+ hex
    chars or contain `_` plus a random suffix. Those churn on every
    deploy and don't anchor stable selectors.
    """
    if "_" in value and any(c.isdigit() for c in value.rsplit("_", 1)[-1]):
        return True
    tail = value[-6:]
    return tail.isalnum() and any(c.isdigit() for c in tail) and any(c.isalpha() for c in tail)


def _looks_semantic_class(cls: str) -> bool:
    """Filter heuristic: prefer human-named classes over hashed atoms."""
    if not cls or len(cls) < 3 or len(cls) > 60:
        return False
    if _looks_hashed(cls):
        return False
    # Tailwind atomic classes (`p-4`, `text-sm`, `md:grid-cols-3`) — useful
    # for layout but rarely identity. Skip when the class is dominated by
    # tailwind-shaped tokens.
    if any(ch in cls for ch in ":[]"):
        return False
    return True


def find_selectors_for_text(
    html: str,
    text: str,
    *,
    max_results: int = 10,
    case_insensitive: bool = True,
) -> dict:
    """Return ranked selectors anchoring `text` within `html`.

    Output shape:
      {
        "text": "...",
        "match_count": int,
        "candidates": [
          {"selector": "h2[aria-label=\"About the job\"]",
           "stability": 80,
           "outline": "html > body > main > article.jobs-description__container > h2",
           "text_snippet": "About the job"},
          ...
        ],
      }

    For each tag whose direct text contains `text`, the function emits
    every candidate selector from `_selector_candidates_for` plus
    selectors against the nearest semantic ancestor (so the enhancer
    sees both "anchor on the matching node" and "anchor on its
    nearest stable wrapper"). Deduped by selector string; top
    `max_results` returned, sorted by stability descending.
    """
    if not html or not text:
        return {"text": text, "match_count": 0, "candidates": []}
    soup = BeautifulSoup(html, "html.parser")
    _strip_noise(soup)
    needle = text.casefold() if case_insensitive else text

    def matches(s: str) -> bool:
        target = s.casefold() if case_insensitive else s
        return needle in target

    hit_tags: list[Tag] = []
    for node in soup.find_all(string=True):
        if isinstance(node, Comment):
            continue
        if not matches(str(node)):
            continue
        parent = node.parent
        if isinstance(parent, Tag):
            hit_tags.append(parent)

    seen: dict[str, dict] = {}
    for tag in hit_tags:
        text_snippet = _text_snippet(tag)
        outline = _outline_path(tag)
        for score, sel in _selector_candidates_for(tag):
            if sel in seen:
                # Keep the higher score if we see it via multiple tags.
                if score > seen[sel]["stability"]:
                    seen[sel]["stability"] = score
                continue
            seen[sel] = {
                "selector": sel,
                "stability": score,
                "outline": outline,
                "text_snippet": text_snippet,
            }

    ranked = sorted(
        seen.values(),
        key=lambda c: (-c["stability"], len(c["selector"])),
    )
    return {
        "text": text,
        "match_count": len(hit_tags),
        "candidates": ranked[: max(1, max_results)],
    }


def derive_hostname(url: Optional[str]) -> Optional[str]:
    """Return the lowercased registrable host of `url`, stripping `www.`.

    Mirrors the lookup convention `_profile_url_rewrites_for_host` uses
    on the api side: a profile keyed `linkedin.com` matches a URL on
    `www.linkedin.com`. Returns None for falsy / unparseable input.
    """
    if not url:
        return None
    from urllib.parse import urlparse
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return None
    if host.startswith("www."):
        host = host[4:]
    return host or None
