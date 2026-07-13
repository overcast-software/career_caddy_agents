#!/usr/bin/env python3
"""Guard Playwright's Firefox pageError dispatcher against an undefined location.

Why this exists
---------------
Playwright 1.60.0's vendored Node driver reads ``pageError.location.url``
unconditionally when a page emits an error (``coreBundle.js``). Firefox /
Camoufox can emit a page error with **no** ``location`` — a cross-origin
"Script error.", a CSP violation, or some uncaught errors. When that happens
the driver throws ``TypeError: Cannot read properties of undefined (reading
'url')`` synchronously in the Node event loop and the **entire driver process
exits**. Every subsequent Playwright call then returns "Connection closed while
reading from the driver" — the CC-141 / CC-160 "driver death". LinkedIn triggers
it deterministically; a relaunched scrape re-crashes on the same pageError, so
CC-141/160's survive-and-relaunch loop can never complete a LinkedIn scrape.

This is upstream bug microsoft/playwright#39767 (introduced in 1.60.0); the
dispatcher guard is microsoft/playwright#41629 and is NOT present in any
released Playwright (verified: 1.60.0 AND 1.61.0 both ship the unguarded read).
camoufox's own fix (daijro/camoufox#625, juggler-side) is unreleased on PyPI
(latest camoufox 0.4.11 does not contain it). So the only fix that keeps
Camoufox stealth today is to surgically guard the vendored bundle.

The fix
-------
Rewrite the two crash sites so a missing ``location`` yields ``null`` fields
instead of throwing::

    url: pageError.location.url,          -> url: pageError.location?.url ?? null,
    line: pageError.location.lineNumber,  -> line: pageError.location?.lineNumber ?? null,
    column: pageError.location.columnNumber -> column: pageError.location?.columnNumber ?? null,

Normal pageerror behaviour is preserved when ``location`` IS present (optional
chaining short-circuits only on null/undefined).

Contract
--------
* **Idempotent** — running it again after it has patched is a no-op.
* **Shape-matched** — it locates the exact vendored snippet; if the upstream
  code shape changed (e.g. a Playwright bump that finally guards this itself),
  it fails loudly rather than silently corrupting the bundle.
* **Repeatable** — safe to run after every ``uv sync`` / Playwright reinstall.

Wiring
------
* Local dev: run ``uv run python scripts/patch_playwright_pageerror.py`` after
  ``uv sync`` (or ``python -m camoufox fetch``).
* Image build: the agents ``Dockerfile`` invokes this after the final
  ``uv sync`` (see ``RUN uv run python scripts/patch_playwright_pageerror.py``).

Exit codes: 0 = patched or already-patched; 1 = bundle/shape not found (loud).
"""

from __future__ import annotations

import sys
from pathlib import Path

# The exact unguarded snippet Playwright 1.60/1.61 emit at each crash site. Both
# the dispatcher (BrowserContext.Events.PageError) and the trace-event writer
# (_onPageError) contain byte-identical copies, so a single find/replace pair
# covers both occurrences.
UNGUARDED = (
    "              url: pageError.location.url,\n"
    "              line: pageError.location.lineNumber,\n"
    "              column: pageError.location.columnNumber\n"
)

GUARDED = (
    "              url: pageError.location?.url ?? null,\n"
    "              line: pageError.location?.lineNumber ?? null,\n"
    "              column: pageError.location?.columnNumber ?? null\n"
)


class PatchError(RuntimeError):
    """Raised when the bundle cannot be located or its shape is unexpected."""


def find_core_bundle() -> Path:
    """Return the path to the installed Playwright driver ``coreBundle.js``.

    Resolves via the installed ``playwright`` package so it works in the venv,
    the editable install, and the Docker image alike.
    """
    try:
        import playwright
    except ImportError as exc:  # pragma: no cover - import guard
        raise PatchError(
            "playwright is not importable; run this after `uv sync`."
        ) from exc

    pkg_dir = Path(playwright.__file__).resolve().parent
    bundle = pkg_dir / "driver" / "package" / "lib" / "coreBundle.js"
    if not bundle.is_file():
        raise PatchError(f"coreBundle.js not found at expected path: {bundle}")
    return bundle


def patch_text(text: str) -> tuple[str, int]:
    """Return (patched_text, occurrences_patched).

    Idempotent: if the text is already guarded and contains no unguarded
    sites, returns (text, 0) without raising. Raises PatchError only when the
    file is neither guarded nor matches the known unguarded shape — i.e. the
    upstream code changed and this patch needs revisiting.
    """
    unguarded_count = text.count(UNGUARDED)
    guarded_count = text.count(GUARDED)

    if unguarded_count == 0:
        if guarded_count > 0:
            # Already patched — nothing to do.
            return text, 0
        raise PatchError(
            "coreBundle.js contains neither the known-unguarded nor the "
            "guarded pageError.location snippet. Playwright's code shape "
            "changed; re-verify the crash site and update this patch "
            "(upstream: microsoft/playwright#39767 / #41629)."
        )

    return text.replace(UNGUARDED, GUARDED), unguarded_count


def apply(bundle: Path | None = None) -> int:
    """Patch the bundle in place. Returns the number of sites newly patched."""
    path = bundle or find_core_bundle()
    original = path.read_text(encoding="utf-8")
    patched, count = patch_text(original)
    if count and patched != original:
        path.write_text(patched, encoding="utf-8")
    return count


def main() -> int:
    try:
        bundle = find_core_bundle()
        count = apply(bundle)
    except PatchError as exc:
        print(f"[patch_playwright_pageerror] FAILED: {exc}", file=sys.stderr)
        return 1

    if count:
        print(
            f"[patch_playwright_pageerror] guarded {count} pageError.location "
            f"site(s) in {bundle}"
        )
    else:
        print(
            f"[patch_playwright_pageerror] already patched (no-op): {bundle}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
