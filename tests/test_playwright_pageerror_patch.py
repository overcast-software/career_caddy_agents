"""CC-161 regression — the vendored Playwright driver must not crash on a
Firefox pageError with an undefined ``location``.

Playwright 1.60.0's Node driver (``coreBundle.js``) reads
``pageError.location.url`` unconditionally. Firefox / Camoufox can emit a page
error with no ``location`` (cross-origin "Script error.", CSP, some uncaught
errors) — LinkedIn does this — and the unguarded read throws a synchronous
``TypeError`` that kills the whole driver process. Every subsequent Playwright
call then returns "Connection closed while reading from the driver" (the
CC-141 / CC-160 "driver death"). The upstream guard (microsoft/playwright#41629)
is unreleased; camoufox's juggler-side fix (daijro/camoufox#625) is likewise
unreleased on PyPI. So ``scripts/patch_playwright_pageerror.py`` surgically
guards the vendored bundle, run after every ``uv sync`` and in the Docker build.

These tests assert (a) the guard shape-matches and is idempotent at the string
level and (b) the *installed* bundle actually carries the guard, so a fresh
checkout / reinstall that forgot to run the patch fails CI here rather than in
prod against a LinkedIn scrape.
"""

import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PATCH_PATH = _REPO_ROOT / "scripts" / "patch_playwright_pageerror.py"


def _load_patch_module():
    spec = importlib.util.spec_from_file_location(
        "patch_playwright_pageerror", _PATCH_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


patch_mod = _load_patch_module()


# --- string-level shape + idempotency (no Playwright required) --------------


def test_patch_guards_unguarded_snippet():
    """An unguarded bundle is rewritten to optional-chaining + null default."""
    text = "before\n" + patch_mod.UNGUARDED + "after\n"
    patched, count = patch_mod.patch_text(text)
    assert count == 1
    assert patch_mod.UNGUARDED not in patched
    assert patch_mod.GUARDED in patched
    # optional chaining short-circuits on null/undefined instead of throwing
    assert "pageError.location?.url ?? null" in patched


def test_patch_handles_both_crash_sites():
    """Both the dispatcher and the trace-event writer copies get guarded."""
    text = patch_mod.UNGUARDED + "\n\n" + patch_mod.UNGUARDED
    patched, count = patch_mod.patch_text(text)
    assert count == 2
    assert patch_mod.UNGUARDED not in patched
    assert patched.count(patch_mod.GUARDED) == 2


def test_patch_is_idempotent():
    """Re-patching an already-guarded bundle is a no-op, not a double-edit."""
    once, _ = patch_mod.patch_text(patch_mod.UNGUARDED)
    twice, count = patch_mod.patch_text(once)
    assert count == 0
    assert twice == once


def test_patch_fails_loudly_on_unknown_shape():
    """If neither the guarded nor unguarded snippet is present, raise — the
    upstream code shape changed and the patch must be revisited rather than
    silently no-op'ing and leaving the driver crash live."""
    with pytest.raises(patch_mod.PatchError):
        patch_mod.patch_text("some unrelated coreBundle contents\n")


def test_guarded_snippet_preserves_present_location():
    """When ``location`` IS present, the guarded expression yields the same
    fields (optional chaining only short-circuits on null/undefined)."""
    # Structural assertion: the guard reads url/line/column off location and
    # only substitutes null when location is nullish.
    assert "pageError.location?.url ?? null" in patch_mod.GUARDED
    assert "pageError.location?.lineNumber ?? null" in patch_mod.GUARDED
    assert "pageError.location?.columnNumber ?? null" in patch_mod.GUARDED


# --- installed-bundle assertion (requires Playwright present) ---------------


def test_installed_bundle_is_guarded():
    """The Playwright bundle actually installed in this environment must carry
    the guard. Fails if a checkout/reinstall skipped the post-sync patch."""
    try:
        bundle = patch_mod.find_core_bundle()
    except patch_mod.PatchError as exc:
        pytest.skip(f"playwright bundle unavailable: {exc}")

    text = bundle.read_text(encoding="utf-8")
    assert patch_mod.UNGUARDED not in text, (
        "coreBundle.js still contains the UNGUARDED pageError.location read — "
        "run `uv run python scripts/patch_playwright_pageerror.py` after uv sync"
    )
    assert patch_mod.GUARDED in text
