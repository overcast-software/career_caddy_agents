#!/usr/bin/env python3
"""Verify whether Camoufox renders window.open()-spawned pages as tabs.

Run:
    uv run python tools/camoufox_tab_spike.py

Then count the windows on screen. Three runs happen back-to-back:

  Phase A — default Camoufox prefs, three pages via ctx.new_page().
            Expected baseline: 4 windows (1 anchor + 3 pages).

  Phase B — default Camoufox prefs, three pages via window.open() from
            an anchor page. If Camoufox honors stock Firefox defaults
            (browser.link.open_newwindow=3), this should land as 1
            window with 4 tabs.

  Phase C — explicit firefox_user_prefs override forcing tabs. Belt-and-
            suspenders in case Camoufox unsets the default.

Each phase pauses 20 seconds so you can count what's on screen, then
prints len(context.pages) before tearing down.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from camoufox.async_api import AsyncCamoufox

TARGETS = ["https://example.com/", "https://example.org/", "https://example.net/"]


async def _phase(label: str, prefs: dict | None, use_window_open: bool, pause_s: int):
    print(f"\n=== {label} ===")
    kwargs: dict = {"headless": False}
    if prefs is not None:
        kwargs["firefox_user_prefs"] = prefs
    async with AsyncCamoufox(**kwargs) as browser:
        # AsyncCamoufox yields a Browser-like object; first context is
        # already there for persistent launches, otherwise create one.
        ctx = browser.contexts[0] if browser.contexts else await browser.new_context()

        anchor = await ctx.new_page()
        await anchor.goto("about:blank")

        for url in TARGETS:
            if use_window_open:
                async with ctx.expect_page() as new_page_info:
                    await anchor.evaluate(f"window.open({url!r}, '_blank')")
                page = await new_page_info.value
                try:
                    await page.wait_for_load_state("domcontentloaded", timeout=10_000)
                except Exception:
                    pass
            else:
                page = await ctx.new_page()
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=10_000)
                except Exception:
                    pass

        print(f"context.pages count: {len(ctx.pages)}")
        print(f"Pausing {pause_s}s — count the WINDOWS on screen.")
        await asyncio.sleep(pause_s)


PHASES = {
    "A": ("Phase A: baseline — ctx.new_page() x3 (default prefs)", None, False),
    "B": ("Phase B: anchor + window.open() x3 (default prefs)", None, True),
    "C": (
        "Phase C: anchor + window.open() x3 (explicit tab pref)",
        {"browser.link.open_newwindow": 3, "browser.link.open_newwindow.restriction": 0},
        True,
    ),
}


async def main():
    parser = argparse.ArgumentParser(description="Camoufox tab vs window spike.")
    parser.add_argument(
        "--phase", choices=["A", "B", "C", "all"], default="all",
        help="Which phase to run (default: all).",
    )
    parser.add_argument(
        "--pause", type=int, default=None,
        help="Seconds to hold each phase open. Default: 20s for all, 60s for single-phase.",
    )
    args = parser.parse_args()

    selected = ["A", "B", "C"] if args.phase == "all" else [args.phase]
    pause_s = args.pause if args.pause is not None else (20 if args.phase == "all" else 60)

    for key in selected:
        label, prefs, use_window_open = PHASES[key]
        await _phase(label, prefs, use_window_open, pause_s)
    print(f"\nDone. Report: windows seen in {' / '.join(selected)}.")


if __name__ == "__main__":
    asyncio.run(main())
