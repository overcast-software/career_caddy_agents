#!/usr/bin/env python3
"""Download the Playwright Chromium binary (equivalent of `python -m camoufox fetch`)."""

import subprocess
import sys


def main():
    subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)


if __name__ == "__main__":
    main()
