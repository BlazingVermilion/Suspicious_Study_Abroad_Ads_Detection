#!/usr/bin/env python3
"""Create a Playwright Instagram session file.

The session is saved to secrets/instagram_session.json by default and is ignored
by Git. Run this once manually before crawling Instagram.
"""
from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from playwright.async_api import async_playwright


def parse_args() -> argparse.Namespace:
    root = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
    parser = argparse.ArgumentParser(description="Save Instagram Playwright storage state.")
    parser.add_argument("--output", type=Path, default=root / "secrets" / "instagram_session.json")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        await page.goto("https://www.instagram.com/accounts/login/")
        print("\nLogin manually in the browser window.")
        print("After login is complete, press ENTER here to save the session.")
        input()
        await context.storage_state(path=str(args.output))
        print(f"Session saved to: {args.output}")
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
