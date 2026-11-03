#!/usr/bin/env python3
"""
Navigation observer — opens a browser and records every page you visit.

Run:
    python observe.py

Then navigate manually through the church website. On every page load, the script
automatically:
  - Takes a full-page screenshot  →  logs/observe/<timestamp>_<slug>.png
  - Saves the visible page text   →  logs/observe/<timestamp>_<slug>.txt
  - Prints the current URL to the terminal

Press Ctrl+C when done. Claude can then read the saved files to understand
how the site navigation and forms are structured.
"""

import asyncio
import re
import sys
from datetime import datetime
from pathlib import Path

from playwright.async_api import async_playwright

OBSERVE_DIR = Path("logs/observe")
CHURCH_URL = "https://missionaryrecommendations.churchofjesuschrist.org/recommendations/home/candidates?vctype=ft"


def _slug(url: str) -> str:
    """Turn a URL into a safe filename segment."""
    path = re.sub(r"https?://[^/]+", "", url)
    slug = re.sub(r"[^\w]", "_", path).strip("_")[:60]
    return slug or "root"


async def main():
    OBSERVE_DIR.mkdir(parents=True, exist_ok=True)

    # Try to load stored credentials
    username = ""
    password = ""
    try:
        import keyring
        username = keyring.get_password("missionary-forms", "username") or ""
        password = keyring.get_password("missionary-forms", username) or ""
    except Exception:
        pass

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False, slow_mo=100)
        context = await browser.new_context(viewport={"width": 1280, "height": 900})
        page = await context.new_page()

        captured: list[str] = []

        async def on_load(event_url: str = ""):
            url = page.url
            ts = datetime.now().strftime("%H%M%S")
            slug = _slug(url)
            name = f"{ts}_{slug}"

            # Screenshot
            img_path = OBSERVE_DIR / f"{name}.png"
            try:
                await page.screenshot(path=str(img_path), full_page=True)
            except Exception:
                pass

            # Visible text
            txt_path = OBSERVE_DIR / f"{name}.txt"
            try:
                body = await page.inner_text("body")
                txt_path.write_text(body[:20000], encoding="utf-8")
            except Exception:
                pass

            # Dump all form fields (ids / names) visible on this page
            fields_path = OBSERVE_DIR / f"{name}_fields.txt"
            try:
                fields = await page.evaluate("""() => {
                    const els = document.querySelectorAll('input, select, textarea, button');
                    return Array.from(els).map(el => ({
                        tag: el.tagName.toLowerCase(),
                        type: el.type || '',
                        id: el.id || '',
                        name: el.name || '',
                        value: el.value || '',
                        placeholder: el.placeholder || '',
                        text: el.innerText ? el.innerText.trim().slice(0,80) : '',
                        visible: el.offsetParent !== null,
                    }));
                }""")
                lines = [f"{f['tag']}[type={f['type']}] id={f['id']!r} name={f['name']!r} value={f['value']!r} visible={f['visible']}" for f in fields]
                fields_path.write_text("\n".join(lines), encoding="utf-8")
            except Exception:
                pass

            print(f"  [{ts}] {url}")
            print(f"         screenshot → {img_path.name}")
            captured.append(url)

        page.on("load", lambda: asyncio.ensure_future(on_load()))

        print("=" * 60)
        print("  Navigation Observer")
        print("=" * 60)
        print(f"  Output directory: {OBSERVE_DIR.resolve()}")
        print("  Navigating to church website …")
        print("  Navigate the site manually. Every page load is recorded.")
        print("  Press Ctrl+C when done.")
        print()

        # If we have credentials, log in automatically
        if username and password:
            await page.goto(CHURCH_URL, wait_until="domcontentloaded")
            await page.wait_for_timeout(2000)
            url = page.url

            if "signin" in url or "login" in url or "account" in url:
                print("  Logging in …")
                for sel in ["input[name='username']", "input[name='identifier']",
                            "input[type='email']", "input[id*='username']"]:
                    el = page.locator(sel)
                    if await el.count() > 0:
                        await el.first.fill(username)
                        break

                for btn in ["Next", "Continue", "Sign In"]:
                    b = page.get_by_role("button", name=btn)
                    if await b.count() > 0:
                        await b.click()
                        await page.wait_for_timeout(1500)
                        break

                for sel in ["input[type='password']", "input[name='password']"]:
                    el = page.locator(sel)
                    if await el.count() > 0:
                        await el.first.fill(password)
                        break

                for btn in ["Sign In", "Log In", "Submit"]:
                    b = page.get_by_role("button", name=btn)
                    if await b.count() > 0:
                        await b.click()
                        break

                await page.wait_for_timeout(3000)

                if "verify" in page.url.lower() or "mfa" in page.url.lower():
                    print("\n  [!] 2FA required — complete it in the browser, then press Enter.")
                    input("  Press Enter after 2FA …")
        else:
            await page.goto(CHURCH_URL, wait_until="domcontentloaded")

        # Keep the browser open until Ctrl+C
        try:
            while True:
                await asyncio.sleep(1)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass

        print(f"\n  Done. Captured {len(captured)} pages.")
        print(f"  Files saved to: {OBSERVE_DIR.resolve()}")

        await context.close()
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
