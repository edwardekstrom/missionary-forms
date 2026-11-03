#!/usr/bin/env python3
"""
Field discovery script.

Opens the Church missionary recommendations site, waits for you to:
  1. Log in
  2. Navigate to any missionary's Medical or Dental form
  3. Press Enter in this terminal

Then inspects every input, select, textarea, checkbox, and radio on the page
and prints a map of label → selector that we can use in the automator.
"""

import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright

URL = "https://missionaryrecommendations.churchofjesuschrist.org/recommendations/home/candidates?vctype=ft"


async def discover():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False, slow_mo=100)
        context = await browser.new_context(viewport={"width": 1280, "height": 900})
        page = await context.new_page()

        print(f"\nOpening: {URL}\n")
        await page.goto(URL, wait_until="domcontentloaded")

        print("=" * 60)
        print("  Please log in and navigate to a missionary's")
        print("  MEDICAL or DENTAL form tab, then come back here")
        print("  and press Enter.")
        print("=" * 60)
        input("\nPress Enter when you're on the form page ...")

        # Take screenshot
        await page.screenshot(path="logs/screenshots/discovery.png", full_page=True)
        print(f"\nScreenshot saved to logs/screenshots/discovery.png")
        print(f"Current URL: {page.url}\n")

        # Collect all form fields
        fields = []

        # Get all input/select/textarea elements
        elements = await page.locator("input, select, textarea, [role='radio'], [role='checkbox']").all()

        for el in elements:
            try:
                tag        = await el.evaluate("e => e.tagName.toLowerCase()")
                input_type = await el.get_attribute("type") or ""
                el_id      = await el.get_attribute("id") or ""
                el_name    = await el.get_attribute("name") or ""
                el_value   = await el.get_attribute("value") or ""
                aria_label = await el.get_attribute("aria-label") or ""
                placeholder= await el.get_attribute("placeholder") or ""
                is_visible = await el.is_visible()

                # Find associated label
                label_text = ""
                if el_id:
                    label_el = page.locator(f"label[for='{el_id}']")
                    if await label_el.count() > 0:
                        label_text = (await label_el.first.inner_text()).strip()

                if not label_text and aria_label:
                    label_text = aria_label

                selector = (
                    f"#{el_id}" if el_id else
                    f"[name='{el_name}']" if el_name else
                    f"{tag}"
                )

                fields.append({
                    "label":       label_text or "(no label)",
                    "tag":         tag,
                    "type":        input_type,
                    "id":          el_id,
                    "name":        el_name,
                    "value":       el_value,
                    "placeholder": placeholder,
                    "aria_label":  aria_label,
                    "selector":    selector,
                    "visible":     is_visible,
                })
            except Exception:
                continue

        # Print results
        print(f"\nFound {len(fields)} form elements:\n")
        print(f"{'Label':<40} {'Type':<15} {'ID/Name':<30} {'Visible'}")
        print("-" * 100)
        for f in fields:
            ident = f["id"] or f["name"] or "(none)"
            print(f"{f['label'][:39]:<40} {(f['tag']+'/'+f['type'])[:14]:<15} {ident[:29]:<30} {f['visible']}")

        # Save full details to JSON
        out_path = Path("logs/discovered_fields.json")
        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(json.dumps(fields, indent=2))
        print(f"\nFull field details saved to: {out_path}")

        # Also check page title and URL
        title = await page.title()
        print(f"\nPage title: {title}")
        print(f"URL: {page.url}")

        # Keep browser open
        print("\nBrowser will stay open. Press Enter to close it.")
        input()
        await browser.close()


if __name__ == "__main__":
    asyncio.run(discover())
