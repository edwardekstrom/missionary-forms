"""
Playwright-based web automation for the LDS Missionary Recommendations site.

Architecture:
  - Uses visible Chromium browser so the user can watch and catch errors
  - Takes screenshots at each major step, saved to logs/screenshots/
  - Uses semantic selectors (get_by_label, get_by_role) for resilience
  - Logs every field attempt: success or failure
  - Does NOT hard-stop on a single field failure; reports a summary at end
"""

import asyncio
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger
from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)
from rapidfuzz import fuzz, process
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table

from .config import CHURCH_URL, SCREENSHOTS_DIR, load_config
from .models import DentalForm, MedicalForm, ProcessingResult

console = Console()


# ---------------------------------------------------------------------------
# Interactive pause helper
# ---------------------------------------------------------------------------

def _pause(msg: str) -> None:
    """
    Prompt the user to press Enter.  Raises RuntimeError cleanly when stdin is
    not a real TTY (e.g. run from a background process or non-interactive shell),
    instead of hanging forever or crashing with an EOFError traceback.
    """
    if not sys.stdin.isatty():
        raise RuntimeError(
            f"Manual intervention needed but stdin is not an interactive TTY.\n"
            f"Please run this command in a regular terminal window:\n"
            f"  cd /Users/eddy.ekstrom/missionary-forms\n"
            f"  source venv/bin/activate\n"
            f"  python3 _test_dry_run.py\n"
            f"Waiting prompt was: {msg}"
        )
    try:
        input(msg)
    except EOFError:
        raise RuntimeError(f"Stdin closed while waiting for: {msg}")


# ---------------------------------------------------------------------------
# Page-wait helper
# ---------------------------------------------------------------------------

async def _wait_nav(page: Page, timeout: int = 8000):
    """
    Wait for the page to settle after a navigation or click.

    Angular/React SPAs keep background polling so "networkidle" never fires.
    We attempt networkidle with a short timeout, then fall back to
    domcontentloaded + a brief fixed delay so JS has time to render.
    """
    try:
        await page.wait_for_load_state("networkidle", timeout=timeout)
    except Exception:
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=3000)
        except Exception:
            pass
        await page.wait_for_timeout(800)


# ---------------------------------------------------------------------------
# Screenshot helper
# ---------------------------------------------------------------------------

async def screenshot(page: Page, label: str):
    cfg = load_config()
    if not cfg.get("take_screenshots"):
        return
    ts = datetime.now().strftime("%H%M%S")
    safe = re.sub(r"[^\w\-]", "_", label)
    path = SCREENSHOTS_DIR / f"{ts}_{safe}.png"
    await page.screenshot(path=str(path), full_page=False)
    logger.debug(f"Screenshot: {path.name}")


# ---------------------------------------------------------------------------
# Radio button value helper
# ---------------------------------------------------------------------------

def _fmt_imm_date(date_str: Optional[str]) -> Optional[str]:
    """Convert MM/DD/YYYY (internal format) to DD Mon YYYY (website format)."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%m/%d/%Y").strftime("%d %b %Y")
    except ValueError:
        return date_str  # pass through if already in another format


def _yn_to_radio(answer: str) -> str:
    """Convert Yes / No / Not Applicable to the website's radio button value.

    Website encoding discovered from the live form:
      1 = Yes
      0 = No
      2 = Not Applicable / Not Indicated
    """
    a = (answer or "").strip().lower()
    if a in ("yes", "true", "1"):
        return "1"
    if a in ("no", "false", "0"):
        return "0"
    if a in ("not applicable", "n/a", "na", "not indicated", "2"):
        return "2"
    return "1"  # fall back to Yes if unrecognised


# ---------------------------------------------------------------------------
# Flexible field filler
# ---------------------------------------------------------------------------

class FieldFiller:
    """
    Attempts to fill a form field using multiple selector strategies.
    Returns True on success, False if the field could not be located.
    """

    def __init__(self, page: Page):
        self.page = page
        self.filled: list[str] = []
        self.failed: list[str] = []

    async def _try_fill_text(self, locator, value: str) -> bool:
        try:
            count = await locator.count()
            if count == 0:
                return False
            await locator.first.wait_for(state="visible", timeout=3000)
            await locator.first.fill(value)
            return True
        except Exception:
            return False

    async def _try_check(self, locator, check: bool) -> bool:
        try:
            count = await locator.count()
            if count == 0:
                return False
            await locator.first.wait_for(state="visible", timeout=3000)
            if check:
                await locator.first.check()
            else:
                await locator.first.uncheck()
            return True
        except Exception:
            return False

    async def _try_select(self, locator, value: str) -> bool:
        try:
            count = await locator.count()
            if count == 0:
                return False
            await locator.first.wait_for(state="visible", timeout=3000)
            await locator.first.select_option(label=value)
            return True
        except Exception:
            try:
                await locator.first.select_option(value=value)
                return True
            except Exception:
                return False

    async def text(self, label: str, value: Optional[str], *, required: bool = False) -> bool:
        if not value:
            return True  # nothing to fill
        locators = [
            self.page.get_by_label(label, exact=True),
            self.page.get_by_label(label),
            self.page.get_by_placeholder(label),
            self.page.locator(f"input[aria-label='{label}']"),
            self.page.locator(f"textarea[aria-label='{label}']"),
        ]
        for loc in locators:
            if await self._try_fill_text(loc, value):
                logger.debug(f"  FILLED  '{label}' = '{value}'")
                self.filled.append(label)
                return True
        if required:
            logger.warning(f"  FAILED  '{label}' (required)")
        else:
            logger.debug(f"  MISSED  '{label}' = '{value}'")
        self.failed.append(label)
        return False

    async def checkbox(self, label: str, value: Optional[str]) -> bool:
        """value should be 'Yes'/'No' or True/False."""
        if value is None:
            return True
        check = str(value).lower() in ("yes", "true", "1")
        locators = [
            self.page.get_by_label(label, exact=True),
            self.page.get_by_label(label),
            self.page.locator(f"input[type='checkbox'][aria-label='{label}']"),
        ]
        for loc in locators:
            if await self._try_check(loc, check):
                logger.debug(f"  CHECKED '{label}' = {check}")
                self.filled.append(label)
                return True
        logger.debug(f"  MISSED  checkbox '{label}' = {check}")
        self.failed.append(label)
        return False

    async def radio(self, name: str, value: Optional[str]) -> bool:
        """Click a radio button whose label/value matches value."""
        if not value:
            return True
        # Try to find the radio input next to a label containing the value
        locators = [
            self.page.get_by_label(value, exact=False),
            self.page.locator(f"input[type='radio'][value='{value}']"),
            self.page.locator(f"input[type='radio'][aria-label*='{value}']"),
        ]
        for loc in locators:
            try:
                count = await loc.count()
                if count > 0:
                    await loc.first.click()
                    logger.debug(f"  RADIO   '{name}' = '{value}'")
                    self.filled.append(name)
                    return True
            except Exception:
                continue
        logger.debug(f"  MISSED  radio '{name}' = '{value}'")
        self.failed.append(name)
        return False

    async def select(self, label: str, value: Optional[str]) -> bool:
        if not value:
            return True
        locators = [
            self.page.get_by_label(label, exact=True),
            self.page.get_by_label(label),
            self.page.locator(f"select[aria-label='{label}']"),
        ]
        for loc in locators:
            if await self._try_select(loc, value):
                logger.debug(f"  SELECT  '{label}' = '{value}'")
                self.filled.append(label)
                return True
        logger.debug(f"  MISSED  select '{label}' = '{value}'")
        self.failed.append(label)
        return False

    def summary(self) -> tuple[list[str], list[str]]:
        return self.filled, self.failed


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------

async def login(page: Page, username: str, password: str):
    """Handle the Church of Jesus Christ SSO login flow."""
    logger.info("Navigating to missionary recommendations site …")
    await page.goto(CHURCH_URL, wait_until="domcontentloaded")
    await screenshot(page, "01_initial_load")

    # The site will redirect to Church SSO if not logged in
    await page.wait_for_timeout(3000)

    current_url = page.url
    logger.info(f"Current URL: {current_url}")

    # Detect sign-in page: either redirected to OAuth server OR page shows embedded sign-in widget
    is_signin_url = (
        "signin" in current_url or "login" in current_url or "account" in current_url
        or "id.churchofjesuschrist.org" in current_url
    )
    has_signin_widget = await page.get_by_label("Username").count() > 0

    # If redirected to login
    if is_signin_url or has_signin_widget:
        logger.info("Entering credentials …")

        # Username field (Church uses 'username' or 'identifier')
        for label in ["Username", "Church Account", "Email or username", "Username or email"]:
            try:
                field = page.get_by_label(label)
                if await field.count() > 0:
                    await field.fill(username)
                    break
            except Exception:
                continue
        else:
            # Fallback: find any username-ish input
            for sel in ["input[name='username']", "input[name='identifier']",
                        "input[type='email']", "input[id*='username']"]:
                try:
                    el = page.locator(sel)
                    if await el.count() > 0:
                        await el.first.fill(username)
                        break
                except Exception:
                    continue

        await screenshot(page, "02_username_entered")

        # Click Next / Continue if it exists before the password field
        for btn_text in ["Next", "Continue", "Sign In"]:
            try:
                btn = page.get_by_role("button", name=btn_text)
                if await btn.count() > 0:
                    await btn.click()
                    await page.wait_for_timeout(1500)
                    break
            except Exception:
                continue

        # Password field
        for sel in ["input[type='password']", "input[name='password']"]:
            try:
                el = page.locator(sel)
                if await el.count() > 0:
                    await el.first.fill(password)
                    break
            except Exception:
                continue

        await screenshot(page, "03_password_entered")

        # Submit login — the Church SSO uses "Verify" as the submit button label
        for btn_text in ["Verify", "Sign In", "Log In", "Submit", "Continue"]:
            try:
                btn = page.get_by_role("button", name=btn_text)
                if await btn.count() > 0:
                    await btn.click()
                    break
            except Exception:
                continue

        # Wait for OAuth redirect to complete (up to 60 s to allow for 2FA push).
        # IMPORTANT: use the full hostname in the pattern so Playwright doesn't
        # match the OAuth authorize URL, which contains the recommendations site
        # as an unencoded redirect_uri query parameter.
        try:
            await page.wait_for_url(
                "https://missionaryrecommendations.churchofjesuschrist.org/**",
                timeout=60000,
            )
        except Exception:
            pass
        # Angular apps keep polling so "networkidle" may never fire — ignore timeout
        try:
            await _wait_nav(page, 15000)
        except Exception:
            pass
        await screenshot(page, "04_after_login")

        from urllib.parse import urlparse
        current_url = page.url
        logger.info(f"Post-login URL: {current_url}")

        # If we're still on the auth/SSO server, the user needs to complete a
        # step manually (e.g., push notification, email OTP, or a 2FA prompt
        # with a label we don't recognise).
        if urlparse(current_url).hostname != "missionaryrecommendations.churchofjesuschrist.org":
            console.print("\n[bold yellow]Login is paused — browser needs your attention.[/bold yellow]")
            console.print("Complete any 2FA / verification step in the browser, then waiting up to 3 minutes …")
            logger.info("MFA required — polling for redirect (up to 180s) …")
            try:
                # Poll until the URL changes to the candidates page (covers push notifications / OTP)
                await page.wait_for_url(
                    "https://missionaryrecommendations.churchofjesuschrist.org/**",
                    timeout=180000,
                )
            except Exception:
                pass
            await _wait_nav(page, 15000)
            current_url = page.url
            logger.info(f"Post-manual-login URL: {current_url}")

        # Handle 2FA if prompted (secondary check after URL poll)
        if "verify" in current_url.lower() or "mfa" in current_url.lower() or "two-factor" in current_url.lower():
            console.print("\n[bold yellow]Two-factor authentication required — waiting up to 3 minutes …[/bold yellow]")
            logger.info("Secondary MFA check — polling for redirect (up to 180s) …")
            try:
                await page.wait_for_url(
                    "https://missionaryrecommendations.churchofjesuschrist.org/**",
                    timeout=180000,
                )
            except Exception:
                pass
            await _wait_nav(page, 15000)


# ---------------------------------------------------------------------------
# Find candidate
# ---------------------------------------------------------------------------

async def get_candidate_names(page: Page) -> list[tuple[str, str]]:
    """
    Return a list of (display_name, locator_text) tuples from the candidates list.
    Candidates are inside collapsed ward accordion sections; expand them all first.
    """
    await _wait_nav(page, 15000)
    # Wait for Angular to finish rendering the candidate rows.  The site keeps
    # network-polling so "networkidle" never fires; give it a fixed extra wait.
    await page.wait_for_timeout(5000)
    await screenshot(page, "05_candidates_list")

    # Collect all candidate name-like text elements
    candidates = []

    # Try common patterns: table rows, cards, list items with names
    for selector in [
        "[class*='candidate'] [class*='name']",
        "table tbody tr td:first-child",
        "[class*='card'] [class*='name']",
        "li [class*='name']",
        "[data-testid*='candidate'] [class*='name']",
    ]:
        elements = await page.locator(selector).all()
        for el in elements:
            text = (await el.inner_text()).strip()
            if text and len(text) > 2:
                candidates.append((text, selector))
        if candidates:
            break

    # Fallback: read the full DOM text (includes collapsed/hidden Angular content)
    # and pull out strings that look like "Last, First" — the site's name format.
    if not candidates:
        # Use textContent (not inner_text) so we get text from collapsed sections too
        all_text = await page.evaluate("() => document.body.textContent")
        # Prefer "Last, First" format (the site's format) over "First Last"
        name_like = re.findall(r"[A-Z][a-z]+(?:[\- ][A-Z][a-z]+)?,\s*[A-Z][a-z]+(?:[\- ][A-Z][a-z]+)*", all_text)
        if not name_like:
            name_like = re.findall(r"[A-Z][a-z]+ [A-Z][a-z]+", all_text)
        candidates = [(n, "body") for n in name_like]

    logger.debug(f"Found {len(candidates)} candidates on page")
    return candidates


async def find_and_click_candidate(page: Page, missionary_name: str) -> bool:
    """
    Navigate to the missionary's record on the candidates page.

    Always returns to the candidates LIST first (CHURCH_URL) so successive
    calls work correctly even when we're already deep inside a candidate record.
    """
    logger.info(f"Searching for candidate: {missionary_name}")

    # Always navigate back to the candidates list for a clean starting point.
    # Unconditional: even if we're already at CHURCH_URL, a fresh goto ensures
    # Angular re-loads the candidates list (important after session renewal or
    # long Ollama runs where the SPA state may have gone stale).
    await page.goto(CHURCH_URL, wait_until="domcontentloaded")
    await _wait_nav(page, 15000)

    # Give Angular time to finish rendering all candidate rows
    await page.wait_for_timeout(5000)

    name_parts = missionary_name.split()
    last_name  = name_parts[-1] if name_parts else missionary_name
    first_name = name_parts[0]  if len(name_parts) > 1 else ""

    # --- Strategy 1: click a <a> or clickable row that contains the name ---
    search_attempts = [
        missionary_name,
        f"{last_name}, {first_name}",
        f"{last_name},{first_name}",
        last_name,
    ]

    for search_text in search_attempts:
        if not search_text.strip():
            continue
        try:
            # Prefer links and table/list cells over arbitrary text nodes
            for sel_fn in [
                lambda t: page.locator(f"a:has-text('{t}')"),
                lambda t: page.locator(f"td:has-text('{t}')"),
                lambda t: page.locator(f"li:has-text('{t}')"),
                lambda t: page.get_by_text(t, exact=False),
            ]:
                locator = sel_fn(search_text)
                count = await locator.count()
                if count > 0:
                    await locator.first.click()
                    await _wait_nav(page, 10000)
                    # Verify we actually navigated somewhere new
                    if page.url != CHURCH_URL:
                        logger.info(f"Clicked candidate via '{search_text}'")
                        await screenshot(page, "06_candidate_selected")
                        return True
        except Exception:
            continue

    # --- Strategy 2: search / filter box ---
    for search_sel in [
        "input[placeholder*='Search']",
        "input[placeholder*='search']",
        "input[type='search']",
        "[aria-label*='search' i]",
    ]:
        try:
            search_box = page.locator(search_sel)
            if await search_box.count() > 0:
                await search_box.first.fill(missionary_name)
                await page.wait_for_timeout(1500)
                result = page.locator(f"a:has-text('{last_name}'), td:has-text('{last_name}')")
                if await result.count() == 0:
                    result = page.get_by_text(last_name, exact=False)
                if await result.count() > 0:
                    await result.first.click()
                    await _wait_nav(page, 10000)
                    if page.url != CHURCH_URL:
                        logger.info(f"Found via search box: {missionary_name}")
                        await screenshot(page, "06_candidate_selected")
                        return True
        except Exception:
            continue

    # --- Strategy 3: JavaScript click on any element whose text matches ---
    try:
        clicked = await page.evaluate(f"""
            () => {{
                const name = {repr(last_name)};
                const walker = document.createTreeWalker(
                    document.body, NodeFilter.SHOW_TEXT
                );
                let node;
                while ((node = walker.nextNode())) {{
                    if (node.textContent.includes(name)) {{
                        const el = node.parentElement;
                        const tag = el.tagName.toLowerCase();
                        if (['a', 'button', 'td', 'li', 'tr'].includes(tag) ||
                            el.style.cursor === 'pointer' ||
                            el.onclick !== null) {{
                            el.click();
                            return true;
                        }}
                    }}
                }}
                return false;
            }}
        """)
        if clicked:
            await _wait_nav(page, 10000)
            if page.url != CHURCH_URL:
                logger.info(f"Clicked candidate via JS walker: {missionary_name}")
                await screenshot(page, "06_candidate_selected")
                return True
    except Exception as e:
        logger.debug(f"JS walker failed: {e}")

    logger.error(f"Could not find candidate '{missionary_name}' on the page")
    await screenshot(page, "06_candidate_NOT_FOUND")
    return False


# ---------------------------------------------------------------------------
# Navigate to medical / dental tab
# ---------------------------------------------------------------------------

async def click_form_tab(page: Page, form_type: str) -> bool:
    """
    Navigate to the Medical or Dental evaluation form from a candidate's page.

    The site uses links like "Complete Physician's Evaluation" (medical) and
    "Complete Dental Evaluation Form" (dental) — not generic tab labels.
    form_type: 'medical' or 'dental'
    """
    label = "Medical" if form_type == "medical" else "Dental"
    logger.info(f"Navigating to {label} form …")

    # Wait for the checklist content to fully render in Angular
    try:
        await page.wait_for_selector("text=Individual Information", timeout=10000)
    except Exception:
        pass
    await page.wait_for_timeout(800)

    # Debug: log all link texts on the page
    try:
        all_links = await page.evaluate("""
            () => Array.from(document.querySelectorAll('a')).map(a => a.innerText.trim()).filter(t => t)
        """)
        logger.debug(f"Links on page: {all_links[:20]}")
    except Exception:
        pass

    # Try Playwright get_by_text first (most reliable for Angular)
    search_terms = (
        ["Complete Physician", "Physician Evaluation", "Medical Evaluation", "Health Evaluation"]
        if form_type == "medical"
        else ["Complete Dental", "Dental Evaluation", "Dental Form"]
    )

    for text in search_terms:
        # Try as any element type (most flexible)
        for loc in [
            page.get_by_text(text, exact=False),
            page.locator(f"a:has-text('{text}')"),
            page.locator(f"*:has-text('{text}') >> nth=0"),
        ]:
            try:
                if await loc.count() > 0:
                    await loc.first.click()
                    await _wait_nav(page, 10000)
                    await screenshot(page, f"07_{form_type}_tab")
                    logger.info(f"Navigated to {label} form via '{text}'")
                    return True
            except Exception:
                continue

    # Fallback: href attribute matching
    for href_frag in (["physician", "health-eval", "medical"] if form_type == "medical"
                      else ["dental"]):
        try:
            loc = page.locator(f"a[href*='{href_frag}']")
            if await loc.count() > 0:
                await loc.first.click()
                await _wait_nav(page, 10000)
                await screenshot(page, f"07_{form_type}_tab")
                logger.info(f"Navigated to {label} form via href*={href_frag}")
                return True
        except Exception:
            continue

    await screenshot(page, f"07_{form_type}_NOT_FOUND")
    logger.warning(f"Could not find {label} form link")
    return False


# ---------------------------------------------------------------------------
# Medical form filler
# ---------------------------------------------------------------------------

async def fill_medical_form(page: Page, form: MedicalForm):
    """
    Fill the medical form using exact field IDs/names from discovered_fields.json.

    Key encoding differences from dental:
      System exam radios (HlthEval*NA): 1 = Normal, 2 = Abnormal
      Fitness level (HlthEvalFitnessLevel): 1=A, 2=B, 3=C, 4=D, 5=E
      TB exposure (HlthEvalTBExposerYN): 1=Yes, 0=No
      IGRA (HlthEvalInterferon): 0=Negative, 1=Positive, 2=Not Done
      Chest X-ray (HlthEvalChestXray): 1=Normal, 2=Abnormal, 3=Not Done
      Chronic condition (HlthEvalChronicConditionYN): 1=Yes, 0=No
      Driving medication (HlthEvalDrivingMedicationYN): 1=Yes, 0=No
      Height units (HlthEvalHeightUnits): 1=inches, 2=cm
      Weight units (HlthEvalWeightUnits): 1=lbs, 2=kg
      Hemo type (HlthEvalHemoOrHema): 2=hemoglobin, 1=hematocrit
    """
    logger.info("Filling medical form …")
    filled: list[str] = []
    failed: list[str] = []

    # ── Navigate past Introduction page ──────────────────────────────────────
    # The form wizard starts with a HIPAA intro that has no form fields.
    # Click "Next" to reach the actual physician evaluation form.
    try:
        is_intro = await page.locator("h1:has-text('Introduction'), h2:has-text('Introduction')").count() > 0
        if is_intro:
            next_btn = page.locator("button:has-text('Next')")
            if await next_btn.count() > 0:
                await next_btn.first.click()
                await _wait_nav(page, 10000)
                try:
                    await page.wait_for_selector(
                        "#HlthEvalHeight, input[name='HlthEvalHeightUnits']",
                        timeout=15000,
                    )
                    logger.info("Navigated past Introduction to medical form fields")
                except Exception:
                    logger.warning("Medical form fields not found after skipping Introduction")
    except Exception as e:
        logger.debug(f"Introduction skip error (medical): {e}")

    async def radio(name: str, val: str) -> None:
        """Click input[name=name][value=val].
        force=True is tried first; if the element is zero-sized and outside the
        viewport (CSS-hidden Angular radios) we fall back to a JS .click() which
        bypasses Playwright's geometry checks entirely.
        """
        sel = f"input[name='{name}'][value='{val}']"
        try:
            el = page.locator(sel)
            if await el.count() > 0:
                try:
                    await el.click(force=True)
                except Exception:
                    # Zero-sized hidden input: dispatch click via JS
                    await el.evaluate("el => el.click()")
                logger.debug(f"  RADIO  {name}={val}")
                filled.append(name)
            else:
                logger.debug(f"  MISSED radio {name}={val}")
                failed.append(name)
        except Exception as e:
            logger.debug(f"  ERROR  radio {name}: {e}")
            failed.append(name)

    async def text(field_id: str, value: Optional[str], *, press_tab: bool = False) -> None:
        """Fill a text input or textarea by id."""
        if not value:
            return
        try:
            el = page.locator(f"#{field_id}")
            if await el.count() > 0:
                await el.first.fill(value)
                if press_tab:
                    await el.first.press("Tab")
                logger.debug(f"  FILLED #{field_id} = {value!r}")
                filled.append(field_id)
            else:
                logger.debug(f"  MISSED #{field_id}")
                failed.append(field_id)
        except Exception as e:
            logger.debug(f"  ERROR  #{field_id}: {e}")
            failed.append(field_id)

    async def fill_date(field_id: str, value: Optional[str]) -> None:
        """
        Fill a calendar-backed date input by simulating real keyboard typing.

        Angular date pickers open a calendar on click and require character-by-
        character input events (not a bulk fill) to register the typed date.
        After typing we press Tab to dismiss the calendar and commit the value.
        """
        if not value:
            return
        try:
            el = page.locator(f"#{field_id}")
            if await el.count() > 0:
                await el.first.click()                              # focus + open calendar
                await page.wait_for_timeout(300)                   # let calendar render
                await el.first.click(click_count=3)                # select existing text
                await el.first.press_sequentially(value, delay=50) # type char-by-char
                await el.first.press("Tab")                        # commit + dismiss
                await page.wait_for_timeout(200)                   # let Angular process
                logger.debug(f"  DATE   #{field_id} = {value!r}")
                filled.append(field_id)
            else:
                logger.debug(f"  MISSED date #{field_id}")
                failed.append(field_id)
        except Exception as e:
            logger.debug(f"  ERROR  date #{field_id}: {e}")
            failed.append(field_id)

    async def select_opt(field_id: str, value: Optional[str]) -> None:
        """Select a <select> option by label text, falling back to value."""
        if not value:
            return
        try:
            el = page.locator(f"#{field_id}")
            if await el.count() > 0:
                try:
                    await el.first.select_option(label=value)
                except Exception:
                    await el.first.select_option(value=value)
                logger.debug(f"  SELECT #{field_id} = {value!r}")
                filled.append(field_id)
            else:
                logger.debug(f"  MISSED select #{field_id}")
                failed.append(field_id)
        except Exception as e:
            logger.debug(f"  ERROR  #{field_id}: {e}")
            failed.append(field_id)

    def exam_radio_val(status: Optional[str]) -> Optional[str]:
        """Convert Normal/Abnormal to 1/2 for system exam radio buttons."""
        if not status:
            return None
        s = status.strip().lower()
        if s == "normal":
            return "1"
        if s == "abnormal":
            return "2"
        return None

    # ── Vital Signs ───────────────────────────────────────────────────────
    vs = form.vital_signs

    # Height: numeric value + units radio (1=inches, 2=cm)
    if vs.height:
        m = re.match(r"([\d\.]+)\s*(in|cm)?", vs.height.strip(), re.IGNORECASE)
        if m:
            await text("HlthEvalHeight", m.group(1))
            unit_val = "2" if m.group(2) and "cm" in m.group(2).lower() else "1"
            await radio("HlthEvalHeightUnits", unit_val)

    # Weight: numeric value + units radio (1=lbs, 2=kg)
    if vs.weight:
        m = re.match(r"([\d\.]+)\s*(lb|kg|pound)?", vs.weight.strip(), re.IGNORECASE)
        if m:
            await text("HlthEvalWeight", m.group(1))
            unit_val = "2" if m.group(2) and "kg" in m.group(2).lower() else "1"
            await radio("HlthEvalWeightUnits", unit_val)

    # Blood pressure: split "120/80" → systolic + diastolic (two separate fields)
    if vs.blood_pressure:
        parts = vs.blood_pressure.split("/")
        await text("HlthEvalBloodPressure", parts[0].strip())
        if len(parts) > 1:
            await text("HlthEvalBloodPressure2", parts[1].strip())

    await text("HlthEvalPulse", vs.pulse)
    await text("HlthEvalVisionLeft", vs.vision_left)
    await text("HlthEvalVisionRight", vs.vision_right)

    await screenshot(page, "08_vital_signs")

    # ── System Exam ───────────────────────────────────────────────────────
    # Each system: radio (1=Normal, 2=Abnormal) + optional description textarea
    se = form.system_exam

    async def exam_radio(field_name: str, status: Optional[str]) -> None:
        val = exam_radio_val(status)
        if val:
            await radio(field_name, val)

    await exam_radio("HlthEvalGenAppearanceNA", se.general)
    await exam_radio("HlthEvalSkinNA", se.skin)
    await exam_radio("HlthEvalEyesNA", se.eyes)
    await exam_radio("HlthEvalEarsNA", se.ears)
    await exam_radio("HlthEvalNoseNA", se.nose_throat)
    await exam_radio("HlthEvalChestNA", se.chest)
    await exam_radio("HlthEvalHeartNA", se.heart)
    await exam_radio("HlthEvalAbdomenNA", se.abdomen)
    await exam_radio("HlthEvalRectalNA", se.genitalia)
    await exam_radio("HlthEvalBackNA", se.back)
    # Upper and lower extremities share the same parsed value
    await exam_radio("HlthEvalUpperExtrmNA", se.extremities)
    await exam_radio("HlthEvalLowerExtrmNA", se.extremities)
    await exam_radio("HlthEvalNeurologicalNA", se.neurological)

    await screenshot(page, "09_system_exam")

    # ── Lab Results ───────────────────────────────────────────────────────
    lab = form.labs

    await text("HlthEvalUrinalysisDipstickBloodText", lab.urinalysis_blood)
    await text("HlthEvalUrinalysisDipstickProtein", lab.urinalysis_protein)
    await text("HlthEvalUrinalysisDipstickSugar", lab.urinalysis_sugar)

    # Hemoglobin: select type (2=hemoglobin) then enter value
    if lab.hemoglobin:
        await radio("HlthEvalHemoOrHema", "2")
        await text("HlthEvalHemoglobin", lab.hemoglobin)

    # TB exposure: 1=Yes, 0=No
    if lab.tb_exposure_risk:
        await radio("HlthEvalTBExposerYN", "1" if lab.tb_exposure_risk.lower() == "yes" else "0")

    # PPD: extract numeric mm from "5 mm" or "10mm"
    if lab.ppd_result:
        m = re.match(r"([\d\.]+)", lab.ppd_result.strip())
        if m:
            await text("HlthEvalTbMmInduration", m.group(1))

    # IGRA: 0=Negative, 1=Positive, 2=Not Done
    if lab.igra_result:
        igra_map = {"negative": "0", "positive": "1", "not done": "2"}
        igra_val = igra_map.get(lab.igra_result.lower().strip())
        if igra_val:
            await radio("HlthEvalInterferon", igra_val)

    # Chest X-ray: 1=Normal, 2=Abnormal, 3=Not Done
    if lab.chest_xray:
        xray_map = {"normal": "1", "abnormal": "2", "not done": "3"}
        xray_val = xray_map.get(lab.chest_xray.lower().strip())
        if xray_val:
            await radio("HlthEvalChestXray", xray_val)

    await screenshot(page, "10_labs")

    # ── Immunizations ─────────────────────────────────────────────────────
    # NOTE: As of the latest form changes, the website no longer requires
    # vaccine/immunization dates to be entered. This section is kept for
    # parsing purposes only (data is still extracted from PDFs).
    # imm = form.immunizations
    #
    # await fill_date("HlthEvalImmunTetanus",   _fmt_imm_date(imm.tdap))
    # await fill_date("HlthEvalImmunTetanusA2", _fmt_imm_date(imm.tdap2))
    # await fill_date("HlthEvalImmunMmr1",      _fmt_imm_date(imm.mmr1))
    # await fill_date("HlthEvalImmunMmr2",      _fmt_imm_date(imm.mmr2))
    # await fill_date("HlthEvalImmunPolio",     _fmt_imm_date(imm.polio))
    # await fill_date("HlthEvalImmunHepA1",     _fmt_imm_date(imm.hep_a1))
    # await fill_date("HlthEvalImmunHepA2",     _fmt_imm_date(imm.hep_a2))
    # await fill_date("HlthEvalImmunHepB1",     _fmt_imm_date(imm.hep_b1))
    # await fill_date("HlthEvalImmunHepB2",     _fmt_imm_date(imm.hep_b2))
    # await fill_date("HlthEvalImmunHepB3",     _fmt_imm_date(imm.hep_b3))
    # await fill_date("HlthEvalImmunInfluenza", _fmt_imm_date(imm.influenza))
    #
    # # COVID: use extracted brand to pick the right per-brand field pair.
    # # imm.other holds the brand name from the PDF (e.g. "Pfizer", "Moderna").
    # # Fall back to Pfizer if brand is absent or unrecognised.
    # if imm.covid_doses:
    #     _COVID_BRAND_PREFIX = {
    #         "pfizer":      "HlthEvalImmunPfizer",
    #         "biontech":    "HlthEvalImmunPfizer",
    #         "moderna":     "HlthEvalImmunModerna",
    #         "j&j":         "HlthEvalImmunJohnson",
    #         "janssen":     "HlthEvalImmunJohnson",
    #         "johnson":     "HlthEvalImmunJohnson",
    #         "astrazeneca": "HlthEvalImmunOxford",
    #         "oxford":      "HlthEvalImmunOxford",
    #         "covaxin":     "HlthEvalImmunCovaxin",
    #         "sinovac":     "HlthEvalImmunSinovac",
    #         "sinopharm":   "HlthEvalImmunBBIBP",
    #         "bbibp":       "HlthEvalImmunBBIBP",
    #         "cansino":     "HlthEvalImmunCanSinoBIO",
    #         "sputnik":     "HlthEvalImmunSputnikV",
    #         "curevac":     "HlthEvalImmunCureVac",
    #     }
    #     brand_raw = (imm.other or "").strip().lower()
    #     prefix = next(
    #         (v for k, v in _COVID_BRAND_PREFIX.items() if k in brand_raw),
    #         "HlthEvalImmunPfizer",  # default
    #     )
    #     if len(imm.covid_doses) >= 1:
    #         await fill_date(f"{prefix}1", _fmt_imm_date(imm.covid_doses[0]))
    #     if len(imm.covid_doses) >= 2:
    #         await fill_date(f"{prefix}2", _fmt_imm_date(imm.covid_doses[1]))

    # await screenshot(page, "11_immunizations")

    # ── Driving/Medication question ───────────────────────────────────────
    if form.driving_medication:
        dm_val = "1" if form.driving_medication.lower() == "yes" else "0"
        await radio("HlthEvalDrivingMedicationYN", dm_val)
    if form.driving_medication_explain:
        await text("HlthEvalDrivingMedicationExplain", form.driving_medication_explain)

    # ── Functional Level + Chronic Condition ─────────────────────────────
    # HlthEvalFitnessLevel: 1=A, 2=B, 3=C, 4=D, 5=E
    if form.functional_level:
        level_map = {"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"}
        level_val = level_map.get(form.functional_level.upper().strip())
        if level_val:
            await radio("HlthEvalFitnessLevel", level_val)

    # HlthEvalChronicConditionYN: 1=Yes, 0=No
    if form.chronic_condition:
        await radio("HlthEvalChronicConditionYN",
                    "1" if form.chronic_condition.lower() == "yes" else "0")

    if form.chronic_comments:
        await text("HlthEvalFitnessComment", form.chronic_comments)

    await screenshot(page, "12_assessment")

    # ── Physician Information ─────────────────────────────────────────────
    pi = form.physician_info
    await text("HlthEvalPhysicianName", pi.name)
    await fill_date("HlthEvalPhysicianExamDate", _fmt_imm_date(form.exam_date))
    await text("HlthEvalPhysicianAddress", pi.address)
    await text("HlthEvalPhysicianCity", pi.city)
    await select_opt("HlthEvalPhysicianCountry", pi.country)
    await page.wait_for_timeout(2000)          # state dropdown repopulates after country change
    await select_opt("HlthEvalPhysicianStateProvince", pi.state)
    await text("HlthEvalPhysicianPostalCode", pi.postal_code)
    await text("HlthEvalPhysicianDistrict", pi.district)
    await text("HlthEvalPhysicianPhone", pi.phone)
    await text("HlthEvalPhysicianEmail", pi.email)

    await screenshot(page, "13_physician_info")

    logger.info(f"Medical form: {len(filled)} fields filled, {len(failed)} not found")
    return filled, failed


# ---------------------------------------------------------------------------
# Dental form filler  (uses exact field IDs/names from discovered_fields.json)
# ---------------------------------------------------------------------------

async def fill_dental_form(page: Page, form: DentalForm):
    """
    Fill the dental form using the exact name= / id= attributes discovered
    from the live website.

    Radio button value encoding (from discovered_fields.json):
      1 = Yes
      0 = No
      2 = Not Applicable
    """
    logger.info("Filling dental form …")
    filled: list[str] = []
    failed: list[str] = []

    # ── Navigate past Introduction page ──────────────────────────────────────
    try:
        is_intro = await page.locator("h1:has-text('Introduction'), h2:has-text('Introduction')").count() > 0
        if is_intro:
            next_btn = page.locator("button:has-text('Next')")
            if await next_btn.count() > 0:
                await next_btn.first.click()
                await _wait_nav(page, 10000)
                try:
                    await page.wait_for_selector(
                        "#DntlHistBrushFrequency, input[name='DntlHistOrthCompletedYN']",
                        timeout=30000,
                    )
                    logger.info("Navigated past Introduction to dental form fields")
                except Exception:
                    logger.warning("Dental form fields not found after skipping Introduction")
    except Exception as e:
        logger.debug(f"Introduction skip error (dental): {e}")

    async def radio(name: str, answer: Optional[str]) -> None:
        """Click the radio button input[name=name][value=radio_value].
        Falls back to JS .click() for CSS-hidden zero-sized inputs.
        """
        if answer is None:
            return
        val = _yn_to_radio(answer)
        sel = f"input[name='{name}'][value='{val}']"
        try:
            el = page.locator(sel)
            if await el.count() > 0:
                try:
                    await el.click(force=True)
                except Exception:
                    await el.evaluate("el => el.click()")
                logger.debug(f"  RADIO  {name}={val} ({answer})")
                filled.append(name)
            else:
                logger.debug(f"  MISSED radio {name}={val}")
                failed.append(name)
        except Exception as e:
            logger.debug(f"  ERROR  radio {name}: {e}")
            failed.append(name)

    async def text(field_id: str, value: Optional[str], *, press_tab: bool = False) -> None:
        """Fill a text input or textarea by id."""
        if not value:
            return
        try:
            el = page.locator(f"#{field_id}")
            if await el.count() > 0:
                await el.first.fill(value)
                if press_tab:
                    await el.first.press("Tab")
                logger.debug(f"  FILLED #{field_id} = {value!r}")
                filled.append(field_id)
            else:
                logger.debug(f"  MISSED #{field_id}")
                failed.append(field_id)
        except Exception as e:
            logger.debug(f"  ERROR  #{field_id}: {e}")
            failed.append(field_id)

    async def fill_date(field_id: str, value: Optional[str]) -> None:
        """
        Fill a calendar-backed date input by simulating real keyboard typing.

        Angular date pickers open a calendar on click and require character-by-
        character input events (not a bulk fill) to register the typed date.
        After typing we press Tab to dismiss the calendar and commit the value.
        """
        if not value:
            return
        try:
            el = page.locator(f"#{field_id}")
            if await el.count() > 0:
                await el.first.click()                              # focus + open calendar
                await page.wait_for_timeout(300)                   # let calendar render
                await el.first.click(click_count=3)                # select existing text
                await el.first.press_sequentially(value, delay=50) # type char-by-char
                await el.first.press("Tab")                        # commit + dismiss
                await page.wait_for_timeout(200)                   # let Angular process
                logger.debug(f"  DATE   #{field_id} = {value!r}")
                filled.append(field_id)
            else:
                logger.debug(f"  MISSED date #{field_id}")
                failed.append(field_id)
        except Exception as e:
            logger.debug(f"  ERROR  date #{field_id}: {e}")
            failed.append(field_id)

    async def select_opt(field_id: str, value: Optional[str]) -> None:
        """Select a <select> option by label text, falling back to value."""
        if not value:
            return
        try:
            el = page.locator(f"#{field_id}")
            if await el.count() > 0:
                try:
                    await el.first.select_option(label=value)
                except Exception:
                    await el.first.select_option(value=value)
                logger.debug(f"  SELECT #{field_id} = {value!r}")
                filled.append(field_id)
            else:
                logger.debug(f"  MISSED select #{field_id}")
                failed.append(field_id)
        except Exception as e:
            logger.debug(f"  ERROR  #{field_id}: {e}")
            failed.append(field_id)

    # ── Candidate History ─────────────────────────────────────────────────
    # DntlHistOrthCompletedYN: 1=Yes, 0=No, 2=Not Applicable
    await radio("DntlHistOrthCompletedYN", form.orthodontic_treatment)
    # DntlHistWisdomYN: 1=Yes, 0=No
    await radio("DntlHistWisdomYN", form.wisdom_teeth_removed)
    await text("DntlHistBrushFrequency", form.brushing_frequency)
    await text("DntlHistFlossFrequency", form.flossing_frequency)
    # DntlHistBleedingYN: 1=Yes, 0=No
    await radio("DntlHistBleedingYN", form.pain_or_bleeding)
    await text("DntlHistBleedingComments", form.bleeding_comments)

    await screenshot(page, "13_dental_candidate_history")

    # ── Dentist Information ───────────────────────────────────────────────
    await text("DntlEvalDentistName", form.dentist_name)
    await fill_date("DntlEvalOralExamDate", _fmt_imm_date(form.exam_date))
    await text("DntlEvalDentistAddress", form.dentist_address)
    await text("DntlEvalDentistCity", form.dentist_city)
    await select_opt("DntlEvalDentistCountry", form.dentist_country)
    await page.wait_for_timeout(2000)          # state dropdown repopulates after country change
    await select_opt("DntlEvalDentistStateProvince", form.dentist_state)
    await text("DntlEvalDentistPostalCode", form.dentist_postal)
    await text("DntlEvalDentistDistrict", form.dentist_district)
    await text("DntlEvalDentistPhone", form.dentist_phone)
    await text("DntlEvalDentistEmail", form.dentist_email)

    await screenshot(page, "14_dentist_info")

    # ── Evaluation Findings ───────────────────────────────────────────────
    # DntlEvalOralExamYN: 1=Yes, 0=No
    await radio("DntlEvalOralExamYN", form.oral_exam_with_bitewing)
    # DntlEvalFullXraysYN: 1=Yes, 0=No, 2=Not Applicable (panoramic/full x-rays)
    await radio("DntlEvalFullXraysYN", form.panoramic_done)
    # DntlEvalMolarsYN: 1=Yes, 0=No, 2=Not Applicable (third molars assessed)
    await radio("DntlEvalMolarsYN", form.third_molars_assessed)
    # DntlEvalRestorationsYN: 1=Yes, 0=No (restorations / bonded retainers)
    await radio("DntlEvalRestorationsYN", form.bonded_retainers)
    # DntlEvalCurrentTreatmentYN: 1=Yes, 0=No, 2=Not Applicable (active ortho complete)
    await radio("DntlEvalCurrentTreatmentYN", form.active_orthodontic)
    # DntlEvalProperHygieneYN: 1=Yes, 0=No
    await radio("DntlEvalProperHygieneYN", form.oral_hygiene_acceptable)
    # DntlEvalProblemsYN: 1=Yes, 0=No (any dental problems/decay present)
    await radio("DntlEvalProblemsYN", form.dental_decay)
    await text("DntlEvalAdditionalComments", form.dental_comments)

    await screenshot(page, "15_dental_evaluation")

    logger.info(f"Dental form: {len(filled)} fields filled, {len(failed)} not found")
    return filled, failed


# ---------------------------------------------------------------------------
# Navigation helpers
# ---------------------------------------------------------------------------

async def _try_next(page: Page, section_hint: str = ""):
    """Click Next / Continue if present."""
    for btn_text in ["Next", "Continue", "Save and Continue", "Save & Continue"]:
        try:
            btn = page.get_by_role("button", name=btn_text, exact=False)
            if await btn.count() > 0:
                await btn.first.click()
                await _wait_nav(page, 8000)
                logger.debug(f"Clicked '{btn_text}' → moving to {section_hint}")
                return
        except Exception:
            continue


async def submit_form(page: Page, dry_run: bool = False) -> bool:
    """
    Click the final Submit button.
    In dry_run mode, takes a screenshot but does NOT click Submit.
    """
    await screenshot(page, "16_before_submit")

    if dry_run:
        logger.info("DRY RUN – skipping Submit click")
        return True

    for btn_text in ["Submit", "Save", "Finish", "Complete", "Done"]:
        try:
            btn = page.get_by_role("button", name=btn_text, exact=False)
            if await btn.count() > 0:
                await btn.first.click()
                await _wait_nav(page, 15000)
                await screenshot(page, "17_after_submit")
                logger.info(f"Form submitted (clicked '{btn_text}')")
                return True
        except Exception:
            continue

    logger.error("Could not find Submit button")
    return False


# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------

def print_summary(name: str, form_type: str, filled: list[str], failed: list[str]):
    table = Table(title=f"{name} — {form_type.title()} Form Summary")
    table.add_column("Status", style="bold")
    table.add_column("Field")
    for f in filled:
        table.add_row("[green]FILLED[/green]", f)
    for f in failed:
        table.add_row("[yellow]NOT FOUND[/yellow]", f)
    console.print(table)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def process_missionary(result: ProcessingResult, dry_run: bool = False):
    """
    Open browser, log in, navigate to the missionary, and fill medical + dental forms.
    """
    from .credentials import ensure_credentials

    config = load_config()
    username, password = ensure_credentials()

    async with async_playwright() as pw:
        browser: Browser = await pw.chromium.launch(
            headless=config.get("browser_headless", False),
            slow_mo=200,  # slight delay so user can follow along
        )
        context: BrowserContext = await browser.new_context(
            viewport={"width": 1280, "height": 900},
        )
        page: Page = await context.new_page()

        try:
            await login(page, username, password)

            missionary_name = result.missionary_name
            console.rule(f"[bold cyan]{missionary_name}[/bold cyan]")

            found = await find_and_click_candidate(page, missionary_name)
            if not found:
                console.print(f"[red]Could not locate '{missionary_name}' on the candidates page.[/red]")
                if dry_run:
                    raise RuntimeError(
                        f"[dry-run] Candidate '{missionary_name}' not found on website. "
                        "Re-run with a candidate currently listed in the candidates page."
                    )
                console.print("Please navigate to the correct candidate manually and press Enter.")
                _pause("Press Enter when you're on the candidate's page …")

            delay_ms = config.get("form_submit_delay_ms", 500)

            # ── Medical ──────────────────────────────────────────────────
            if result.has_medical and result.medical:
                clicked = await click_form_tab(page, "medical")
                if not clicked:
                    console.print("[yellow]Could not auto-navigate to Medical tab. Please click it manually.[/yellow]")
                    _pause("Press Enter when you're on the Medical form …")

                filled, failed = await fill_medical_form(page, result.medical)
                print_summary(missionary_name, "Medical", filled, failed)

                if config.get("confirm_before_submit") and not dry_run:
                    if not Confirm.ask(f"\nSubmit MEDICAL form for [bold]{missionary_name}[/bold]?"):
                        console.print("[yellow]Medical form submission skipped.[/yellow]")
                    else:
                        await submit_form(page, dry_run=dry_run)
                else:
                    await submit_form(page, dry_run=dry_run)

                await page.wait_for_timeout(delay_ms)

            # ── Dental ───────────────────────────────────────────────────
            if result.has_dental and result.dental:
                # Navigate back to candidate if needed
                if result.has_medical:
                    await find_and_click_candidate(page, missionary_name)

                clicked = await click_form_tab(page, "dental")
                if not clicked:
                    console.print("[yellow]Could not auto-navigate to Dental tab. Please click it manually.[/yellow]")
                    _pause("Press Enter when you're on the Dental form …")

                filled, failed = await fill_dental_form(page, result.dental)
                print_summary(missionary_name, "Dental", filled, failed)

                if config.get("confirm_before_submit") and not dry_run:
                    if not Confirm.ask(f"\nSubmit DENTAL form for [bold]{missionary_name}[/bold]?"):
                        console.print("[yellow]Dental form submission skipped.[/yellow]")
                    else:
                        await submit_form(page, dry_run=dry_run)
                else:
                    await submit_form(page, dry_run=dry_run)

        except Exception as e:
            logger.exception(f"Automation error: {e}")
            await screenshot(page, "ERROR")
            console.print(f"\n[red]Error during automation: {e}[/red]")
            console.print("The browser is still open. You can complete the form manually.")
            _pause("Press Enter to close the browser …")
        finally:
            await context.close()
            await browser.close()


def run_for_result(result: ProcessingResult, dry_run: bool = False):
    """Synchronous wrapper around the async automation."""
    asyncio.run(process_missionary(result, dry_run=dry_run))
