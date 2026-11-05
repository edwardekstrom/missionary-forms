"""
Application configuration management.
Settings stored in ~/missionary-forms/config.json
"""

import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
INPUT_DIR = BASE_DIR / "input"
PROCESSED_DIR = BASE_DIR / "processed"
LOGS_DIR = BASE_DIR / "logs"
SCREENSHOTS_DIR = LOGS_DIR / "screenshots"
VERIFIED_DIR = BASE_DIR / "verified"
CONFIG_FILE = BASE_DIR / "config.json"

CHURCH_URL = "https://missionaryrecommendations.churchofjesuschrist.org/recommendations/home/candidates?vctype=ft"
CHURCH_LOGIN_URL = "https://missionaryrecommendations.churchofjesuschrist.org"

DEFAULT_CONFIG = {
    "confirm_before_submit": False,     # Ask for confirmation before submitting each form
    "take_screenshots": True,           # Save screenshots during automation
    "ollama_model": "qwen2.5vl",        # Ollama model for vision-based OCR fallback
    "ollama_enabled": True,             # Whether to use Ollama as OCR fallback
    "gemini_enabled": False,            # Use Gemini Vision as primary parser (requires GEMINI_API_KEY)
    "gemini_model": "gemini-1.5-flash",  # Gemini model to use
    "gemini_api_key": "",               # Gemini API key (set via environment or config)
    "openrouter_enabled": False,        # Use OpenRouter Vision as fallback parser (requires OPENROUTER_API_KEY)
    "openrouter_model": "openai/gpt-4o",  # OpenRouter model to use
    "openrouter_api_key": "",           # OpenRouter API key (set via environment or config)
    "claude_enabled": False,            # Use Claude Vision as fallback parser (requires ANTHROPIC_API_KEY)
    "claude_model": "claude-3-5-sonnet-20241022",  # Claude model to use
    "claude_api_key": "",               # Claude API key (set via environment or config)
    "tesseract_lang": "eng",            # Tesseract language
    "browser_headless": False,          # Show browser (True = hidden)
    "form_submit_delay_ms": 500,        # Delay between form submissions (ms)
    "log_level": "INFO",
    # Parser pipeline — controls order and which parsers are tried.
    # Options: "ollama-text", "ollama", "docling", "regex"  (regex is always last)
    #   "ollama-text" = Ollama with Apple Vision OCR text (fast ~5-20s total)
    #   "ollama"      = Ollama vision on rendered images (slow ~45-120s/page)
    #   "docling"     = Granite-docling VLM (good for typed docs; poor for handwritten/checkbox forms)
    "parser_order": ["ollama-text", "ollama", "regex"],
    "docling_enabled": False,           # Granite-docling VLM; disabled by default for handwritten forms
    "docling_ollama_text_pass": True,   # If docling text is sparse, also run Ollama text-only
}


def ensure_dirs():
    for d in [INPUT_DIR, PROCESSED_DIR, LOGS_DIR, SCREENSHOTS_DIR, VERIFIED_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    ensure_dirs()
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            stored = json.load(f)
        # Merge with defaults so new keys are always present
        return {**DEFAULT_CONFIG, **stored}
    return DEFAULT_CONFIG.copy()


def save_config(config: dict):
    ensure_dirs()
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
