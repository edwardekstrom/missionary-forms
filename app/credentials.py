"""
Secure credential storage using macOS Keychain via the keyring library.
Credentials are never written to disk in plain text.
"""

import keyring
from rich.console import Console
from rich.prompt import Prompt

SERVICE_NAME = "MissionaryFormsProcessor"
console = Console()


def save_credentials(username: str, password: str):
    keyring.set_password(SERVICE_NAME, "username", username)
    keyring.set_password(SERVICE_NAME, "password", password)
    console.print("[green]Credentials saved to macOS Keychain.[/green]")


def get_credentials() -> tuple[str | None, str | None]:
    username = keyring.get_password(SERVICE_NAME, "username")
    password = keyring.get_password(SERVICE_NAME, "password")
    return username, password


def clear_credentials():
    try:
        keyring.delete_password(SERVICE_NAME, "username")
        keyring.delete_password(SERVICE_NAME, "password")
        console.print("[yellow]Credentials cleared from Keychain.[/yellow]")
    except keyring.errors.PasswordDeleteError:
        console.print("[yellow]No credentials found to clear.[/yellow]")


def prompt_and_save_credentials():
    """Interactive prompt to collect and save credentials."""
    console.print("\n[bold cyan]Church Account Login Setup[/bold cyan]")
    console.print("Your credentials will be stored securely in macOS Keychain.\n")
    username = Prompt.ask("[bold]Church username[/bold] (username or email)")
    password = Prompt.ask("[bold]Password[/bold]", password=True)
    save_credentials(username, password)

    # Offer to save Anthropic API key
    console.print("\n[bold cyan]Anthropic API Key (for Claude Vision parsing)[/bold cyan]")
    console.print("Paste your key from [link]https://console.anthropic.com/settings/api-keys[/link]")
    console.print("[dim]Press Enter to skip — you can add it later with: python main.py setup[/dim]")
    api_key = Prompt.ask("[bold]ANTHROPIC_API_KEY[/bold]", password=True, default="")
    if api_key.strip():
        save_anthropic_key(api_key.strip())

    return username, password


# ---------------------------------------------------------------------------
# Anthropic API key
# ---------------------------------------------------------------------------

def save_anthropic_key(api_key: str):
    keyring.set_password(SERVICE_NAME, "anthropic_api_key", api_key)
    console.print("[green]Anthropic API key saved to macOS Keychain.[/green]")


def get_anthropic_key() -> str | None:
    return keyring.get_password(SERVICE_NAME, "anthropic_api_key")


def ensure_credentials() -> tuple[str, str]:
    """Return credentials, prompting if not yet stored."""
    username, password = get_credentials()
    if not username or not password:
        console.print("[yellow]No credentials stored. Please enter your Church account credentials.[/yellow]")
        username, password = prompt_and_save_credentials()
    return username, password
