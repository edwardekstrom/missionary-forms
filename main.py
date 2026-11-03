#!/usr/bin/env python3
"""
Missionary Forms Processor
--------------------------
Automatically extracts data from medical/dental PDF forms and enters it into
the LDS Missionary Recommendations website.

Commands:
  setup    — First-time setup: save church credentials
  watch    — Watch input/ folder and process PDFs automatically  (default)
  process  — Manually process a specific PDF file
  preview  — Extract and display data from a PDF without opening browser
  scrape   — Scrape submitted values from website → ground_truth.json (run once)
  validate — Compare PDF extraction accuracy against saved ground truth
  config   — Show or edit current configuration
"""

import sys
import time
from pathlib import Path
from typing import Optional

import click
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Adjust log level based on --verbose flag before importing submodules
console = Console()

# ---------------------------------------------------------------------------
# Shared processing pipeline
# ---------------------------------------------------------------------------

def process_pdf(pdf_path: Path, dry_run: bool = False, auto_submit: bool = False) -> bool:
    """
    Full pipeline for a single PDF:
      extract → parse → display preview → (optionally) submit to website
    Returns True if processing completed without fatal errors.
    """
    from app.config import PROCESSED_DIR, load_config
    from app.file_watcher import move_to_processed
    from app.form_parser import parse_forms
    from app.pdf_processor import (
        combine_pages,
        detect_form_types,
        extract_pdf_text,
        parse_missionary_name_from_filename,
    )
    from app.web_automator import run_for_result

    cfg = load_config()

    console.rule(f"[bold cyan]{pdf_path.name}[/bold cyan]")

    # 1. Extract text
    console.print(f"[dim]Extracting text from PDF…[/dim]")
    try:
        pages = extract_pdf_text(pdf_path)
    except Exception as e:
        console.print(f"[red]PDF extraction failed: {e}[/red]")
        return False

    full_text = combine_pages(pages)

    # 2. Determine name and form types
    missionary_name = parse_missionary_name_from_filename(pdf_path.name)
    has_medical, has_dental = detect_form_types(pdf_path.name, full_text)

    console.print(f"[bold]Missionary:[/bold] {missionary_name}")
    console.print(f"[bold]Forms detected:[/bold] "
                  f"{'[green]Medical[/green]' if has_medical else ''} "
                  f"{'[blue]Dental[/blue]' if has_dental else ''}")

    if not has_medical and not has_dental:
        console.print("[yellow]Warning: could not detect Medical or Dental forms in this PDF.[/yellow]")

    # 3. Parse structured data
    console.print("[dim]Parsing form fields…[/dim]")
    from app.form_parser import parse_forms
    result = parse_forms(
        text=full_text,
        missionary_name=missionary_name,
        pdf_path=str(pdf_path),
        has_medical=has_medical,
        has_dental=has_dental,
    )

    # 4. Preview extracted data
    _print_preview(result)

    if result.errors:
        for err in result.errors:
            console.print(f"[red]Error: {err}[/red]")

    # 5. Submit to website
    if dry_run:
        console.print("\n[yellow]DRY RUN – browser will open but form will NOT be submitted.[/yellow]")

    if not auto_submit and not dry_run:
        from rich.prompt import Confirm
        if not Confirm.ask("\nOpen browser and submit to church website?"):
            console.print("[dim]Skipped. PDF left in input/ folder.[/dim]")
            return True

    run_for_result(result, dry_run=dry_run)

    # 6. Move PDF to processed/
    if not dry_run:
        try:
            move_to_processed(pdf_path)
        except Exception as e:
            console.print(f"[yellow]Could not move PDF to processed/: {e}[/yellow]")

    return True


def _print_preview(result):
    """Pretty-print the extracted form data so the user can verify before submitting."""
    from rich.tree import Tree

    tree = Tree(f"[bold]{result.missionary_name}[/bold]")

    if result.medical:
        m = result.medical
        pi = m.personal_info
        med = tree.add("[green]Medical Form[/green]")
        med.add(f"Name: {pi.first_name} {pi.middle_name or ''} {pi.last_name}".strip())
        med.add(f"DOB: {pi.date_of_birth or '?'}   Age: {pi.age or '?'}   Gender: {pi.gender or '?'}")
        med.add(f"Record #: {pi.record_number or '?'}")
        ph = m.physician_info
        med.add(f"Physician: {ph.name or '?'} {ph.credentials or ''} – {ph.city or '?'}, {ph.state or '?'}")
        med.add(f"Functional Level: {m.functional_level or '?'}")
        med.add(f"Chronic Condition: {m.chronic_condition or '?'}")
        vs = m.vital_signs
        med.add(f"H/W: {vs.height or '?'} / {vs.weight or '?'}   BP: {vs.blood_pressure or '?'}   Pulse: {vs.pulse or '?'}")
        med.add(f"Vision: OD {vs.vision_right or '?'}  OS {vs.vision_left or '?'}")
        labs = m.labs
        med.add(f"Labs: Hgb {labs.hemoglobin or '?'}  TB risk {labs.tb_exposure_risk or '?'}  IGRA {labs.igra_result or '?'}")

    if result.dental:
        d = result.dental
        pi = d.personal_info
        dent = tree.add("[blue]Dental Form[/blue]")
        dent.add(f"Name: {pi.first_name} {pi.middle_name or ''} {pi.last_name}".strip())
        dent.add(f"Dentist: {d.dentist_name or '?'}  – {d.dentist_city or '?'}, {d.dentist_state or '?'}")
        dent.add(f"Brushing: {d.brushing_frequency or '?'}   Flossing: {d.flossing_frequency or '?'}")
        dent.add(f"Ortho: {d.orthodontic_treatment or '?'}   Wisdom teeth removed: {d.wisdom_teeth_removed or '?'}")
        dent.add(f"Pain/Bleeding: {d.pain_or_bleeding or '?'}   Decay: {d.dental_decay or '?'}")
        dent.add(f"Oral hygiene acceptable: {d.oral_hygiene_acceptable or '?'}")

    console.print(tree)


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

@click.group(invoke_without_command=True)
@click.pass_context
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def cli(ctx, verbose):
    """Missionary Forms Processor — drop PDFs in input/ to get started."""
    level = "DEBUG" if verbose else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=level, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
    logger.add(
        "logs/processor.log",
        rotation="5 MB",
        retention="30 days",
        level="DEBUG",
        format="{time} | {level} | {message}",
    )

    if ctx.invoked_subcommand is None:
        ctx.invoke(watch)


@cli.command()
def setup():
    """Save your Church account credentials to macOS Keychain."""
    from app.credentials import prompt_and_save_credentials
    prompt_and_save_credentials()
    console.print("\n[green]Setup complete! Run [bold]python main.py watch[/bold] to start.[/green]")


@cli.command()
@click.option("--dry-run", is_flag=True, help="Open browser but do NOT submit forms")
def watch(dry_run):
    """Watch the input/ folder and process new PDFs automatically."""
    from app.config import INPUT_DIR, ensure_dirs
    from app.credentials import ensure_credentials
    from app.file_watcher import start_watcher

    ensure_dirs()
    ensure_credentials()  # Prompt once if not stored

    console.print(Panel(
        f"[bold green]Watching for PDFs in:[/bold green]\n{INPUT_DIR}\n\n"
        "Drop missionary PDF files into that folder to process them.\n"
        "Press [bold]Ctrl+C[/bold] to stop.",
        title="Missionary Forms Processor",
        expand=False,
    ))

    def on_pdf(path: Path):
        process_pdf(path, dry_run=dry_run, auto_submit=True)

    observer = start_watcher(on_pdf)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        observer.join()
        console.print("\n[dim]Watcher stopped.[/dim]")


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True, path_type=Path))
@click.option("--dry-run", is_flag=True, help="Open browser but do NOT submit forms")
def process(pdf_path: Path, dry_run: bool):
    """Process a specific PDF file."""
    from app.config import ensure_dirs
    from app.credentials import ensure_credentials

    ensure_dirs()
    ensure_credentials()
    process_pdf(pdf_path, dry_run=dry_run)


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True, path_type=Path))
def preview(pdf_path: Path):
    """Extract and display data from a PDF (no browser, no submission)."""
    from app.config import ensure_dirs
    from app.form_parser import parse_forms
    from app.pdf_processor import (
        combine_pages,
        detect_form_types,
        extract_pdf_text,
        parse_missionary_name_from_filename,
    )

    ensure_dirs()
    console.rule(f"[bold cyan]{pdf_path.name}[/bold cyan]")

    pages = extract_pdf_text(pdf_path)
    full_text = combine_pages(pages)
    missionary_name = parse_missionary_name_from_filename(pdf_path.name)
    has_medical, has_dental = detect_form_types(pdf_path.name, full_text)

    result = parse_forms(
        text=full_text,
        missionary_name=missionary_name,
        pdf_path=str(pdf_path),
        has_medical=has_medical,
        has_dental=has_dental,
    )

    _print_preview(result)

    # Also print the raw OCR text for debugging
    console.print("\n[dim]──── Raw extracted text (first 3 pages) ────[/dim]")
    for p in pages[:3]:
        console.print(f"\n[dim]Page {p['page']} ({p['method']}):[/dim]")
        console.print(p["text"][:1500])


@cli.command("config")
@click.option("--set", "key_value", nargs=2, metavar="KEY VALUE",
              help="Set a config value (e.g. --set confirm_before_submit false)")
def config_cmd(key_value):
    """Show or update configuration."""
    from app.config import load_config, save_config

    cfg = load_config()

    if key_value:
        key, value = key_value
        if key not in cfg:
            console.print(f"[red]Unknown config key: {key}[/red]")
            return
        # Type coercion
        existing = cfg[key]
        if isinstance(existing, bool):
            cfg[key] = value.lower() in ("true", "1", "yes")
        elif isinstance(existing, int):
            cfg[key] = int(value)
        else:
            cfg[key] = value
        save_config(cfg)
        console.print(f"[green]Set {key} = {cfg[key]}[/green]")
    else:
        table = Table(title="Current Configuration")
        table.add_column("Key", style="bold")
        table.add_column("Value")
        for k, v in cfg.items():
            table.add_row(k, str(v))
        console.print(table)


@cli.command()
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Output JSON path (default: logs/ground_truth.json)",
)
def scrape(output: Optional[Path]):
    """
    Scrape submitted form values from the church website and save as ground truth.

    Logs in, navigates to 'Ready for Stake President's Action' candidates,
    opens each missionary's Medical and Dental form, reads the submitted values,
    and saves them to a JSON file for fast offline validation.

    Run this once (or after entering new data) before running 'validate'.
    """
    from app.config import ensure_dirs, LOGS_DIR
    from app.credentials import ensure_credentials
    from app.validator import run_scrape_ground_truth

    ensure_dirs()
    ensure_credentials()

    log_path = LOGS_DIR / "scrape_run.log"
    logger.add(str(log_path), level="INFO", mode="w", buffering=1, colorize=False)
    console.print(f"[dim]Live log: {log_path}[/dim]")

    out = run_scrape_ground_truth(output_path=output)
    console.print(f"\n[green bold]Ground truth saved:[/green bold] {out}")
    console.print("Now run [bold]python main.py validate[/bold] to compare PDFs against it.")


@cli.command()
@click.option(
    "--pdf-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Folder of PDFs to validate (default: downloads/All folder)",
)
@click.option(
    "--ground-truth",
    type=click.Path(path_type=Path),
    default=None,
    help="Ground truth JSON from 'scrape' command (default: logs/ground_truth.json). "
         "If omitted, falls back to live website scraping.",
)
@click.option(
    "--skip-parse",
    is_flag=True,
    default=False,
    help="Skip Ollama PDF parse — only test site navigation and scraping (live mode only).",
)
def validate(pdf_dir: Optional[Path], ground_truth: Optional[Path], skip_parse: bool):
    """
    Compare PDF extraction accuracy against already-submitted website data.

    By default loads ground truth from logs/ground_truth.json (created by
    'scrape' command) — no browser required.  Pass --ground-truth to use a
    different JSON file, or omit it entirely to fall back to live website scraping.
    """
    from app.config import ensure_dirs, LOGS_DIR
    from app.credentials import ensure_credentials
    from app.validator import GROUND_TRUTH_PATH, run_validation, run_validation_offline

    ensure_dirs()

    if pdf_dir is None:
        pdf_dir = Path(
            "/Users/eddy.ekstrom/Downloads/"
            "Prospective Missionary_s Documents (File responses)/All"
        )

    log_path = LOGS_DIR / "validate_run.log"
    logger.add(str(log_path), level="INFO", mode="w", buffering=1, colorize=False)
    console.print(f"[bold]Validating PDFs in:[/bold] {pdf_dir}")
    console.print(f"[dim]Live log: {log_path}[/dim]")

    # Use saved ground truth if available; otherwise fall back to live scraping
    gt_path = ground_truth or GROUND_TRUTH_PATH
    if gt_path.exists():
        console.print(f"[dim]Using saved ground truth: {gt_path}[/dim]")
        run_validation_offline(pdf_dir, ground_truth_path=gt_path)
    else:
        ensure_credentials()
        console.print("[dim]No ground truth file found — falling back to live website scraping.[/dim]")
        console.print("[dim]Tip: run 'python main.py scrape' first for faster offline validation.[/dim]")
        run_validation(pdf_dir, skip_parse=skip_parse)


@cli.command()
def clear_credentials():
    """Remove stored credentials from macOS Keychain."""
    from app.credentials import clear_credentials as _clear
    _clear()


if __name__ == "__main__":
    cli()
