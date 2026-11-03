"""
Watchdog-based directory monitor.
Watches the input/ folder and processes new PDFs automatically.
"""

import time
from pathlib import Path
from threading import Timer
from typing import Callable

from loguru import logger
from watchdog.events import FileCreatedEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .config import INPUT_DIR, PROCESSED_DIR


class _PDFHandler(FileSystemEventHandler):
    """
    Fires process_callback after a PDF appears in the watched directory.
    Uses a short debounce so large file copies finish before processing starts.
    """

    DEBOUNCE_SECONDS = 3.0

    def __init__(self, process_callback: Callable[[Path], None]):
        super().__init__()
        self._callback = process_callback
        self._pending: dict[str, Timer] = {}

    def on_created(self, event: FileCreatedEvent):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() != ".pdf":
            return

        logger.info(f"Detected new PDF: {path.name}")

        # Cancel any existing timer for this path and restart
        key = str(path)
        if key in self._pending:
            self._pending[key].cancel()

        timer = Timer(self.DEBOUNCE_SECONDS, self._fire, args=[path])
        self._pending[key] = timer
        timer.start()

    def _fire(self, path: Path):
        self._pending.pop(str(path), None)
        if not path.exists():
            logger.warning(f"PDF disappeared before processing: {path.name}")
            return
        try:
            self._callback(path)
        except Exception as e:
            logger.error(f"Error processing {path.name}: {e}")


def move_to_processed(pdf_path: Path):
    """Move a processed PDF to the processed/ directory."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dest = PROCESSED_DIR / pdf_path.name
    # Avoid overwriting if a file with the same name already exists
    if dest.exists():
        stem = pdf_path.stem
        suffix = pdf_path.suffix
        ts = int(time.time())
        dest = PROCESSED_DIR / f"{stem}_{ts}{suffix}"
    pdf_path.rename(dest)
    logger.info(f"Moved to processed/: {dest.name}")


def start_watcher(process_callback: Callable[[Path], None]) -> Observer:
    """
    Start watching the input/ directory.
    Returns the Observer so the caller can stop it.
    """
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    handler = _PDFHandler(process_callback)
    observer = Observer()
    observer.schedule(handler, str(INPUT_DIR), recursive=False)
    observer.start()
    logger.info(f"Watching for PDFs in: {INPUT_DIR}")
    return observer
