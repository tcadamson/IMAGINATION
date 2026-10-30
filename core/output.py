"""Console and log file output for IMAGINATION.

What a command was run to show is printed; everything else is logged. The console shows
INFO and up, a warning or an error with its level in front. The log file keeps every
level, rotating, so a user's report can include what the bots saw and did.
"""

import io
import logging
import logging.handlers
import sys
import typing

import core.paths

LOG_PATH: typing.Final = core.paths.USER_DIRECTORY / "IMAGINATION.log"

_LOG_BYTES: typing.Final = 10_000_000
_LOG_BACKUPS: typing.Final = 3

_console: logging.Handler | None = None


def capitalize(text: str) -> str:
    """Return `text` with its first character in upper case, the rest unchanged."""
    return text[:1].upper() + text[1:]


class _ConsoleFormatter(logging.Formatter):
    """The message alone; a warning or an error gets its level in front, and a capital."""

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)

        if record.levelno < logging.WARNING:
            return text

        return f"{record.levelname.capitalize()}: {capitalize(text)}"


def setup() -> None:
    """Send log records to the console and the log file.

    Call once, before anything is printed. The standard streams are reconfigured to
    UTF-8: redirected to a file or a pipe, Python would use the Windows code page and
    fail on text outside it.
    """
    global _console

    for stream in (sys.stdout, sys.stderr):
        typing.cast(io.TextIOWrapper, stream).reconfigure(encoding="utf-8")

    _console = logging.StreamHandler(
        sys.stdout
    )  # Same stream as print, so log lines and printed results keep their order
    _console.setLevel(logging.INFO)
    _console.setFormatter(_ConsoleFormatter("%(message)s"))
    LOG_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )  # An IMAGINATION_USER_DIRECTORY override may not exist yet
    log_file = logging.handlers.RotatingFileHandler(
        LOG_PATH, maxBytes=_LOG_BYTES, backupCount=_LOG_BACKUPS, encoding="utf-8"
    )
    log_file.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(_console)
    root.addHandler(log_file)
