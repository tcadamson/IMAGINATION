"""Console and log file output for IMAGINATION.

What a command was run to show is printed; everything else is logged. The console shows
INFO and up, and DEBUG too with run --verbose; a debug line, a warning, or an error has its
level in front. The log file keeps every level, rotating, so a user's report can include
what the bots saw and did.

The wording, the same everywhere, the bots' lines and the help texts included:
- Capitals only where a line or a bullet starts, and on names. After a colon, lower case
  unless a name follows, "Debug:", "Warning:", and "Error:" included: the console puts
  the level in front of the message as it is, so debug lines, exception messages and
  logged problems start in lower case. No full stop at a line's end; one between two sentences.
- A colon after a label or between a statement and its reason, a semicolon before what
  to do: "what: why; what to do". The Oxford comma in a list of three or more: "A, B,
  and C". A refusal to start is "Can't start:" with a line for
  each reason.
- Thousands separators on every count, and `count` for its word.
- No Python reprs in a line meant for the user.
Typer's own messages keep their words, but start in lower case after "Error:" and lose
their full stop, as the console shows them (main._show_usage_error).
"""

import collections.abc
import contextlib
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

_printed: logging.Logger = logging.getLogger(
    "printed"
)  # What a bot prints, for the log file alone: the console has it already


def capitalize(text: str) -> str:
    """Return `text` with its first character in upper case, the rest unchanged."""
    return text[:1].upper() + text[1:]


def count(n: int, word: str, plural: str | None = None) -> str:
    """Return `n` with thousands separators and its `word`, singular for 1.

    count(3, "preset") is "3 presets".
    """
    return f"{n:,} {word if n == 1 else plural or word + 's'}"


class _ConsoleFormatter(logging.Formatter):
    """The message alone; a debug line, a warning, or an error gets its level in front."""

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)

        if record.levelno == logging.INFO:
            return text

        return f"{record.levelname.capitalize()}: {text}"


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
    _printed.propagate = False
    _printed.addHandler(log_file)


def show(text: str) -> None:
    """Print `text`, a line a bot was run to show, and log it to the log file, so that a
    log sent with a report holds the bot's results too."""
    print(text)
    _printed.info("%s", text)


@contextlib.contextmanager
def verbose(enabled: bool = True) -> collections.abc.Iterator[None]:
    """Show DEBUG records on the console within the context, if `enabled`."""
    if _console is None or not enabled:
        yield
        return

    level = _console.level
    _console.setLevel(logging.DEBUG)

    try:
        yield
    finally:
        _console.setLevel(level)
