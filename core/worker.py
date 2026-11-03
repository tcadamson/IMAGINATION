"""Run a bot of the bot bundle in a process of its own.

IMAGINATION starts its own program again with `FLAG` to run a bundled bot, so that a new
bundle is used from the next run, and the bundle stays out of the console's process.
The command line after `FLAG` is the bundle file, the bot, the client's process ID, and
the bot's options as a JSON object.

With `NAMES_FLAG` instead, the command line is the bundle file and the process IDs of
the clients open, and the worker sends the names of their characters, for the user to
choose a client by.

The bundle is a module, compiled or not, with `API_VERSION` and `run(bot, options, pid)`.
`run` returns 0 when the bot finished and 1 when it stopped for a reason it logged, lets
KeyboardInterrupt out on Ctrl+C, and sets up no output of its own: it logs, and prints
what it was run to show. It may also have `character_name(pid)`, the name of the
character logged in on that client, or None; without it, the clients go by process ID.

The worker sends every log record and every printed line to stdout as one line of JSON
each, in UTF-8, for IMAGINATION to log and print as its own (core.bundle). A record is
{"kind": "log", "name", "level", "created", "message", "exc"}; a printed line is
{"kind": "print", "text"}; the characters' names are {"kind": "names", "names"}, each
process ID as text to its character's name or null.
"""

import collections.abc
import enum
import importlib.util
import io
import json
import logging
import pathlib
import sys
import threading
import types
import typing

FLAG: typing.Final = "--worker"
NAMES_FLAG: typing.Final = "--worker-names"

API_VERSIONS: typing.Final = frozenset({1})

_MODULE_NAME: typing.Final = "bundle"  # The name the bundle is compiled as

_logger: logging.Logger = logging.getLogger(__name__)


class ExitCode(enum.IntEnum):
    """How a worker ended."""

    FINISHED = 0
    STOPPED = 1  # The bot stopped for a reason it logged
    ABORTED = 2  # The user pressed Ctrl+C
    NOT_STARTED = 3  # The bundle could not be loaded, or speaks another API version
    CRASHED = 4  # The bot raised an exception, logged with its traceback


class _Channel:
    """JSON lines to a stream, written whole across threads."""

    def __init__(self, stream: typing.TextIO):
        self._stream = stream
        self._lock = threading.Lock()

    def send(self, message: collections.abc.Mapping[str, object]) -> None:
        """Write `message` as one line of JSON."""
        line = json.dumps(message, ensure_ascii=False)
        with self._lock:
            self._stream.write(line + "\n")
            self._stream.flush()


class _RecordHandler(logging.Handler):
    """Log handler sending each record over a channel."""

    def __init__(self, channel: _Channel):
        super().__init__(logging.DEBUG)
        self._channel = channel
        self._exception_formatter = logging.Formatter()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._channel.send(
                {
                    "kind": "log",
                    "name": record.name,
                    "level": record.levelno,
                    "created": record.created,
                    "message": record.getMessage(),
                    "exc": self._exception_formatter.formatException(record.exc_info)
                    if record.exc_info
                    else None,
                }
            )
        except Exception:
            self.handleError(record)


class _PrintWriter(io.TextIOBase):
    """Stand-in for sys.stdout sending each printed line over a channel."""

    def __init__(self, channel: _Channel):
        super().__init__()
        self._channel = channel
        self._pending = ""
        self._lock = threading.Lock()

    def writable(self) -> bool:
        return True

    def write(self, s: str, /) -> int:
        with self._lock:
            *lines, self._pending = (self._pending + s).split("\n")
        for line in lines:
            self._channel.send({"kind": "print", "text": line})
        return len(s)

    def flush(self) -> None:
        """Send a line left without its newline, as a line of its own."""
        with self._lock:
            pending, self._pending = self._pending, ""

        if pending:
            self._channel.send({"kind": "print", "text": pending})


def _load(path: pathlib.Path) -> types.ModuleType:
    """Import the bundle at `path`, an extension module or a Python source file."""
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, path)

    if spec is None or spec.loader is None:
        raise ImportError(f"no loadable module spec for {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _run(bundle_path: str, bot_id: str, pid: str, options: str) -> ExitCode:
    """Load the bundle and run the bot, returning how it ended."""
    try:
        bot_options = json.loads(options)
        client_pid = int(pid)

        if not isinstance(bot_options, dict):
            raise ValueError("options are not a JSON object")
    except ValueError:
        _logger.exception("can't start bot %s: malformed command line", bot_id)
        return ExitCode.NOT_STARTED

    try:
        bundle = _load(pathlib.Path(bundle_path))
    except Exception:
        _logger.exception("can't load the bot bundle %s", bundle_path)
        return ExitCode.NOT_STARTED

    api_version = getattr(bundle, "API_VERSION", None)

    if api_version not in API_VERSIONS:
        _logger.error(
            "can't run bot %s: the bot bundle is for another version of IMAGINATION "
            "(API version %s, not %s)",
            bot_id,
            api_version,
            " or ".join(str(version) for version in sorted(API_VERSIONS)),
        )
        return ExitCode.NOT_STARTED

    try:
        status = bundle.run(bot_id, bot_options, client_pid)
    except Exception:
        _logger.exception("bot %s crashed", bot_id)
        return ExitCode.CRASHED

    return ExitCode.FINISHED if status == 0 else ExitCode.STOPPED


def _names(bundle_path: str, pids: collections.abc.Sequence[str]) -> dict[str, object]:
    """Read the name of the character on each client of `pids` with the bundle; none
    when the bundle can't tell. Every failure is a debug line: the clients then go by
    process ID."""
    try:
        bundle = _load(pathlib.Path(bundle_path))
    except Exception:
        _logger.debug("can't load the bot bundle %s", bundle_path, exc_info=True)
        return {}

    character_name = getattr(bundle, "character_name", None)

    if (
        getattr(bundle, "API_VERSION", None) not in API_VERSIONS
        or character_name is None
    ):
        _logger.debug("the bot bundle can't name the characters")
        return {}

    names = {}

    for pid in pids:
        try:
            names[pid] = character_name(int(pid))
        except Exception:
            _logger.debug("can't read pid %s's character", pid, exc_info=True)

    return names


def _connect() -> _Channel:
    """Send this process's log records and printed lines to stdout, as JSON lines."""
    for stream in (sys.stdout, sys.stderr):
        typing.cast(io.TextIOWrapper, stream).reconfigure(encoding="utf-8")

    channel = _Channel(sys.stdout)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(_RecordHandler(channel))
    sys.stdout = _PrintWriter(channel)
    return channel


def names_main(argv: collections.abc.Sequence[str]) -> int:
    """Send the names of the characters on the clients that `argv`, the command line
    after `NAMES_FLAG`, names after the bundle file."""
    channel = _connect()

    try:
        if not argv:
            _logger.debug("can't name the characters: no bundle file")
            return ExitCode.NOT_STARTED

        channel.send({"kind": "names", "names": _names(argv[0], argv[1:])})
        return ExitCode.FINISHED
    except KeyboardInterrupt:
        return ExitCode.ABORTED
    finally:
        sys.stdout.flush()


def main(argv: collections.abc.Sequence[str]) -> int:
    """Run the bundled bot that `argv`, the command line after `FLAG`, names."""
    _connect()

    try:
        if len(argv) != 4:
            _logger.error(
                "can't start a worker: expected the bundle file, bot, process ID, and "
                "options, not %s",
                " ".join(argv),
            )
            return ExitCode.NOT_STARTED

        return _run(*argv)
    except KeyboardInterrupt:
        return ExitCode.ABORTED
    finally:
        sys.stdout.flush()
