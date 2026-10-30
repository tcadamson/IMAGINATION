"""Register the bot bundle's bots from their catalog, and run them in worker processes.

The catalog, resources/bots.json, names the bundle by its sha256 and lists its bots, each
with its help and options. An option is typed on the console as --name-with-dashes, and
reaches the bundle by its name, of its type, with its default filled in:

    {"bundle": "<sha256>",
     "bots": {"<bot>": {"help": "...",
                        "options": {"<name>": {"type": "int", "default": 1,
                                               "min": 1, "max": 4, "help": "..."}}}}}

`type` is int, float, str, or bool (a flag, false by default). Without a `default`, the
option is required; a null default makes it optional, passed as None. `min`, `max`, and
`hidden` may be left out.

A bot with `extra` true also takes options the catalog doesn't declare, each with a
value, which its help describes; they reach the bundle as typed, in a list under "extra".

A bundled bot runs in a worker (core.worker) attached to one client process. Its log
records and printed lines are logged and printed here as IMAGINATION's own.
"""

import collections.abc
import dataclasses
import hashlib
import json
import logging
import pathlib
import subprocess
import sys
import threading
import time
import typing

import core.api
import core.output
import core.paths
import core.worker

CATALOG_PATH: typing.Final = core.paths.USER_DIRECTORY / "resources" / "bots.json"

_OPTION_TYPES: typing.Final[collections.abc.Mapping[str, type]] = {
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
}

_STOP_SECONDS: typing.Final = (
    10.0  # Allowed to a worker after Ctrl+C before it is ended
)

_logger: logging.Logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Option:
    """One option of a bundled bot."""

    name: str
    type: type
    required: bool
    default: object = None
    help: str = ""
    hidden: bool = False
    min: float | None = None
    max: float | None = None


@dataclasses.dataclass(frozen=True)
class BundledBotSpec:
    """A bot of the bot bundle, as its catalog describes it."""

    bot_id: str
    help: str
    options: tuple[Option, ...]
    extra: bool  # Whether it takes options the catalog doesn't declare
    bundle_path: pathlib.Path


@dataclasses.dataclass(frozen=True)
class Outcome:
    """How a bundled bot's run ended, with the last error it logged."""

    exit_code: core.worker.ExitCode
    error: str | None


def _parse_option(name: str, data: collections.abc.Mapping[str, typing.Any]) -> Option:
    """Build an option from its catalog entry, raising if it is invalid."""
    if not name.isidentifier():
        raise ValueError(f"option name {name} is not an identifier")

    option_type = _OPTION_TYPES[data["type"]]
    return Option(
        name,
        option_type,
        required="default" not in data and option_type is not bool,
        default=data.get("default", False if option_type is bool else None),
        help=data.get("help", ""),
        hidden=data.get("hidden", False),
        min=data.get("min"),
        max=data.get("max"),
    )


def _installed_bundle(sha256: str) -> pathlib.Path | None:
    """Return the installed bundle file with this `sha256`, or None."""
    for path in core.paths.BUNDLE_DIRECTORY.glob(f"{sha256}.*"):
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == sha256:
            return path
    return None


def load_catalog(
    catalog_path: pathlib.Path = CATALOG_PATH,
) -> dict[str, BundledBotSpec]:
    """Read the bundled bots from the catalog at `catalog_path`, keyed by bot.

    Return none if there is no catalog, or, with a warning, if the bundle it names is
    not installed. Raise if the catalog is malformed.
    """
    try:
        data = json.loads(catalog_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exception:
        raise RuntimeError(f"malformed JSON at {catalog_path}") from exception

    try:
        bundle_path = _installed_bundle(data["bundle"])
        bots = {
            bot_id: (
                bot["help"],
                tuple(
                    _parse_option(name, option)
                    for name, option in bot["options"].items()
                ),
                bot.get("extra", False) is True,
            )
            for bot_id, bot in data["bots"].items()
        }
    except (KeyError, TypeError, ValueError) as exception:
        raise RuntimeError(f"invalid catalog at {catalog_path}") from exception

    if bundle_path is None:
        _logger.warning(
            "%s not available: the bot bundle is not installed",
            ", ".join(bots),
        )
        return {}

    return {
        bot_id: BundledBotSpec(bot_id, bot_help, options, extra, bundle_path)
        for bot_id, (bot_help, options, extra) in bots.items()
    }


def _command(
    spec: BundledBotSpec, options: collections.abc.Mapping[str, object], pid: int
) -> list[str]:
    """Command line starting a worker for `spec`: this program, again."""
    program = (
        [sys.executable]
        if getattr(sys, "frozen", False)
        else [sys.executable, str(core.api.ROOT_DIRECTORY / "main.py")]
    )
    return [
        *program,
        core.worker.FLAG,
        str(spec.bundle_path),
        spec.bot_id,
        str(pid),
        json.dumps(options),
    ]


class _Relay(threading.Thread):
    """Logs and prints a worker's output as it comes, keeping its last error."""

    def __init__(self, stream: typing.TextIO):
        super().__init__(daemon=True)
        self._stream = stream
        self.error: str | None = None

    def run(self) -> None:
        for line in self._stream:
            line = line.rstrip("\n")

            if line:
                self._relay(line)

    def _relay(self, line: str) -> None:
        """Log or print one line of the worker's output."""
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            message = None

        if not isinstance(
            message, dict
        ):  # Not the worker's own, such as a native crash
            _logger.error("%s", line)
            self.error = line
            return

        if message.get("kind") == "print":
            core.output.show(message["text"])
            return

        level, created, exc = message["level"], message["created"], message["exc"]
        record = logging.makeLogRecord(
            {
                "name": message["name"],
                "levelno": level,
                "levelname": logging.getLevelName(level),
                "msg": message["message"],
                "created": created,
                "msecs": (created - int(created)) * 1000,
                "exc_text": exc,
            }
        )
        logging.getLogger(record.name).handle(record)

        if level >= logging.ERROR:
            self.error = exc.rstrip().splitlines()[-1] if exc else message["message"]


def run(
    spec: BundledBotSpec, options: collections.abc.Mapping[str, object], pid: int
) -> Outcome:
    """Run the bundled bot `spec` against client process `pid` until it ends.

    Ctrl+C reaches the worker too, which stops the bot; if it has not ended a while
    after, it is ended here.
    """
    command = _command(spec, options, pid)
    _logger.debug("starting worker: %s", subprocess.list2cmdline(command))
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
    )
    relay = _Relay(typing.cast(typing.TextIO, process.stdout))
    relay.start()
    interrupted = None
    killed = False
    while True:
        try:
            returncode = process.wait(timeout=0.1)
            break
        except subprocess.TimeoutExpired:
            if (
                interrupted is not None
                and not killed
                and time.monotonic() - interrupted > _STOP_SECONDS
            ):
                _logger.warning(
                    "bot %s did not stop within %s seconds of Ctrl+C; ending it",
                    spec.bot_id,
                    f"{_STOP_SECONDS:g}",
                )
                process.kill()
                killed = True
        except KeyboardInterrupt:
            core.api.safe_abort()

            if interrupted is None:
                interrupted = time.monotonic()
    relay.join()

    if killed:
        return Outcome(core.worker.ExitCode.ABORTED, None)

    try:
        exit_code = core.worker.ExitCode(returncode)
    except ValueError:
        _logger.error("bot %s ended with exit code %s", spec.bot_id, returncode)
        return Outcome(
            core.worker.ExitCode.CRASHED,
            f"bot {spec.bot_id} ended with exit code {returncode}",
        )

    return Outcome(exit_code, relay.error)
