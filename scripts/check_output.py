"""Show what IMAGINATION prints, without a game client or the network.

Run with `uv run python -m scripts.check_output` for every scenario, or name some
(`... help errors`; `--list` names them). The bots' help comes from resources/bots.json,
or from `--catalog PATH` (the bot repository's dist/bots.json).

Every line comes from the real code: main's commands and startup, core's output as
core.output sets it up, and a worker in its own process for each run. Stand-ins: the bot
bundle (a stub that acts out each way a run ends, chosen by the client's process ID), the
client windows, the update branch, the error dialog, shown as a "[dialog]" line, and for
the program's own update the built program and the processes it starts, shown as
"[started ...]" lines. Files go to a temporary directory, shown as
%LOCALAPPDATA%\\IMAGINATION; the program's own directory, or the built program's
stand-in, is shown as <program>.

Run it after changing a message, and compare its output before and after a change that
should leave the output alone.
"""

import argparse
import builtins
import contextlib
import ctypes
import hashlib
import importlib
import io
import json
import os
import pathlib
import shlex
import shutil
import sys
import tempfile
import types
import typing
import urllib.error
import urllib.parse
import zipfile

_REPOSITORY: typing.Final = pathlib.Path(__file__).resolve().parents[1]

_OLD_VERSION: typing.Final = "1.0.0"  # The program updated from: any older version

# The stub bundle's way to end, by the client's process ID
_FINISHED, _STOPPED, _CRASHED, _ABORTED, _EXIT_CODE, _NATIVE = range(1, 7)
# The stub bundle's characters, by the client's process ID: none logged in on the others
_CHARACTERS: typing.Final = {_FINISHED: "Akemi", _STOPPED: "Daichi"}

_STUB_BUNDLE: typing.Final = f"""
import logging
import os
import sys

API_VERSION = 1

log = logging.getLogger("bundle")


def run(bot, options, pid):
    log.debug("(a debug line of the bot's)")

    if pid == {_FINISHED}:
        print("(a line the bot prints)")
        log.info("(a line the bot logs)")
        return 0
    if pid == {_STOPPED}:
        log.error("(the bot's reason to stop)")
        return 1
    if pid == {_CRASHED}:
        raise RuntimeError("(the bot's bug)")
    if pid == {_ABORTED}:
        raise KeyboardInterrupt
    if pid == {_EXIT_CODE}:
        sys.__stdout__.flush()
        os._exit(7)
    if pid == {_NATIVE}:
        sys.__stdout__.write("Windows fatal exception: access violation\\n")
        sys.__stdout__.flush()
        os._exit(-0x3FFFFFFB)  # 0xC0000005 as a C int, an access violation


def character_name(pid):
    return {_CHARACTERS!r}.get(pid)
"""

_OTHER_API_BUNDLE: typing.Final = (
    "API_VERSION = 99\n\ndef run(bot, options, pid):\n    return 0\n"
)
_UNNAMING_BUNDLE: typing.Final = _STUB_BUNDLE.split("\n\ndef character_name")[0]
_BROKEN_BUNDLE: typing.Final = 'raise ImportError("(a bundle that fails to load)")\n'

_main: types.ModuleType
_program_directory: pathlib.Path
_catalog: dict[str, typing.Any]
_scenarios: dict[str, typing.Callable[[], None]] = {}


class _Shown(io.TextIOBase):
    """Stand-in for sys.stdout writing to it with the temporary paths named as a user's."""

    def __init__(self, stream: typing.TextIO, names: dict[str, str]):
        super().__init__()
        self._stream = stream
        self._names = names

    def writable(self) -> bool:
        return True

    def write(self, s: str, /) -> int:
        for path, name in self._names.items():
            s = s.replace(path, name)
        self._stream.write(s)
        return len(s)

    def flush(self) -> None:
        self._stream.flush()

    def reconfigure(self, **_) -> None:
        """core.output.setup reconfigures the real stream, done already."""


def _scenario(function: typing.Callable[[], None]) -> typing.Callable[[], None]:
    _scenarios[function.__name__.lstrip("_")] = function
    return function


def _user_directory() -> pathlib.Path:
    return _main.core.paths.USER_DIRECTORY


def _write(relative_path: str, data: str | bytes) -> pathlib.Path:
    path = _user_directory() / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data, str):
        data = data.encode("utf-8")

    path.write_bytes(data)
    return path


def _install(bundle: str = _STUB_BUNDLE, catalog: object = None) -> str:
    """Install `bundle` by its sha256 and a catalog naming it; return the sha256."""
    sha256 = hashlib.sha256(bundle.encode("utf-8")).hexdigest()
    _write(f"resources/bundles/{sha256}.py", bundle)
    _write(
        "resources/bots.json",
        json.dumps(catalog if catalog is not None else _catalog | {"bundle": sha256}),
    )
    return sha256


def _fresh() -> None:
    """Empty the user directory but the log, then install the stub bundle and catalog."""
    shutil.rmtree(_user_directory() / "resources", ignore_errors=True)
    _write("resources/presets.json", "{}\n")
    _write("resources/presets.default.json", "{}\n")
    _install()
    _register()


def _register() -> None:
    """Register the installed bots and presets quietly, as at a start already shown."""
    with contextlib.redirect_stdout(io.StringIO()), _quiet():
        _main.update()
        _main._cli_presets_load()


@contextlib.contextmanager
def _quiet():
    console = _main.core.output._console
    level = console.level
    console.setLevel(100)

    try:
        yield
    finally:
        console.setLevel(level)


def _type(line: str) -> None:
    """Type `line` at the console's prompt."""
    print(f">>> {line}")
    _main._guard(lambda: _main._dispatch(shlex.split(line)))
    print()


def _start() -> None:
    """Start IMAGINATION as main.py does, after its output's setup."""
    _main._guard(_main.update)
    _main._guard(_main._cli_presets_load)
    print(_main._BANNER)
    _main._guard(lambda: _main._dispatch(("--help",)))
    print()


@contextlib.contextmanager
def _client(*pids: int):
    """A client window found for each process of `pids`, or none without any."""
    api = _main.core.api
    window_title, locate_all = api._window_title, api.Client.locate_all

    if not pids:
        api._window_title = lambda handle: ""
    else:
        api.Client.locate_all = classmethod(
            lambda cls, identifier=api.Client.CLIENT_IDENTIFIER: tuple(
                types.SimpleNamespace(pid=pid) for pid in pids
            )
        )

    try:
        yield
    finally:
        api._window_title, api.Client.locate_all = window_title, locate_all


@contextlib.contextmanager
def _updates(
    manifest: dict[str, object],
    downloads: dict[str, bytes] | None = None,
    release: dict[str, str] | None = None,
):
    """Updates from a branch holding `manifest`, the user's resources, and `release` as
    its release.json, as on a start without IMAGINATION_USER_DIRECTORY; `downloads` by
    URL, or None for no network."""
    registry, paths = _main.core.registry, _main.core.paths
    request_archive, request = registry.request_archive, registry.request
    override = paths.USER_DIRECTORY_OVERRIDE_PASSED

    def archive(url: str = "") -> dict[str, bytes]:
        if downloads is None:
            raise urllib.error.URLError("[Errno 11001] getaddrinfo failed")

        return {
            "resources/manifest.json": json.dumps(manifest).encode("utf-8"),
            "resources/bots.json": (_user_directory() / "stub-bots.json").read_bytes(),
            "resources/presets.default.json": b"{}\n",
        } | (
            {"resources/release.json": json.dumps(release).encode("utf-8")}
            if release is not None
            else {}
        )

    def download(url: str, timeout: float = 30.0) -> bytes:
        if urllib.parse.urlsplit(url).scheme != "https":
            return request(url)  # The real refusal

        return typing.cast(dict[str, bytes], downloads)[url]

    registry.request_archive, registry.request = archive, download
    paths.USER_DIRECTORY_OVERRIDE_PASSED = False

    try:
        yield
    finally:
        registry.request_archive, registry.request = request_archive, request
        paths.USER_DIRECTORY_OVERRIDE_PASSED = override


def _branch(url: str = "https://github.com/tcadamson/IMAGINATION/releases/x.pyd"):
    """A manifest and downloads as the update branch would have them, with the stub
    bundle at `url`; the catalog naming it, for the archive."""
    bundle = _STUB_BUNDLE.encode("utf-8")
    sha256 = hashlib.sha256(bundle).hexdigest()
    catalog = json.dumps(_catalog | {"bundle": sha256}).encode("utf-8")
    _write("stub-bots.json", catalog)
    manifest = {
        f"resources/bundles/{sha256}.py": {"url": url, "sha256": sha256},
        "resources/bots.json": hashlib.sha256(catalog).hexdigest(),
        "resources/presets.default.json": hashlib.sha256(b"{}\n").hexdigest(),
    }
    return manifest, {url: bundle}


class _Exited(BaseException):
    """The program's exit, os._exit's stand-in: past _guard and `except Exception`."""


@contextlib.contextmanager
def _built(
    directory: pathlib.Path, console_processes: int = 1, version: str | None = None
):
    """The built program, in `directory`, in a console with `console_processes`
    processes, of `version` if not this one's; the processes it starts shown as
    "[started ...]" lines, its exit as "[exited]"."""
    api, program = _main.core.api, _main.core.program
    kernel32 = program._kernel32
    saved = (
        _main.core.VERSION,
        getattr(sys, "frozen", False),
        sys.executable,
        api.ROOT_DIRECTORY,
        program.subprocess.Popen,
        os._exit,
        _main.logging.shutdown,
        kernel32.GetConsoleProcessList,
    )

    def popen(args: list[str], **_) -> None:
        shown = [*args]

        if program.INSTALL_FLAG in shown:  # This process's ID, different on each run
            shown[shown.index(program.INSTALL_FLAG) + 2] = "<pid>"

        print(f"[started {' '.join(shown)}]")

    def exit_(code: int) -> typing.NoReturn:
        print("[exited]")
        raise _Exited

    _main.core.VERSION = version or _main.core.VERSION
    sys.frozen = True
    sys.executable = str(directory / "IMAGINATION.exe")
    api.ROOT_DIRECTORY = directory
    program.subprocess.Popen = popen
    os._exit = exit_
    _main.logging.shutdown = lambda: None
    kernel32.GetConsoleProcessList = lambda processes, count: console_processes

    try:
        yield
    finally:
        (
            _main.core.VERSION,
            sys.frozen,
            sys.executable,
            api.ROOT_DIRECTORY,
            program.subprocess.Popen,
            os._exit,
            _main.logging.shutdown,
            kernel32.GetConsoleProcessList,
        ) = saved


def _launch(*args: str) -> None:
    """Launch IMAGINATION's update as main.py does, with `args` on its command line."""
    argv = sys.argv
    sys.argv = ["IMAGINATION.exe", *args]

    try:
        _main._guard(lambda: _main._update(on_launch=True))
    except _Exited:
        pass
    finally:
        sys.argv = argv


def _old_program() -> None:
    """Put the old program's files in the stand-in program directory."""
    shutil.rmtree(_program_directory, ignore_errors=True)
    for relative_path in ("IMAGINATION.exe", "LICENSE", "lib/library.zip"):
        path = _program_directory / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("(old)")


def _new_program() -> tuple[str, bytes, dict[str, str]]:
    """The new program's zip as a release holds it: its URL, its bytes, and the
    release.json naming it. The new program is this one, the old one `_OLD_VERSION`."""
    url = (
        "https://github.com/tcadamson/IMAGINATION/releases/download/"
        f"{_main.core.version_text()}/IMAGINATION.zip"
    )
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as zip_file:
        for relative_path in ("IMAGINATION.exe", "LICENSE", "lib/library.zip"):
            zip_file.writestr(f"IMAGINATION/{relative_path}", "(new)")
    release = {
        "version": _main.core.VERSION,
        "url": url,
        "sha256": hashlib.sha256(data.getvalue()).hexdigest(),
    }
    return url, data.getvalue(), release


def _bot_without_options() -> str:
    """A bot of the catalog that runs with no option given."""
    return next(
        bot_id
        for bot_id, bot in _catalog["bots"].items()
        if all(
            "default" in option or option["type"] == "bool"
            for option in bot["options"].values()
        )
    )


@_scenario
def _start_beta5() -> None:
    """The first start after beta.5: its bots and templates removed, everything installed."""
    shutil.rmtree(_user_directory() / "resources", ignore_errors=True)
    for relative_path in (
        "resources/bots/tmg.py",
        "resources/templates/yes.png",
        "resources/presets.default.json",
        "debug.log",
    ):
        _write(relative_path, "")
    _write("resources/presets.json", '{"r4": "run rebirth --ranks 4"}\n')
    _write("resources/bundles/0123abcd.cp314-win_amd64.pyd", "")  # An older bundle

    manifest, downloads = _branch()
    with _updates(manifest, downloads):
        _start()


@_scenario
def _start_up_to_date() -> None:
    """A start with nothing to install."""
    _fresh()
    manifest, downloads = _branch()
    with _updates(manifest, downloads):
        _main._guard(_main.update)
        _main._guard(_main._cli_presets_load)
    print()


@_scenario
def _start_offline() -> None:
    """A start with no network: the bots installed before stay."""
    _fresh()
    manifest, _ = _branch()
    with _updates(manifest, None):
        _main._guard(_main.update)
    print()


@_scenario
def _update_refusals() -> None:
    """What a sync refuses: a program too old, a checksum, a path, a URL."""
    for name, change in (
        ("requires", lambda entry: entry | {"requires": "2.1.0"}),
        ("checksum", lambda entry: entry | {"sha256": "0" * 64}),
        ("http", lambda entry: entry | {"url": entry["url"].replace("https", "http")}),
    ):
        _fresh()
        (_user_directory() / "resources" / "bots.json").unlink()  # As at a first start
        shutil.rmtree(_user_directory() / "resources" / "bundles")
        manifest, downloads = _branch()
        bundle_path = next(path for path in manifest if path.endswith(".py"))
        manifest[bundle_path] = change(manifest[bundle_path])
        print(f"({name})")
        with _updates(manifest, downloads):
            _main._guard(_main.update)
        print()

    _fresh()
    manifest, downloads = _branch()
    manifest["../outside.txt"] = "0" * 64
    print("(unsafe path)")
    with _updates(manifest, downloads):
        _main._guard(_main.update)
    print()


@_scenario
def _update_program() -> None:
    """The program's own update, from launch to the new program's: from Explorer, from a
    shell, typed (yes, no), a checksum, a folder that can't be written to, a file of the
    old program in use."""
    url, data, release = _new_program()
    staged = _user_directory() / "update" / "IMAGINATION"

    for name, change, console_processes, answer, locked in (
        ("from Explorer", {}, 1, None, False),
        ("from a shell", {}, 2, None, False),
        ("typed, yes", {}, 1, "y", False),
        ("typed, no", {}, 1, "", False),
        ("checksum", {"sha256": "0" * 64}, 1, None, False),
        ("folder can't be written to", {}, 1, None, False),
        ("a file of the old program in use", {}, 1, None, True),
    ):
        _fresh()
        _old_program()
        (
            _user_directory() / "resources" / "bots.json"
        ).unlink()  # For an update to show
        manifest, downloads = _branch()
        writable, typed = _main.core.program._writable, builtins.input

        if name == "folder can't be written to":
            _main.core.program._writable = lambda directory: False

        def input_(prompt: str = "") -> str:
            print(f"{prompt}{answer}")
            return typing.cast(str, answer)

        print(f"({name})")
        try:
            with _updates(manifest, downloads | {url: data}, release | change):
                with _built(_program_directory, console_processes, _OLD_VERSION):
                    if answer is None:
                        _launch()
                    else:
                        builtins.input = input_
                        try:
                            _type("update")
                        except _Exited:
                            pass

                if console_processes == 1 and staged.is_dir():  # The installer
                    with (
                        _built(staged),
                        open(_program_directory / "LICENSE")
                        if locked
                        else contextlib.nullcontext(),
                    ):
                        _main._guard(
                            lambda: _main._install_update(
                                [str(_program_directory), "0", _OLD_VERSION]
                            )
                        )

                    with _built(_program_directory):
                        _launch(_main.core.program.UPDATED_FLAG)
        finally:
            _main.core.program._writable, builtins.input = writable, typed

        print()


@_scenario
def _help() -> None:
    """Every help screen."""
    _fresh()
    for line in ("--help", "run", "run --help", "presets", "update --help"):
        _type(line)
    for bot_id in _catalog["bots"]:
        _type(f"run {bot_id} --help")
    _type("presets list --help")
    _type("presets load --help")


@_scenario
def _errors() -> None:
    """Typer's refusals, as the console words them."""
    _fresh()
    bots = list(_catalog["bots"])
    for line in (
        "tmg",
        "run nobot",
        f"run {bots[0]} --nosuch",
        "run rebirth --chek",
        "run rebirth",
        "run rebirth --ranks 1 --end-path 13",
        "run rebirth --ranks 1 --end-path x",
        "run rebirth --ranks 1 extra",
        "run demon_force --lbp",
        "run demon_force --fcc 60",
        "run rebirth --verbose --ranks 1",
        'run rebirth --ranks "1',
    ):
        _type(line)


@_scenario
def _runs() -> None:
    """Each way a run ends, framed as IMAGINATION frames it."""
    _fresh()
    bot = _bot_without_options()
    for pids, what in (
        ((), "no client"),
        ((_FINISHED,), "finished"),
        ((_STOPPED,), "stopped"),
        ((_CRASHED,), "crashed"),
        ((_ABORTED,), "Ctrl+C"),
        ((_EXIT_CODE,), "an exit code of its own"),
        ((_NATIVE,), "a native crash"),
    ):
        print(f"({what})")
        with _client(*pids):
            _type(f"run {bot}")

    print("(finished, with --verbose)")
    with _client(_FINISHED):
        _type(f"run --verbose {bot}")


@_scenario
def _clients() -> None:
    """Several clients open: chosen by their characters' names, one with none logged
    in, an answer out of range, Ctrl+C, and a bundle that can't name them."""
    bot = _bot_without_options()
    typed = builtins.input

    for name, pids, answers, bundle in (
        ("two", (_FINISHED, _STOPPED), ["1"], None),
        ("one with no character", (_STOPPED, _FINISHED, _CRASHED), ["4", "2"], None),
        ("Ctrl+C", (_FINISHED, _STOPPED), [None], None),
        (
            "a bundle that can't name them",
            (_FINISHED, _STOPPED),
            ["1"],
            _UNNAMING_BUNDLE,
        ),
    ):
        _fresh()
        if bundle is not None:
            _install(bundle)
            _register()

        def input_(prompt: str = "") -> str:
            answer = answers.pop(0)
            if answer is None:  # Ctrl+C
                print(prompt, end="")
                raise KeyboardInterrupt
            print(f"{prompt}{answer}")
            return answer

        print(f"({name})")
        builtins.input = input_
        try:
            with _client(*pids):
                _type(f"run {bot}")
        finally:
            builtins.input = typed


@_scenario
def _bundles() -> None:
    """A bundle that can't run, and a catalog that can't be read."""
    bot = _bot_without_options()
    for name, bundle in (
        ("another API", _OTHER_API_BUNDLE),
        ("broken", _BROKEN_BUNDLE),
    ):
        _fresh()
        _install(bundle)
        _register()
        print(f"({name})")
        with _client(_FINISHED):
            _type(f"run {bot}")

    for name, catalog in (
        ("not installed", _catalog | {"bundle": "0" * 64}),
        ("invalid", {"bundle": "0" * 64}),
    ):
        _fresh()
        _install(catalog=catalog)
        print(f"({name})")
        _type("update")

    _fresh()
    _write("resources/bots.json", "{")
    print("(malformed)")
    _type("update")


@_scenario
def _presets() -> None:
    """Presets: none, some, one named as a command, a malformed file."""
    _fresh()
    _type("presets list")

    _write(
        "resources/presets.json",
        json.dumps({"F5": "run tmg --cycles 5", "update": "run tmg"}),
    )
    _type("presets load")
    _type("presets list")
    with _client():
        _type("f5")

    _write("resources/presets.json", "{")
    _type("presets load")
    _write("resources/presets.json", '{"r4": 4}')
    _type("presets list")


def main() -> None:
    global _main, _catalog, _program_directory

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("names", nargs="*")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--catalog", type=pathlib.Path)
    arguments = parser.parse_args()

    if arguments.list:
        for name, function in _scenarios.items():
            print(f"{name}: {(function.__doc__ or '').splitlines()[0]}")
        return

    unknown = [name for name in arguments.names if name not in _scenarios]
    if unknown:
        sys.exit(f"no scenario {', '.join(unknown)}; --list names them")

    catalog_path = arguments.catalog or _REPOSITORY / "resources" / "bots.json"
    if not catalog_path.is_file():
        sys.exit(f"no catalog at {catalog_path}; give one with --catalog")
    _catalog = json.loads(catalog_path.read_text(encoding="utf-8"))

    user_directory = pathlib.Path(tempfile.mkdtemp(prefix="imagination-output-"))
    _program_directory = pathlib.Path(tempfile.mkdtemp(prefix="imagination-program-"))
    os.environ["IMAGINATION_USER_DIRECTORY"] = str(user_directory)
    os.environ["COLUMNS"] = "120"  # Help as wide as in a console, not a pipe's 80
    _main = importlib.import_module("main")  # Its paths read the variable as they load

    typing.cast(io.TextIOWrapper, sys.stdout).reconfigure(encoding="utf-8")
    names = {
        sys.executable: "<python>",
        str(_REPOSITORY): "<program>",
        str(_program_directory): "<program>",
        str(pathlib.Path(sys.base_prefix)): "<python home>",
        str(user_directory): "%LOCALAPPDATA%\\IMAGINATION",
    }
    names |= {  # As an exception's message quotes a path
        path.replace("\\", "\\\\"): name.replace("\\", "\\\\")
        for path, name in names.items()
    }
    sys.stdout = _Shown(sys.stdout, names)
    ctypes.windll.user32.MessageBoxW = lambda handle, text, title, flags: print(
        f"[dialog] {text.rstrip()}"
    )

    _main.core.output.setup()
    _main._cli.add_typer(_main._cli_presets, name="presets")
    _main._cli.add_typer(_main._cli_run, name="run")

    for name in arguments.names or _scenarios:
        print(f"======== {name}")
        _scenarios[name]()

    sys.stdout.flush()


if __name__ == "__main__":
    main()
