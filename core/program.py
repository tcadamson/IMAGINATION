"""Update IMAGINATION's own program from the branch updates come from.

The branch's resources/release.json names the latest program: its `version`, and the
`url` and `sha256` of a release's zip of the built program. It is not a manifest entry,
so programs older than this updater never see it.

A newer program is unpacked into the user directory and started from there with
`INSTALL_FLAG`, the program's directory, this process's ID, and its version; this
process exits. The
new program waits for it to close, moves the old program's files into a `previous`
folder beside them, and copies its own in; if the copy fails, the old files are put
back. Either way it then starts the program installed there with `UPDATED_FLAG`, which
continues the update: no program update, and no second "Checking for updates..." line.

The command line after `INSTALL_FLAG` is read by the new program, and `UPDATED_FLAG` by
the old one after a failure: keep both as they are.
"""

import collections.abc
import ctypes
import ctypes.wintypes
import hashlib
import io
import json
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import typing
import zipfile

import core
import core.api
import core.paths
import core.registry

INSTALL_FLAG: typing.Final = "--install"
UPDATED_FLAG: typing.Final = "--updated"

_RELEASE_PATH: typing.Final = "resources/release.json"
_PREVIOUS_NAME: typing.Final = "previous"

_EXIT_TIMEOUT_MS: typing.Final = 60_000
_SYNCHRONIZE: typing.Final = 0x00100000
_WAIT_OBJECT_0: typing.Final = 0

_kernel32: typing.Final = ctypes.WinDLL("kernel32", use_last_error=True)
_kernel32.OpenProcess.argtypes = [
    ctypes.wintypes.DWORD,
    ctypes.wintypes.BOOL,
    ctypes.wintypes.DWORD,
]
_kernel32.OpenProcess.restype = ctypes.wintypes.HANDLE
_kernel32.WaitForSingleObject.argtypes = [ctypes.wintypes.HANDLE, ctypes.wintypes.DWORD]
_kernel32.WaitForSingleObject.restype = ctypes.wintypes.DWORD
_kernel32.CloseHandle.argtypes = [ctypes.wintypes.HANDLE]
_kernel32.GetConsoleProcessList.argtypes = [
    ctypes.POINTER(ctypes.wintypes.DWORD),
    ctypes.wintypes.DWORD,
]
_kernel32.GetConsoleProcessList.restype = ctypes.wintypes.DWORD

_logger: logging.Logger = logging.getLogger(__name__)


def _frozen() -> bool:
    """Whether this is the built program, which can update itself, and not the source."""
    return getattr(sys, "frozen", False)


def _executable_name() -> str:
    return pathlib.Path(sys.executable).name


def newer_release(
    archive: collections.abc.Mapping[str, bytes],
) -> dict[str, str] | None:
    """The release that `archive`'s release.json names, if it is newer than this
    program, and this program can update itself; None otherwise."""
    data = archive.get(_RELEASE_PATH)

    if data is None or not _frozen():
        return None

    release = json.loads(data)

    if core.registry.version_key(release["version"]) <= core.registry.version_key(
        core.VERSION
    ):
        return None

    return release


def update(release: collections.abc.Mapping[str, str]) -> bool:
    """Start the program of `release`, from newer_release, to install itself over this
    one.

    True when it was started, and this program should exit at once. Skipped, with a
    warning, if this program's folder can't be written to.
    """
    version = release["version"]

    if not _writable(core.api.ROOT_DIRECTORY):
        _logger.warning(
            "skipping IMAGINATION %s: %s can't be written to; download it from %s",
            core.version_text(version),
            core.api.ROOT_DIRECTORY,
            core.registry.RELEASES_URL,
        )
        return False

    staged = _stage(release["url"], release["sha256"])
    _start_installer(staged)
    return True


def _writable(directory: pathlib.Path) -> bool:
    """Whether files can be made in `directory`."""
    try:
        with tempfile.TemporaryFile(dir=directory):
            pass
    except OSError:
        return False

    return True


def _stage(url: str, sha256: str) -> pathlib.Path:
    """Download the zipped program at `url`, check it, and unpack it into the update
    directory; return the unpacked program's directory."""
    data = core.registry.request(url)
    sha256_actual = hashlib.sha256(data).hexdigest()

    if sha256_actual != sha256:
        raise RuntimeError(
            f"checksum mismatch for {url}: {sha256_actual}, expected {sha256}"
        )

    if core.paths.UPDATE_DIRECTORY.exists():
        shutil.rmtree(core.paths.UPDATE_DIRECTORY)

    with zipfile.ZipFile(io.BytesIO(data)) as zip_file:
        zip_file.extractall(core.paths.UPDATE_DIRECTORY)

    programs = list(core.paths.UPDATE_DIRECTORY.glob(f"*/{_executable_name()}"))

    if len(programs) != 1:
        raise RuntimeError(f"no single {_executable_name()} in {url}")

    return programs[0].parent


def _start_installer(staged: pathlib.Path) -> None:
    """Start the program in `staged` to install itself over this one.

    It shares this console when nothing else uses it, as when IMAGINATION was opened
    from Explorer, and gets a console of its own otherwise: a shell IMAGINATION was
    started from would read the same input.
    """
    processes = (ctypes.wintypes.DWORD * 1)()
    console_to_itself = _kernel32.GetConsoleProcessList(processes, 1) == 1
    subprocess.Popen(
        [
            str(staged / _executable_name()),
            INSTALL_FLAG,
            str(core.api.ROOT_DIRECTORY),
            str(os.getpid()),
            core.VERSION,
        ],
        cwd=staged,
        creationflags=0 if console_to_itself else subprocess.CREATE_NEW_CONSOLE,
    )

    if not console_to_itself:
        _logger.info("IMAGINATION continues in a new window")


def install(argv: collections.abc.Sequence[str]) -> None:
    """Install this program over the one in directory argv[0], of version argv[2], once
    process argv[1] has closed; on a failure, the old program is left as it was."""
    directory, pid, old_version = pathlib.Path(argv[0]), int(argv[1]), argv[2]
    _wait_for_exit(pid)
    _replace(directory)
    _logger.info(
        "Updated IMAGINATION from %s to %s",
        core.version_text(old_version),
        core.version_text(),
    )


def start(argv: collections.abc.Sequence[str]) -> None:
    """Start the program in directory argv[0] to continue the update."""
    directory = pathlib.Path(argv[0])
    subprocess.Popen([str(directory / _executable_name()), UPDATED_FLAG], cwd=directory)


def _wait_for_exit(pid: int) -> None:
    """Wait for process `pid` to close."""
    handle = _kernel32.OpenProcess(_SYNCHRONIZE, False, pid)

    if not handle:  # Closed already
        return

    try:
        if _kernel32.WaitForSingleObject(handle, _EXIT_TIMEOUT_MS) != _WAIT_OBJECT_0:
            raise RuntimeError(f"IMAGINATION (process {pid}) didn't close")
    finally:
        _kernel32.CloseHandle(handle)


def _replace(directory: pathlib.Path) -> None:
    """Replace the program in `directory` with this one, the old files put back on a
    failure.

    The old files are moved aside, not overwritten, so that a failure part way leaves
    the old program whole. Files the new program doesn't have are left where they are.
    """
    names = sorted(path.name for path in core.api.ROOT_DIRECTORY.iterdir())
    previous = directory / _PREVIOUS_NAME

    if previous.exists():
        shutil.rmtree(previous)

    previous.mkdir()
    moved: list[str] = []
    copying = False

    try:
        for name in names:
            if (directory / name).exists():
                os.replace(directory / name, previous / name)
                moved.append(name)

        copying = True
        for name in names:
            source = core.api.ROOT_DIRECTORY / name

            if source.is_dir():
                shutil.copytree(source, directory / name)
            else:
                shutil.copy2(source, directory / name)
    except Exception:
        try:
            if copying:
                for name in names:
                    _remove(directory / name)

            for name in moved:
                os.replace(previous / name, directory / name)
        except Exception as exception:
            raise RuntimeError(
                f"can't put the old program back: it is in {previous}"
            ) from exception

        raise

    shutil.rmtree(previous, ignore_errors=True)

    if previous.exists():  # Held by another IMAGINATION still open
        _logger.debug("%s is in use; removing it at a later update", previous)


def _remove(path: pathlib.Path) -> None:
    """Delete the file or directory at `path`, if there is one."""
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def clean_up() -> None:
    """Delete what an update leaves behind: the unpacked program, and the old program's
    files if they were still in use."""
    shutil.rmtree(core.paths.UPDATE_DIRECTORY, ignore_errors=True)

    if _frozen():
        shutil.rmtree(core.api.ROOT_DIRECTORY / _PREVIOUS_NAME, ignore_errors=True)
