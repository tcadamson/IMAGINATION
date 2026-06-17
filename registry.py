"""Discover, synchronize, and register IMAGINATION bots."""

import collections.abc
import hashlib
import importlib.util
import json
import logging
import os
import pathlib
import sys
import tempfile
import typing
import urllib.request

import api
import config

type Manifest = dict[str, str]

_ROOT_URL: typing.Final = (
    "https://raw.githubusercontent.com/tcadamson/IMAGINATION/stable/"
)

_logger: logging.Logger = logging.getLogger(__name__)


def _request_manifest(
    url: str = _ROOT_URL + "manifest.json", timeout: float = 10.0
) -> Manifest:
    """Fetch and parse the bot manifest from `url`."""
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def _download(url: str, sha256_expected: str, timeout: float = 30.0) -> bytes:
    """Download bytes from `url`, raising unless they match `sha256_expected`."""
    with urllib.request.urlopen(url, timeout=timeout) as response:
        data = response.read()
    sha256_actual = hashlib.sha256(data).hexdigest()

    if sha256_actual != sha256_expected:
        raise RuntimeError(
            f"Checksum mismatch for {url}: {sha256_actual} != {sha256_expected}"
        )

    return data


def _destination(relative_path: str) -> pathlib.Path:
    """Resolve `relative_path` against the user directory, raising if it escapes."""
    path = pathlib.PurePosixPath(relative_path)

    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe path in manifest.json: {relative_path!r}")

    return config.USER_DIRECTORY / path


def _atomic_write(path: pathlib.Path, data: bytes) -> None:
    """Write `data` to `path` via a temporary file, replacing on success.

    Writing to a sibling temp file and replacing keeps `path` from being left partially
    written if the process is interrupted mid-write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(dir=path.parent)

    try:
        with os.fdopen(fd, "wb") as fw:
            fw.write(data)
            fw.flush()
        os.replace(filename, path)
    except Exception:
        os.unlink(filename)
        raise


def register_bot_directory(
    bot_directory: pathlib.Path,
) -> collections.abc.Mapping[str, api.BotSpec]:
    """Dynamically import every bot module in `bot_directory`, keyed by file stem.

    Automatic namespace prefix prevents collisions on sys.modules.
    """
    bots = {}
    for path in sorted(bot_directory.glob("*.py")):
        bot_id = path.stem
        bot_module_id = f"_bot_{bot_id}"

        try:
            spec = importlib.util.spec_from_file_location(bot_module_id, path)

            if spec is None or spec.loader is None:
                raise ImportError("No loadable module spec")

            module = importlib.util.module_from_spec(spec)
            sys.modules[bot_module_id] = (
                module  # Executing before registering here would break dataclasses, among other things
            )
            spec.loader.exec_module(module)
            bots[bot_id] = module.SPEC
        except Exception:
            sys.modules.pop(bot_module_id, None)
            _logger.exception("Failed to load bot: %s", path)
            continue

        _logger.info("Loaded bot: %s", bot_id)
    return bots


def sync(manifest: Manifest | None = None) -> None:
    """Download each manifest entry whose local copy is absent or stale.

    Fetch the `manifest` from the repo if omitted. Files already matching their
    expected checksum are left untouched.
    """
    if manifest is None:
        manifest = _request_manifest()

    for relative_path in sorted(
        manifest,
        key=lambda path: path.startswith(
            f"{config.BOT_DIRECTORY.name}/"
        ),  # Sort templates before the bots depending on them
    ):
        destination = _destination(relative_path)
        sha256_expected = manifest[relative_path]

        if (
            destination.is_file()
            and hashlib.sha256(destination.read_bytes()).hexdigest() == sha256_expected
        ):
            continue

        _atomic_write(
            destination, _download(_ROOT_URL + relative_path, sha256_expected)
        )
        _logger.info("Installed: %s", relative_path)
