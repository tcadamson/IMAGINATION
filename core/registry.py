"""Synchronize IMAGINATION resources with the branch updates come from."""

import collections.abc
import hashlib
import io
import json
import logging
import os
import pathlib
import re
import ssl
import tarfile
import tempfile
import typing
import urllib.parse
import urllib.request

import truststore

import core
import core.paths

_ARCHIVE_URL: typing.Final = (
    f"https://codeload.github.com/tcadamson/IMAGINATION/tar.gz/refs/heads/{core.BRANCH}"
)
_RELEASES_URL: typing.Final = "https://github.com/tcadamson/IMAGINATION/releases"

_PRE_RELEASE_PHASES: typing.Final = ("a", "b", "rc")

type _ManifestEntry = str | dict[str, str]

_logger: logging.Logger = logging.getLogger(__name__)


def _request(url: str, timeout: float = 30.0) -> bytes:
    """Download the body at `url`, which must be https."""
    if urllib.parse.urlsplit(url).scheme != "https":
        raise ValueError(f"not an https URL: {url}")

    context = truststore.SSLContext(
        ssl.PROTOCOL_TLS_CLIENT
    )  # Solves SSL: CERTIFICATE_VERIFY_FAILED
    with urllib.request.urlopen(url, timeout=timeout, context=context) as response:
        return response.read()


def _request_archive(url: str = _ARCHIVE_URL) -> dict[str, bytes]:
    """Download the update branch tarball, keyed by relative path."""
    archive = {}
    data = _request(url)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar:
            extracted = tar.extractfile(member) if member.isfile() else None

            if extracted is not None:  # Strip the top-level "IMAGINATION-<branch>/"
                archive[member.name.partition("/")[2]] = extracted.read()
    return archive


def _destination(relative_path: str) -> pathlib.Path:
    """Resolve `relative_path` against the user directory, raising if it escapes."""
    path = pathlib.PurePosixPath(relative_path)

    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe path in manifest.json: {relative_path}")

    return core.paths.USER_DIRECTORY / path


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


def _version_key(version: str) -> tuple[tuple[int, ...], tuple[int, int]]:
    """Sort key for a PEP 440 `version` of the form 2.0.0, 2.0.0b5, or 2.1rc1.

    Trailing zeros of the release are dropped (2.0 equals 2.0.0), and a pre-release
    sorts before its final release.
    """
    match = re.fullmatch(r"(\d+(?:\.\d+)*)(?:(a|b|rc)(\d+))?", version)

    if match is None:
        raise ValueError(f"not a version: {version}")

    release = tuple(int(part) for part in match[1].split("."))
    while len(release) > 1 and release[-1] == 0:
        release = release[:-1]

    if match[2] is None:
        return release, (len(_PRE_RELEASE_PHASES), 0)

    return release, (_PRE_RELEASE_PHASES.index(match[2]), int(match[3]))


def _supported(relative_path: str, entry: _ManifestEntry) -> bool:
    """Whether this program is recent enough for `entry`, warning if it is not."""
    requires = entry.get("requires") if isinstance(entry, dict) else None

    if requires is None or _version_key(core.VERSION) >= _version_key(requires):
        return True

    _logger.warning(
        "skipping %s: it needs IMAGINATION %s or later; download it from %s",
        relative_path,
        core.version_text(requires),
        _RELEASES_URL,
    )
    return False


def _remove_unlisted_bundles(manifest: collections.abc.Mapping[str, object]) -> None:
    """Delete each file in the bundle directory that no manifest entry names.

    A bundle in use by a running bot is locked, and is left for a later sync.
    """
    listed = {_destination(relative_path) for relative_path in manifest}
    for path in core.paths.BUNDLE_DIRECTORY.glob("*"):
        if not path.is_file() or path in listed:
            continue

        try:
            path.unlink()
        except PermissionError:
            _logger.debug("%s is in use; removing it at a later update", path)
            continue

        _logger.info(
            "Removed %s", path.relative_to(core.paths.USER_DIRECTORY).as_posix()
        )


def sync(archive: dict[str, bytes] | None = None) -> None:
    """Install each manifest entry whose local copy is absent or stale.

    An entry is a file of the archive, by its sha256, or an object downloaded from its
    `url` with its `sha256`, skipped if its `requires` names a newer program version.
    """
    if archive is None:
        archive = _request_archive()

    manifest: dict[str, _ManifestEntry] = json.loads(archive["resources/manifest.json"])

    for relative_path in sorted(
        manifest,
        key=lambda path: (not isinstance(manifest[path], dict), path),
    ):  # Downloads first, so that bots.json never names a bundle that failed to install
        destination = _destination(relative_path)
        entry = manifest[relative_path]
        sha256_expected = entry["sha256"] if isinstance(entry, dict) else entry

        if not _supported(relative_path, entry):
            continue

        if (
            destination.is_file()
            and hashlib.sha256(destination.read_bytes()).hexdigest() == sha256_expected
        ):
            continue

        data = (
            _request(entry["url"])
            if isinstance(entry, dict)
            else archive[relative_path]
        )
        sha256_actual = hashlib.sha256(data).hexdigest()

        if sha256_actual != sha256_expected:
            raise RuntimeError(
                f"checksum mismatch for {relative_path}: {sha256_actual}, "
                f"expected {sha256_expected}"
            )

        _atomic_write(destination, data)
        _logger.info("Installed %s", relative_path)

    _remove_unlisted_bundles(manifest)
