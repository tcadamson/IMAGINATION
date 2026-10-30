"""Synchronize IMAGINATION resources with the stable branch."""

import hashlib
import io
import json
import logging
import os
import pathlib
import ssl
import tarfile
import tempfile
import typing
import urllib.request

import truststore

import core.paths

_ARCHIVE_URL: typing.Final = (
    "https://codeload.github.com/tcadamson/IMAGINATION/tar.gz/refs/heads/stable"
)

_logger: logging.Logger = logging.getLogger(__name__)


def _request_archive(
    url: str = _ARCHIVE_URL, timeout: float = 30.0
) -> dict[str, bytes]:
    """Download the stable branch tarball, keyed by relative path."""
    archive = {}
    context = truststore.SSLContext(
        ssl.PROTOCOL_TLS_CLIENT
    )  # Solves SSL: CERTIFICATE_VERIFY_FAILED
    with urllib.request.urlopen(url, timeout=timeout, context=context) as response:
        data = response.read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar:
            extracted = tar.extractfile(member) if member.isfile() else None

            if extracted is not None:  # Strip the top-level "IMAGINATION-stable/"
                archive[member.name.partition("/")[2]] = extracted.read()
    return archive


def _destination(relative_path: str) -> pathlib.Path:
    """Resolve `relative_path` against the user directory, raising if it escapes."""
    path = pathlib.PurePosixPath(relative_path)

    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe path in manifest.json: {relative_path!r}")

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


def sync(archive: dict[str, bytes] | None = None) -> None:
    """Install each manifest entry whose local copy is absent or stale."""
    if archive is None:
        archive = _request_archive()

    manifest: dict[str, str] = json.loads(archive["resources/manifest.json"])

    for relative_path in sorted(manifest):
        destination = _destination(relative_path)
        sha256_expected = manifest[relative_path]

        if (
            destination.is_file()
            and hashlib.sha256(destination.read_bytes()).hexdigest() == sha256_expected
        ):
            continue

        data = archive[relative_path]
        sha256_actual = hashlib.sha256(data).hexdigest()

        if sha256_actual != sha256_expected:
            raise RuntimeError(
                f"Checksum mismatch for {relative_path!r}: {sha256_actual} != {sha256_expected}"
            )

        _atomic_write(destination, data)
        _logger.info("Installed: %s", relative_path)
