"""User data directory layout for IMAGINATION resources."""

import os
import pathlib
import shutil
import typing

import platformdirs

_USER_DIRECTORY_OVERRIDE: typing.Final = os.environ.get("IMAGINATION_USER_DIRECTORY")

USER_DIRECTORY_OVERRIDE_PASSED: typing.Final = (
    _USER_DIRECTORY_OVERRIDE is not None
)  # `set "IMAGINATION_USER_DIRECTORY=." && .venv\Scripts\python.exe main.py`

USER_DIRECTORY: typing.Final = (
    pathlib.Path(_USER_DIRECTORY_OVERRIDE).resolve()
    if _USER_DIRECTORY_OVERRIDE
    else platformdirs.user_data_path("IMAGINATION", appauthor=False, ensure_exists=True)
)


def migrate() -> None:
    """Handle cleanup of resources following project reorganization.

    Remove legacy resources not in the resources/ subdirectory, as installs prior to
    reorganization kept all resources at the root level of the user directory.

    Remove installed bots and their templates, as template matching was removed.
    """
    for legacy_directory in (
        USER_DIRECTORY / "bots",
        USER_DIRECTORY / "templates",
        USER_DIRECTORY / "resources" / "bots",
        USER_DIRECTORY / "resources" / "templates",
    ):
        shutil.rmtree(legacy_directory, ignore_errors=True)
