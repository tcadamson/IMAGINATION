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
BOT_DIRECTORY: typing.Final = USER_DIRECTORY / "resources" / "bots"
TEMPLATE_DIRECTORY: typing.Final = USER_DIRECTORY / "resources" / "templates"


def migrate() -> None:
    """Remove legacy resources not in the resources/ subdirectory.

    Installs from before the reorganization kept all resources at the root level of the
    user directory.
    """
    for legacy_directory in (
        USER_DIRECTORY / BOT_DIRECTORY.name,
        USER_DIRECTORY / TEMPLATE_DIRECTORY.name,
    ):
        shutil.rmtree(legacy_directory, ignore_errors=True)
