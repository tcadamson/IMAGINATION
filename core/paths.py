"""User data directory layout for IMAGINATION resources."""

import hashlib
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
    """Handle cleanup of resources following project reorganization.

    Remove legacy resources not in the resources/ subdirectory, as installs prior to
    reorganization kept all resources at the root level of the user directory.

    Remove legacy specs.json, as these became user-owned to shadow specs.default.json
    (mirroring preset behavior). Any hanging specs.json files would overwrite future
    changes to specs.default.json.
    """
    for legacy_directory in (
        USER_DIRECTORY / BOT_DIRECTORY.name,
        USER_DIRECTORY / TEMPLATE_DIRECTORY.name,
    ):
        shutil.rmtree(legacy_directory, ignore_errors=True)
    sha256_legacy_specs = frozenset(
        {
            "f4b2c1d45c818157d6d53f9eb9c63fb6f0d99f70ff01d1505c35a0d57678bdcf",
            "78897788bf330d584573320f9322b9038ae04a88ba5099f82e4936d68104682c",
            "0d7ba3e1c9e209a1de59b59b4c7a288438db053bd4b9daa44433e7217cb8d7a4",
            "a803afe02efe05e013bc7ccaed11e7efd24a84ad9021afd9d5cbd1c2970de452",
        }
    )
    for legacy_specs in TEMPLATE_DIRECTORY.rglob("specs.json"):
        if hashlib.sha256(legacy_specs.read_bytes()).hexdigest() in sha256_legacy_specs:
            legacy_specs.unlink()
