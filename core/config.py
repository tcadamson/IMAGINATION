"""Load, parse, and persist the IMAGINATION console presets."""

import json
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

PRESETS_DEFAULT_PATH: typing.Final = (
    USER_DIRECTORY / "resources" / "presets.default.json"
)
PRESETS_PATH: typing.Final = USER_DIRECTORY / "resources" / "presets.json"


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


def _read_presets_json(path: pathlib.Path) -> dict[str, str]:
    """Read and parse a JSON file with presets."""
    try:
        with path.open(encoding="utf-8") as fp:
            data = json.load(fp)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exception:
        raise RuntimeError(f"Malformed JSON at {path}") from exception

    return {
        preset_id: preset
        for preset_id, preset in data.items()
        if isinstance(preset, str)
    }


def load_presets() -> dict[str, str]:
    """Load presets, overriding defaults with user-authored presets."""
    return {
        **_read_presets_json(PRESETS_DEFAULT_PATH),
        **_read_presets_json(PRESETS_PATH),
    }


def ensure_presets() -> None:
    """Create an empty user presets file next to the defaults if it is absent."""
    if not PRESETS_PATH.exists():
        PRESETS_PATH.parent.mkdir(parents=True, exist_ok=True)
        PRESETS_PATH.write_text("{}\n", encoding="utf-8")
