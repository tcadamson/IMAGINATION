"""Load, parse, and persist the IMAGINATION console presets."""

import collections.abc
import json
import os
import pathlib
import typing

import platformdirs

_USER_DIRECTORY_OVERRIDE: typing.Final = os.environ.get("IMAGINATION_USER_DIRECTORY")

USER_DIRECTORY_OVERRIDE_PASSED: typing.Final = (
    _USER_DIRECTORY_OVERRIDE is not None
)  # Use project root during development

USER_DIRECTORY: typing.Final = (
    pathlib.Path(_USER_DIRECTORY_OVERRIDE).resolve()
    if _USER_DIRECTORY_OVERRIDE
    else platformdirs.user_data_path("IMAGINATION", appauthor=False, ensure_exists=True)
)
BOT_DIRECTORY: typing.Final = USER_DIRECTORY / "bots"
TEMPLATE_DIRECTORY: typing.Final = USER_DIRECTORY / "templates"

_PRESETS_PATH: typing.Final = USER_DIRECTORY / "presets.json"


def load_presets() -> dict[str, str]:
    """Load and parse presets.json into a preset dict."""
    try:
        with _PRESETS_PATH.open(encoding="utf-8") as fp:
            data = json.load(fp)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exception:
        raise RuntimeError(f"Malformed JSON at {_PRESETS_PATH}") from exception

    return {
        preset_id: preset
        for preset_id, preset in data.items()
        if isinstance(preset, str)
    }


def save_presets(presets: collections.abc.Mapping[str, str]) -> None:
    """Serialize `presets` to presets.json on disk."""
    with _PRESETS_PATH.open("w", encoding="utf-8", newline="\n") as fp:
        json.dump(dict(presets), fp, indent=4)
        fp.write("\n")
