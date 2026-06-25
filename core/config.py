"""Load, parse, and persist the IMAGINATION console presets."""

import collections.abc
import json
import typing

import core.api

USER_DIRECTORY: typing.Final = core.api.ROOT_DIRECTORY  # TODO: Derive from platformdirs
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
