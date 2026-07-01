"""Load, parse, and persist the IMAGINATION console presets."""

import json
import pathlib
import typing

import core.paths

_PRESETS_DEFAULT_PATH: typing.Final = (
    core.paths.USER_DIRECTORY / "resources" / "presets.default.json"
)

PRESETS_PATH: typing.Final = core.paths.USER_DIRECTORY / "resources" / "presets.json"


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
        **_read_presets_json(_PRESETS_DEFAULT_PATH),
        **_read_presets_json(PRESETS_PATH),
    }


def ensure_presets() -> None:
    """Create an empty user presets file next to the defaults if it is absent."""
    if not PRESETS_PATH.exists():
        PRESETS_PATH.parent.mkdir(parents=True, exist_ok=True)
        PRESETS_PATH.write_text("{}\n", encoding="utf-8")
